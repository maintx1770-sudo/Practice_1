# Quy trình xử lý dữ liệu và huấn luyện – Practice 1

## 1. Mục đích

Tài liệu mô tả luồng xử lý thực tế của dự án: từ dữ liệu ảnh FashionMNIST, tiền xử lý, tạo batch, đưa ảnh vào mô hình, tính loss, cập nhật trọng số, đánh giá và lưu kết quả. Quy trình dựa trên các module hiện có trong `src/` và `experiments/`.

## 2. Luồng xử lý tổng quát

Quy trình xử lý của dự án được thực hiện theo thứ tự:

**Dữ liệu FashionMNIST và nhãn → ToTensor và Normalize → Dataset/DataLoader → Mô hình MLP, CNN hoặc DeepCNN → Forward (logits) → Tính CrossEntropyLoss → Backward → Optimizer cập nhật trọng số → Đánh giá Loss/Accuracy → Lưu checkpoint, history, biểu đồ và dự đoán.**

Trong quá trình huấn luyện, các bước Forward → Loss → Backward → Optimize được lặp lại cho từng batch. Hàm `fit()` tổ chức các epoch và có thể gọi đánh giá sau mỗi epoch khi được cung cấp loader đánh giá.

## 3. Các bước xử lý chi tiết

### Bước 1 – Nạp dữ liệu

- File: `src/preprocessing/dataset.py`.
- Hàm `get_fashion_mnist_datasets()` lấy FashionMNIST qua `torchvision.datasets.FashionMNIST`, tự tải dữ liệu vào `data/input/` khi cần.
- Dữ liệu gồm **60.000 ảnh train**, **10.000 ảnh test**, mỗi ảnh **grayscale 28 × 28**, thuộc **10 lớp**.
- Hàm `get_data_loaders()` tạo `train_loader` có `shuffle=True`, `test_loader` có `shuffle=False`.

### Bước 2 – Tiền xử lý ảnh

- File: `src/preprocessing/transforms.py`.
- `ToTensor()` chuyển ảnh thành tensor `float32` có kích thước `(1, 28, 28)`; giá trị ban đầu trong `[0, 1]`.
- `Normalize(mean=(0.5,), std=(0.5,))` chuẩn hóa theo công thức:

  `x_norm = (x - 0.5) / 0.5`

- Giá trị sau chuẩn hóa nằm trong khoảng `[-1, 1]`. Train và test hiện dùng cùng các phép biến đổi xác định, **không có tăng cường dữ liệu ngẫu nhiên**.
- Mỗi batch ảnh có dạng `(B, 1, 28, 28)` và nhãn có dạng `(B,)`, với `B` là batch size (mặc định **64**).

### Bước 3 – Đưa ảnh vào mô hình

- File: `src/models/neural_network.py`.
- Các kiến trúc được cài đặt: `MLP`, `CNN`, `DeepCNN`.
- `MLP` làm phẳng ảnh từ `1×28×28` thành vector **784**, qua các lớp fully connected để tạo **10 logits**.
- `CNN` và `DeepCNN` trích xuất đặc trưng bằng `Conv2d`, `ReLU`, `MaxPool2d`, sau đó phân loại bằng các lớp tuyến tính.
- Đầu ra cho một batch luôn có dạng `(B, 10)`, mỗi cột là điểm số của một lớp. **Không cần gọi Softmax trước `CrossEntropyLoss`.**

### Bước 4 – Huấn luyện và cập nhật trọng số

- File: `src/training/trainer.py`.
- `train_one_epoch()` thực hiện các thao tác cho mỗi batch:

  1. Chuyển ảnh và nhãn sang cùng thiết bị với mô hình (CPU/CUDA/MPS).
  2. `optimizer.zero_grad()` để xóa gradient cũ.
  3. `logits = model(images)` – **Forward**.
  4. `loss = loss_fn(logits, labels)` – tính **CrossEntropyLoss**.
  5. `loss.backward()` – lan truyền ngược để tính gradient.
  6. `optimizer.step()` – cập nhật trọng số.
  7. Cộng dồn loss và số dự đoán đúng để tính chỉ số trung bình.

- `fit()` lặp qua các epoch (cấu hình mặc định trong `src/config.py`: **10 epoch**, learning rate **0.001**). Mỗi epoch trả về `train_loss`, `train_accuracy` và, nếu có loader đánh giá, thêm `test_loss`, `test_accuracy`.
- Tham số `writer` của `fit()` **chỉ ghi TensorBoard khi được truyền vào**; hai script thí nghiệm hiện không mặc định bật ghi TensorBoard.

### Bước 5 – Đánh giá mô hình

- File: `src/evaluation/metrics.py`.
- `evaluate()` chuyển mô hình sang chế độ `eval()`, không tính gradient trong quá trình đánh giá, và trả về loss/accuracy trên toàn loader. Sau đó khôi phục trạng thái train/eval trước khi gọi.
- Accuracy được tính theo tỷ lệ số dự đoán đúng trên tổng số ảnh.
- `predict_all()` thu thập nhãn thật và nhãn dự đoán, phù hợp cho các phân tích như confusion matrix.

### Bước 6 – Lưu và tải mô hình

- File: `src/models/model_io.py`.
- `save_model()` lưu `state_dict` ra file `.pth`, mặc định ở `data/output/checkpoints/`.
- `load_model()` nạp trọng số vào mô hình cùng kiến trúc, chuyển sang thiết bị mong muốn và đặt chế độ `eval()`.

### Bước 7 – Trực quan hóa và xuất kết quả

- `src/visualization/plot_loss.py`: `plot_loss()` tạo biểu đồ loss, có thể kèm accuracy, lưu PNG ở `results/figures/`.
- `src/visualization/show_predictions.py`: `show_predictions()` vẽ lưới ảnh cùng nhãn thật (`True`) và dự đoán (`Pred`); `export_predictions_csv()` xuất bảng dự đoán ở `data/output/predictions/`.
- `results/tables/` lưu các bảng history và so sánh thí nghiệm (CSV).

## 4. Hai luồng thực nghiệm trong dự án

| Luồng | Module điều khiển | Dữ liệu và mục đích |
|---|---|---|
| So sánh kiến trúc | `experiments/compare_models.py` | Huấn luyện `MLP`, `CNN`, `DeepCNN` trong điều kiện chung. Script hiện dùng `test_loader` để theo dõi chỉ số ở mỗi epoch và đánh giá cuối. |
| Điều chỉnh siêu tham số | `experiments/tune_hyperparams.py` | Tách tập train thành **train/validation** (mặc định 90%/10%), so sánh cấu hình bằng validation accuracy, khôi phục trọng số epoch tốt nhất, chọn mô hình tốt nhất rồi đánh giá một lần trên test. |

Ở chế độ nhanh `--quick`, script tuning thử 3 learning rate: `3e-4`, `1e-3`, `3e-3`. Đây là chạy thử nhanh, **không phải toàn bộ** lưới cấu hình của chế độ mặc định.

## 5. Đầu vào, đầu ra và kiểm tra

| Thành phần | Đầu vào | Đầu ra chính |
|---|---|---|
| Dataset / transforms | Ảnh FashionMNIST, nhãn 0–9 | Tensor `(1,28,28)` đã chuẩn hóa |
| DataLoader | Dataset | Batch ảnh `(B,1,28,28)` và nhãn `(B,)` |
| Mô hình | Batch ảnh | Logits `(B,10)` |
| Trainer | Model, loader, loss, optimizer | Trọng số cập nhật; history loss/accuracy |
| Evaluation | Model, loader | `loss`, `accuracy` |
| Model I/O | Model, đường dẫn | Checkpoint `.pth` |
| Visualization | History hoặc model + loader | PNG, CSV dự đoán |

**Đã hoàn thành kiểm tra các module có bài test tự động:** Đã chạy toàn bộ bộ kiểm thử hiện có bằng `python -m pytest tests -q --tb=short` trên máy thành viên. **Kết quả: 32/32 bài test ĐẠT trong 47,61 giây; 0 FAIL, 0 ERROR.** Các module dữ liệu, tiền xử lý, mô hình, lưu/tải mô hình, huấn luyện và đánh giá đều đáp ứng các tiêu chí trong bộ test đã chạy.

## 6. Kết luận

Pipeline xử lý đã được tổ chức thành các module độc lập và có thể tái sử dụng. Dữ liệu được chuẩn hóa thống nhất; mô hình tạo đầu ra 10 lớp; training loop thực hiện đầy đủ Forward → Loss → Backward → Optimize; kết quả được đánh giá, lưu và trực quan hóa. TensorBoard là khả năng tùy chọn, không phải điều kiện để pipeline hiện tại tạo các biểu đồ PNG.
