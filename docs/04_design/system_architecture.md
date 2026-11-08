# Kiến trúc hệ thống – Practice 1

## 1. Tổng quan

Dự án là ứng dụng thực nghiệm học sâu chạy cục bộ bằng **Python và PyTorch** để phân loại ảnh FashionMNIST. Hệ thống chia thành các module **cấu hình, tiền xử lý, mô hình, huấn luyện, đánh giá, lưu/tải, trực quan hóa**; các script trong `experiments/` điều phối việc chạy thí nghiệm. Đây là kiến trúc chương trình Python theo module, **không có API server hoặc cơ sở dữ liệu riêng** trong mã nguồn đã đối chiếu.

## 2. Kiến trúc tổng quát và quan hệ giữa các module

Dự án được tổ chức theo các nhóm chức năng độc lập, cùng được sử dụng bởi các script trong thư mục `experiments/`:

- **Cấu hình:** `src/config.py` lưu đường dẫn, tham số mặc định, tên lớp và thiết bị tính toán.
- **Dữ liệu và tiền xử lý:** `src/preprocessing/dataset.py` nạp FashionMNIST và gọi các hàm chuyển đổi ảnh từ `src/preprocessing/transforms.py`.
- **Mô hình:** `src/models/neural_network.py` định nghĩa MLP, CNN, DeepCNN; `src/models/model_io.py` lưu và tải trọng số.
- **Huấn luyện và đánh giá:** `src/training/trainer.py` thực hiện vòng lặp học và gọi `evaluate()` từ `src/evaluation/metrics.py` khi có dữ liệu đánh giá.
- **Trực quan hóa:** `src/visualization/plot_loss.py` nhận history để tạo biểu đồ; `src/visualization/show_predictions.py` nhận mô hình và DataLoader để tạo ảnh dự đoán, bảng CSV.
- **Điều phối:** `experiments/compare_models.py` và `experiments/tune_hyperparams.py` khởi tạo và kết nối các thành phần trên, sau đó ghi kết quả vào `data/output/` và `results/`.

Đây là các module Python chạy cục bộ, không phải kiến trúc các dịch vụ mạng.

## 3. Vai trò các module `src/`

| Module | Vai trò | Các hàm / lớp chính |
|---|---|---|
| `src/config.py` | Tập trung đường dẫn, lớp nhãn, batch size, learning rate, seed, device | `ensure_dirs()`, `set_seed()`, `get_device()` |
| `src/preprocessing/transforms.py` | Chuyển ảnh sang tensor và chuẩn hóa | `get_train_transforms()`, `get_test_transforms()` |
| `src/preprocessing/dataset.py` | Tải FashionMNIST, tạo train/test loader | `get_fashion_mnist_datasets()`, `get_data_loaders()` |
| `src/models/neural_network.py` | Định nghĩa kiến trúc phân loại | `MLP`, `CNN`, `DeepCNN` |
| `src/models/model_io.py` | Lưu và tải trọng số mô hình | `save_model()`, `load_model()` |
| `src/training/trainer.py` | Cập nhật trọng số theo batch và epoch; theo dõi history | `train_one_epoch()`, `fit()` |
| `src/evaluation/metrics.py` | Đánh giá loss/accuracy và thu thập dự đoán | `accuracy_from_logits()`, `evaluate()`, `predict_all()` |
| `src/visualization/plot_loss.py` | Vẽ biểu đồ loss, accuracy từ history | `plot_loss()`, `plot_training_history()` |
| `src/visualization/show_predictions.py` | Vẽ ảnh dự đoán và xuất bảng CSV | `collect_predictions()`, `show_predictions()`, `export_predictions_csv()` |

## 4. Luồng tương tác giữa các module

1. Script thực nghiệm gọi `get_fashion_mnist_datasets()` hoặc `get_data_loaders()` trong `dataset.py`; module này sử dụng `get_train_transforms()` và `get_test_transforms()` trong `transforms.py`.
2. Script khởi tạo một mô hình từ `neural_network.py`, chọn hàm mất mát `CrossEntropyLoss` và optimizer.
3. Script truyền mô hình, DataLoader và các tham số cho `fit()` trong `trainer.py`.
4. Với mỗi batch, `train_one_epoch()` gọi mô hình để tạo logits, tính loss, lan truyền ngược và cập nhật trọng số bằng optimizer.
5. Khi có loader đánh giá, `fit()` gọi `evaluate()` trong `metrics.py` để lấy loss và accuracy; các kết quả được đưa vào history.
6. Script thực nghiệm sử dụng `model_io.py` để lưu checkpoint, `plot_loss.py` để lưu biểu đồ, và `show_predictions.py` để tạo ảnh hoặc bảng dự đoán; các file CSV/JSON kết quả được lưu ở các thư mục đầu ra tương ứng.

**Quan hệ cốt lõi:** `dataset.py` sử dụng `transforms.py`; `trainer.py` sử dụng `metrics.py`; các script trong `experiments/` điều phối nhiều module để hoàn thành quy trình thử nghiệm.

## 5. Tầng điều phối thực nghiệm

### `experiments/compare_models.py`

- Định nghĩa `MODELS = {"MLP": MLP, "CNN": CNN, "DeepCNN": DeepCNN}`.
- Khởi tạo model, optimizer **Adam** và `CrossEntropyLoss`.
- Chạy `fit()` trên từng model; tính loss/accuracy, thời gian chạy và số parameter.
- Ghi checkpoint `data/output/checkpoints/{model}_compare.pth` cùng CSV so sánh tại `results/tables/`.
- Tham số `--skip-train` chỉ hiển thị thông tin kiến trúc, không train.

### `experiments/tune_hyperparams.py`

- Tách dữ liệu từ tập train thành train/validation theo seed cố định; test giữ riêng.
- Tạo các `ExperimentConfig` để thử learning rate, batch size, epoch, optimizer và dropout.
- Gọi `fit()` từng epoch, ghi history, chọn epoch có validation accuracy cao nhất (nếu hòa thì dùng validation loss thấp hơn).
- Chọn cấu hình tốt nhất theo validation, sau đó mới đánh giá trên test.
- Lưu `best_tuned_model.pth`, `hyperparameter_results.csv`, `best_hyperparameter_summary.json`, các hình history, biểu đồ so sánh và kết quả dự đoán.

## 6. Tổ chức đầu vào và đầu ra

```text
Practice_1/
├── src/
│   ├── config.py
│   ├── preprocessing/  # dataset.py, transforms.py
│   ├── models/         # neural_network.py, model_io.py
│   ├── training/       # trainer.py
│   ├── evaluation/     # metrics.py
│   └── visualization/  # plot_loss.py, show_predictions.py
├── experiments/        # compare_models.py, tune_hyperparams.py
├── tests/              # conftest.py và 6 file test_*.py
├── data/
│   ├── input/          # FashionMNIST
│   └── output/
│       ├── checkpoints/ # .pth
│       ├── predictions/ # .csv
│       └── tensorboard/ # có dữ liệu nếu được ghi log
├── results/
│   ├── tables/         # .csv, .json
│   ├── figures/        # .png
│   └── reports/        # báo cáo cuối
└── docs/04_design/     # Hai tài liệu thiết kế
```

## 7. Thiết kế kiểm thử và khả năng mở rộng

- Các chức năng cốt lõi được kiểm tra qua `tests/test_dataset.py`, `test_transforms.py`, `test_model.py`, `test_model_io.py`, `test_trainer.py`, `test_metrics.py`.
- `tests/conftest.py` cung cấp fixture ảnh/nhãn giả, model mẫu, DataLoader và đường dẫn checkpoint tạm.
- Muốn mở rộng kiến trúc, có thể thêm model trong `src/models/`, sau đó đăng ký vào `MODELS` của script so sánh hoặc `build_model()` của script tuning. Giữ nguyên giao diện đầu vào `(B,1,28,28)` và đầu ra `(B,10)`.

## 8. Kết luận

Kiến trúc module hóa tách biệt quá trình lấy dữ liệu, định nghĩa mô hình, huấn luyện, đánh giá và xuất kết quả. Cách tổ chức này giúp từng thành viên phát triển và kiểm thử riêng, đồng thời tái sử dụng các thành phần chung trong hai script thực nghiệm.
