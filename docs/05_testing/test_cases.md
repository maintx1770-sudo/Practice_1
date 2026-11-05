# Danh sách trường hợp kiểm thử – Practice 1

## 1. Thông tin chung

| Mục | Nội dung |
|---|---|
| Tên dự án | **Practice 1 – PyTorch FashionMNIST Classification** |
| Số trường hợp dự kiến | **24 test case** (TC-01 đến TC-24) |
| Tình trạng | **NOT RUN** – chưa có kết quả thực tế |
| Tài liệu liên quan | `docs/05_testing/test_plan.md` |

## 2. Quy ước

- Mỗi test case có mã `TC-xx` để đối chiếu với test tự động và `test_report.md`.
- Các tên test `test_...` ở dưới là **tên gợi ý**, không khẳng định nhóm đã viết hàm đúng tên đó. Khi code hoàn thiện, đối chiếu tên hàm/API thực tế.
- Dữ liệu giả lấy từ fixture của `tests/conftest.py`. Các ca kiểm tra dữ liệu FashionMNIST thật yêu cầu đã tải dataset hoặc có kết nối Internet trong lần chạy đầu.
- Trạng thái ban đầu của tất cả trường hợp: **NOT RUN**. Chỉ chuyển PASS/FAIL/BLOCKED sau khi chạy thực tế.

## 3. Bảng test case

### Nhóm A — Transforms (`tests/test_transforms.py`)

| Mã | Test case (gợi ý) | Dữ liệu / thao tác | Kết quả mong đợi | Trạng thái |
|---|---|---|---|---|
| TC-01 | `test_to_tensor` | Đưa `sample_pil_image` (L, 28×28) qua `ToTensor()` | `torch.float32`, shape `(1, 28, 28)`, giá trị trong `[0,1]` | NOT RUN |
| TC-02 | `test_normalize` | Áp dụng `ToTensor()` rồi `Normalize((0.5,), (0.5,))` | Shape giữ nguyên, pixel đã được chuẩn hóa; giá trị nằm trong `[-1,1]` | NOT RUN |
| TC-03 | `test_transform_consistency` | Gọi transform hợp lệ trên ảnh grayscale 28×28 | Không phát sinh lỗi; đầu ra phù hợp đầu vào của model | NOT RUN |

### Nhóm B — Dataset / DataLoader (`tests/test_dataset.py`)

| Mã | Test case (gợi ý) | Dữ liệu / thao tác | Kết quả mong đợi | Trạng thái |
|---|---|---|---|---|
| TC-04 | `test_mock_dataset` | Đọc `sample_dataset` | Có 8 mẫu; từng ảnh `(1, 28, 28)`; nhãn hợp lệ từ 0 đến 9 | NOT RUN |
| TC-05 | `test_mock_loader_batch` | Lấy batch từ `sample_loader` | Batch ảnh `(4, 1, 28, 28)`; nhãn shape `(4,)`, kiểu `torch.long` | NOT RUN |
| TC-06 | `test_fashionmnist_split_sizes` | Tải/đọc train và test FashionMNIST thật | Train 60.000 mẫu, test 10.000 mẫu | NOT RUN |
| TC-07 | `test_fashionmnist_sample` | Lấy 1 ảnh sau transform từ tập thật | Tensor ảnh `(1, 28, 28)`; nhãn nguyên thuộc `[0,9]` | NOT RUN |

### Nhóm C — Mô hình (`tests/test_model.py`)

| Mã | Test case (gợi ý) | Dữ liệu / thao tác | Kết quả mong đợi | Trạng thái |
|---|---|---|---|---|
| TC-08 | `test_model_forward_shape` | Truyền `sample_images` vào **mô hình chính thức** của nhóm | Output logits shape `(8, 10)` | NOT RUN |
| TC-09 | `test_model_logits_finite` | Thực hiện forward với ảnh hợp lệ | Logits là số hữu hạn, không có NaN/Inf | NOT RUN |
| TC-10 | `test_model_backward` | Forward → CrossEntropyLoss với `sample_labels` → `backward()` | Tính được gradient hợp lệ cho các tham số cần học | NOT RUN |

### Nhóm D — Save / Load (`tests/test_model_io.py`)

| Mã | Test case (gợi ý) | Dữ liệu / thao tác | Kết quả mong đợi | Trạng thái |
|---|---|---|---|---|
| TC-11 | `test_save_checkpoint` | Lưu trạng thái model vào `checkpoint_path` (thư mục tạm) | File checkpoint được tạo, không ghi vào dữ liệu thực của nhóm | NOT RUN |
| TC-12 | `test_load_checkpoint` | Nạp checkpoint vào model có cùng kiến trúc, đặt hai model ở `eval()` | Tham số khôi phục đúng; output trên cùng batch khớp trong sai số số học cho phép | NOT RUN |

### Nhóm E — Trainer (`tests/test_trainer.py`)

| Mã | Test case (gợi ý) | Dữ liệu / thao tác | Kết quả mong đợi | Trạng thái |
|---|---|---|---|---|
| TC-13 | `test_one_training_step` | `sample_model`, `sample_batch`, criterion, optimizer; chạy một bước huấn luyện | Forward/loss/backward/optimizer chạy không lỗi; loss hữu hạn | NOT RUN |
| TC-14 | `test_optimizer_updates_weights` | Lưu bản sao trọng số; thực hiện bước train; so sánh | Ít nhất một tham số được cập nhật khi gradient khác 0 | NOT RUN |
| TC-15 | `test_training_history` | Chạy huấn luyện tối thiểu 1 epoch trên `sample_loader` (nếu trainer có API này) | Có loss/accuracy theo epoch, không có NaN/Inf; độ dài history đúng số epoch | NOT RUN |

### Nhóm F — Metrics (`tests/test_metrics.py`)

| Mã | Test case (gợi ý) | Dữ liệu / thao tác | Kết quả mong đợi | Trạng thái |
|---|---|---|---|---|
| TC-16 | `test_accuracy_perfect` | Nhãn dự đoán trùng hoàn toàn với nhãn thật | Accuracy = 1.0 (hoặc 100% nếu API dùng đơn vị %) | NOT RUN |
| TC-17 | `test_accuracy_wrong` | Nhãn dự đoán sai hoàn toàn | Accuracy = 0.0 | NOT RUN |
| TC-18 | `test_evaluate_model` | Đánh giá model trên `sample_loader` | Hàm chạy ở chế độ đánh giá, trả về loss hữu hạn và accuracy hợp lệ | NOT RUN |
| TC-19 | `test_no_parameter_update_during_eval` | Lưu trọng số trước/sau đánh giá | Trọng số không đổi sau evaluation | NOT RUN |

### Nhóm G — Tích hợp và sản phẩm đầu ra (kiểm tra thủ công hoặc thêm integration test)

| Mã | Test case | Dữ liệu / thao tác | Kết quả mong đợi | Trạng thái |
|---|---|---|---|---|
| TC-20 | Pipeline train-to-evaluate | Nối transforms, loader, model, trainer, metrics | Chạy qua một lượt nhỏ; không lỗi kích thước/kiểu; loss và accuracy hợp lệ | NOT RUN |
| TC-21 | Kiểm tra biểu đồ loss | Mở output của `plot_loss.py` | Có biểu đồ loss, tiêu đề, nhãn trục và đúng dữ liệu history | NOT RUN |
| TC-22 | Kiểm tra ảnh dự đoán | Mở output của `show_predictions.py` | Có ảnh, nhãn thật (actual) và nhãn dự đoán (predicted) | NOT RUN |
| TC-23 | Notebook tổng hợp | Chạy `notebooks/03_practice1_full.ipynb` theo hướng dẫn nhóm | Thể hiện được pipeline, kết quả, lưu/tải model | NOT RUN |
| TC-24 | Kiểm tra xuất báo cáo | Kiểm tra các đường dẫn `results/` và tài liệu liên quan | File hình/bảng/báo cáo cần nộp tồn tại, mở được | NOT RUN |

## 4. Ví dụ chi tiết một test case

**Mã:** TC-05 — Kiểm tra DataLoader với dữ liệu giả  
**Mục tiêu:** Xác nhận `DataLoader` đưa ra batch đúng hình dạng và kiểu nhãn.  
**Tiền điều kiện:** `pytest`, `torch` đã cài; fixture `sample_loader` trong `conftest.py` hoạt động.  
**Các bước:**

1. Lấy một batch từ `sample_loader` bằng `next(iter(sample_loader))`.
2. Kiểm tra shape ảnh là `(4, 1, 28, 28)`.
3. Kiểm tra shape nhãn là `(4,)`, `dtype == torch.long` và mỗi nhãn thuộc `[0,9]`.

**Kết quả mong đợi:** Các điều kiện trên đúng, test PASS.  
**Kết quả thực tế:** Chưa chạy.  
**Trạng thái:** NOT RUN.

## 5. Các điểm cần đối chiếu với code của nhóm trước khi chạy

1. Hàm trong `src/preprocessing/transforms.py` có tên và giá trị Normalize đúng theo cấu hình không?
2. `src/preprocessing/dataset.py` trả về dataset, DataLoader hay một tuple các DataLoader? Có validation set không?
3. Model trong `src/models/neural_network.py` là MLP hay CNN và nhận input dạng `(N,1,28,28)` không?
4. Các hàm trong `model_io.py`, `trainer.py`, `metrics.py` có chữ ký và định dạng kết quả nào?
5. Accuracy của nhóm biểu diễn dưới dạng số từ 0 đến 1 hay phần trăm từ 0 đến 100? Thống nhất trước khi viết assertion.
6. Các file notebook và `results/` đã có kết quả thực hay vẫn đang là khung?

## 6. Hướng dẫn chạy và cập nhật kết quả

Chạy từ thư mục gốc `Practice_1`:

```powershell
python -m pytest tests -v
```

Sau khi chạy, ghi số test pass/fail/skipped, thông báo lỗi, nguyên nhân và hướng xử lý vào `docs/05_testing/test_report.md`. **Không đánh dấu PASS theo dự kiến khi chưa có log chạy thực tế.**

> Ghi chú: TC-20 đến TC-24 có thể kiểm tra thủ công nếu chưa có automation phù hợp. Những case dùng dataset thật có thể BLOCKED khi chưa tải dữ liệu; không vì vậy mà xem các unit test dùng dữ liệu giả là thất bại.

