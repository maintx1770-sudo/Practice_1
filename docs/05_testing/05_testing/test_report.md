# Báo cáo kiểm thử – Practice 1

## 1. Thông tin lần chạy

| Mục | Giá trị |
|---|---|
| Ngày chạy | 09/11/2026 |
| Lệnh | `pytest -q` (chạy từ thư mục gốc dự án) |
| Hệ điều hành | Linux (kernel 6.18, x86_64) |
| Python / PyTorch / torchvision / pytest | 3.12.16 / 2.14.1 (CPU) / 0.29.2 / 9.2.0 |
| Thiết bị | CPU, 1 nhân |
| Commit nền | `8b68616` (cộng các thay đổi chưa commit mô tả ở mục 5) |
| Dữ liệu | FashionMNIST thật trong `data/input/` (60.000 train, 10.000 test) |

Kế hoạch và danh sách ca kiểm thử nằm ở `docs/05_testing/test_plan.md` và `docs/05_testing/test_cases.md`. Nhóm đã chạy trên môi trường Linux, không phải Windows như dự kiến trong kế hoạch; kết quả không phụ thuộc hệ điều hành vì chỉ dùng CPU và dữ liệu giả hoặc FashionMNIST.

## 2. Kết quả tổng hợp

| Chỉ số | Giá trị |
|---|---|
| Số test tự động | 34 |
| Đạt (PASS) | **34** |
| Lỗi (FAIL) / Bỏ qua | 0 / 0 |
| Thời gian chạy | khoảng 3 giây |

| File test | Số test | Kết quả |
|---|---|---|
| `tests/test_transforms.py` | 3 | PASS |
| `tests/test_dataset.py` | 5 | PASS |
| `tests/test_model.py` | 12 (4 nhóm × 3 mô hình MLP/CNN/DeepCNN) | PASS |
| `tests/test_model_io.py` | 4 | PASS |
| `tests/test_trainer.py` | 4 | PASS |
| `tests/test_metrics.py` | 6 | PASS |

## 3. Đối chiếu ca kiểm thử (TC) với test thực tế

| Mã | Test tự động hoặc cách kiểm tra | Kết quả |
|---|---|---|
| TC-01 | `test_transforms.py::test_to_tensor` | PASS |
| TC-02 | `test_transforms.py::test_normalize` | PASS |
| TC-03 | `test_transforms.py::test_transform_consistency` | PASS |
| TC-04 | `test_dataset.py::test_mock_dataset` | PASS |
| TC-05 | `test_dataset.py::test_mock_loader_batch` | PASS |
| TC-06 | `test_dataset.py::test_fashionmnist_split_sizes` | PASS |
| TC-07 | `test_dataset.py::test_fashionmnist_sample` (và `test_get_data_loaders_batch_shapes`) | PASS |
| TC-08 | `test_model.py::test_output_shape[MLP/CNN/DeepCNN]` | PASS |
| TC-09 | `test_model.py::test_output_is_finite[...]` | PASS |
| TC-10 | `test_model.py::test_backward_runs[...]` | PASS |
| TC-11 | `test_model_io.py::test_save_then_load_same_predictions[...]` (kiểm tra file checkpoint tồn tại, lưu vào thư mục tạm) | PASS |
| TC-12 | cùng test trên (so sánh output hai mô hình ở `eval()` với `allclose`, atol = 1e-6); thêm `test_load_missing_file_raises` | PASS |
| TC-13 | `test_trainer.py::test_one_epoch_runs_with_valid_loss` | PASS |
| TC-14 | `test_trainer.py::test_weights_are_updated` | PASS |
| TC-15 | `test_trainer.py::test_fit_history_keys_and_length` (và `test_loss_decreases_on_learnable_data`) | PASS |
| TC-16 | `test_metrics.py::test_accuracy_perfect` (viết thêm ngày 09/11) | PASS |
| TC-17 | `test_metrics.py::test_accuracy_wrong` (viết thêm ngày 09/11) | PASS |
| TC-18 | `test_metrics.py::test_evaluate_returns_valid_loss_and_accuracy` | PASS |
| TC-19 | `test_metrics.py::test_evaluate_does_not_change_weights_or_mode` | PASS |
| TC-20 | Kiểm tra tích hợp: `compare_models.py`, `tune_hyperparams.py` và notebook 03 chạy trọn pipeline transforms → loader → model → trainer → metrics → save/load trên dữ liệu thật, không lỗi kích thước hoặc kiểu dữ liệu | PASS |
| TC-21 | Mở các biểu đồ trong `results/figures/` (ví dụ `tune-epoch-long_history.png`): có tiêu đề, nhãn trục, chú giải, đúng dữ liệu history | PASS |
| TC-22 | Mở `results/figures/best_tuned_predictions.png`: lưới 16 ảnh, mỗi ảnh có nhãn thật (True) và nhãn dự đoán (Pred), đúng màu xanh | PASS |
| TC-23 | Chạy `notebooks/03_practice1_full.ipynb` bằng `jupyter nbconvert --execute`: 12/12 ô code có output, không lỗi; bước lưu/tải khớp accuracy (0,9019 trước và sau khi tải) | PASS |
| TC-24 | Các đường dẫn `results/figures/`, `results/tables/`, `results/reports/` và `docs/` chứa đủ các file cần nộp | PASS |

Tổng: **24/24 ca PASS**.

## 4. Lỗi và vấn đề phát hiện trong quá trình kiểm thử

| # | Phát hiện | Xử lý |
|---|---|---|
| 1 | Lần chạy đầu: 3 test trong `test_dataset.py` lỗi (`RuntimeError: Error downloading train-images-idx3-ubyte.gz`). Repo chỉ chứa file test của FashionMNIST (`t10k-*`), thiếu file train, và môi trường chạy không truy cập được máy chủ tải của torchvision | Đặt hai file train (`train-images-idx3-ubyte.gz`, `train-labels-idx1-ubyte.gz`) vào `data/input/FashionMNIST/raw/`; kiểm tra MD5 khớp giá trị chuẩn của torchvision. Sau đó 32/32 test (số test lúc đó) PASS. Trên máy có Internet, `download=True` tự tải đủ nên không cần làm gì |
| 2 | `src/models/neural_network.py` có `from torch import dropout, nn`: `dropout` là hàm của torch, không dùng và bị tham số `dropout` của lớp che mất | Đổi thành `from torch import nn`; không đổi hành vi, test vẫn PASS |
| 3 | Thiếu test cho TC-16 và TC-17 (accuracy bằng 1,0 và 0,0) | Viết thêm `test_accuracy_perfect` và `test_accuracy_wrong` vào `tests/test_metrics.py` |

## 5. Thay đổi mã được thực hiện khi kiểm thử
- `src/models/neural_network.py`: sửa import (lỗi 2).
- `tests/test_metrics.py`: thêm 2 test (lỗi 3).

## 6. Giới hạn
- Các test tự động dùng dữ liệu giả cho phần lớn ca; chất lượng mô hình (accuracy) không được kiểm bằng unit test mà bằng thử nghiệm ở `docs/06_results/`.
- TC-20 đến TC-24 là kiểm tra tích hợp và thủ công, không có test tự động tương ứng.
- Lần chạy chỉ thực hiện trên CPU và trên một hệ điều hành; chưa kiểm tra trên GPU hoặc Windows.
