# Timeline triển khai – Practice 1

**Thời gian:** 03/11/2026 – 11/11/2026

## 1. Lịch chi tiết

| Ngày | TV | Công việc | Trạng thái |
|---|---|---|---|
| **03/11 (T3)** | Mai | `src/config.py`, các `__init__.py`, `requirements.txt`, `.gitignore`; chốt cấu trúc | Xong |
| | Mai | `lab_requirements.md` | Xong |
| | Đức | `hyperparameters.md` | Trễ|
| | Trúc Chi | viết `conftest.py` | Xong |
| **04/11 (T4)** | Nguyễn Chi | `transforms.py` | |
| | My | `neural_network.py` | |
| | Huynh | `metrics.py` | |
| | Trúc Chi | `tests/conftest.py` | |
| **05/11 (T5)** | Nguyễn Chi | `dataset.py` | |
| | My | `test_model.py`, bắt đầu `model_io.py` | |
| | Huynh | Khung `trainer.py`, chạy thử bằng dữ liệu giả | |
| | Đức | `plot_loss.py` bằng history giả | |
| | Trúc Chi | `test_plan.md`, `test_cases.md` | |
| | Mai | 4 file `docs/01_project_management/`, `experiment_requirements.md` | |
| **06/11 (T6)** | Nguyễn Chi | `test_transforms.py`, `test_dataset.py` | |
| | My | `model_io.py`, `test_model_io.py` | |
| | Huynh | Hoàn chỉnh `trainer.py`, bắt đầu `test_metrics.py`, `test_trainer.py` | |
| | Đức | Khung `show_predictions.py` | |
| **07/11 (T7)** | Huynh | Chạy `02_fashionmnist_training.ipynb`, sinh history và checkpoint thật | |
| | Nguyễn Chi | `01_pytorch_basics.ipynb` | |
| | My | Hỗ trợ kiểm tra tích hợp model và model_io | |
| | Trúc Chi | Kiểm tra các module và test đã xong | |
| **08/11 (CN)** | Đức | `tune_hyperparams.py`; chạy `show_predictions.py`, `plot_loss.py` trên dữ liệu thật | |
| | My | `compare_models.py` | |
| | Huynh | Kiểm tra, xác nhận kết quả huấn luyện | |
| | Trúc Chi | `processing_pipeline.md`, `system_architecture.md` | |
| **09/11 (T2)** | Huynh | `evaluation_results.md` | |
| | Đức | `experiment_results.md` | |
| | Trúc Chi | Chạy `pytest -q`, sửa lỗi, viết `test_report.md` | |
| | Mai | `03_practice1_full.ipynb` | |
| **10/11 (T3)** | Trúc Chi | `conclusion.md` | |
| | Mai | Cập nhật `README.md`, `CHANGELOG.md` | |
| | Nguyễn Chi, My, Huynh, Đức | Hỗ trợ sửa lỗi, kiểm tra kết quả, tích hợp module | |
| **11/11 (T4)** | Trúc Chi | Báo cáo cuối trong `results/reports/` | |
| | Cả nhóm | Kiểm tra toàn bộ project, chạy kiểm tra cuối, xử lý lỗi, xác nhận file đã lên GitHub, bàn giao | |

## 2. Các mốc

| Mốc | Hạn | Kết quả cần đạt |
|---|---|---|
| 1 | 03/11 | Hoàn thành cấu hình và thống nhất cấu trúc |
| 2 | 04/11 | Hoàn thành các module nền tảng |
| 3 | 06/11 | Hoàn thành các module chính và bộ test cơ bản |
| 4 | 07/11 | Chạy được mô hình huấn luyện |
| 5 | 08/11 | Có checkpoint, history và kết quả thực nghiệm |
| 6 | 09/11 | Có kết quả đánh giá và báo cáo test |
| 7 | 10/11 | Hoàn thành notebook tổng hợp và kết luận |
| 8 | 11/11 | Hoàn thành báo cáo và tài liệu dự án |
| 9 | 11/11 | Kiểm tra, nghiệm thu và bàn giao |

## 3. Cách cập nhật

Cột "Trạng thái" ghi *Xong* khi file đã push lên GitHub. Nếu lệch lịch, ghi lý do và ngày dự kiến mới ngay tại dòng đó.
