
Mọi thay đổi quan trọng của dự án Practice 1 (PyTorch FashionMNIST Classification) được ghi lại tại đây.
Định dạng tham khảo [Keep a Changelog](https://keepachangelog.com/). Các phiên bản tương ứng với các mốc trong kế hoạch dự án (03/11 – 11/11/2026).

> **Lưu ý (09/11/2026):** các phiên bản 0.4.0 đến 1.0.0 bên dưới được ghi theo kế hoạch ngày; trạng thái thực tế đến ngày 09/11 xem mục `[Unreleased]`. Chỉ đổi thành phiên bản chính thức khi cả nhóm đã kiểm tra và bàn giao.

## [Unreleased] - 2026-11-09

### Added
- Chạy thật toàn bộ pipeline trên CPU: `experiments/compare_models.py --epochs 5` (`results/tables/model_comparison*.csv`), `experiments/tune_hyperparams.py --epochs 3` với đủ 11 cấu hình (`results/tables/hyperparameter_results.csv`, `tune-*_history.csv`, `results/figures/tune-*_history.png`).
- `notebooks/02_fashionmnist_training.ipynb` đã chạy (có output; `results/tables/training_history.csv`, `class_report.csv`); `notebooks/03_practice1_full.ipynb` được viết và chạy đầy đủ.
- `docs/06_results/evaluation_results.md`, `experiment_results.md`, `conclusion.md`; `docs/05_testing/test_report.md` (34/34 test, 24/24 TC PASS); `docs/03_research/evaluation_methods.md`.
- `results/reports/practice1_report.md`: báo cáo ngắn của bài thực hành.
- `tests/test_metrics.py`: thêm `test_accuracy_perfect`, `test_accuracy_wrong` (TC-16, TC-17).

### Changed
- `docs/03_research/pytorch_basics.md`, `data_preprocessing.md`: mở rộng nội dung và thêm nguồn tham khảo.
- `docs/05_testing/test_cases.md`, `test_plan.md`: cập nhật trạng thái sau khi chạy thực tế.
- `README.md`: cập nhật trạng thái và cách chạy.

### Fixed
- `src/models/neural_network.py`: bỏ `from torch import dropout` không dùng (bị tham số `dropout` che mất).

### Notes
- Dữ liệu FashionMNIST (`data/input/`) và checkpoint `.pth` không nên đưa lên Git (đã có trong `.gitignore`); nếu đã lỡ commit, gỡ khỏi index bằng `git rm -r --cached data/input data/output/checkpoints`.

## [1.0.0] - 2026-11-11

Bàn giao dự án.

### Added
- `results/reports/`: báo cáo cuối của bài thực hành (Trúc Chi).
- `results/figures/`, `results/tables/`: biểu đồ loss, ảnh dự đoán so với nhãn thật, bảng lịch sử huấn luyện và bảng so sánh thử nghiệm (Đức).
- `docs/06_results/conclusion.md`: kết luận, mô hình và cấu hình tốt nhất, hạn chế, hướng cải thiện (Trúc Chi).

### Changed
- Kiểm tra toàn bộ cấu trúc thư mục, code, notebook và kết quả; xử lý lỗi còn lại trước khi bàn giao (cả nhóm).

## [0.7.0] - 2026-11-10

Hoàn thành notebook tổng hợp và tài liệu.

### Added
- `notebooks/03_practice1_full.ipynb`: notebook tổng hợp toàn bộ quy trình từ tải dữ liệu đến lưu/tải mô hình (Mai).

### Changed
- `README.md`: cập nhật cách chạy theo quy trình thực tế (Mai).
- `CHANGELOG.md`: cập nhật các phiên bản (Mai).
- Tích hợp và sửa lỗi giữa các module (Nguyễn Chi, My, Huynh, Đức).

## [0.6.0] - 2026-11-09

Có kết quả đánh giá và báo cáo kiểm thử.

### Added
- `docs/06_results/evaluation_results.md`: kết quả đánh giá mô hình cuối cùng (Huynh).
- `docs/06_results/experiment_results.md`: tổng hợp thử nghiệm kiến trúc và hyperparameters (Đức).
- `docs/05_testing/test_report.md`: kết quả chạy `pytest -q` (Trúc Chi).

### Fixed
- Sửa các lỗi phát hiện khi chạy toàn bộ bộ test (Trúc Chi và các thành viên liên quan).

## [0.5.0] - 2026-11-08

Có checkpoint, history và kết quả thực nghiệm.

### Added
- `experiments/compare_models.py`: so sánh các kiến trúc mạng (My).
- `experiments/tune_hyperparams.py`: thử nghiệm learning rate, batch size, epoch, optimizer, dropout (Đức).
- `docs/04_design/processing_pipeline.md`, `docs/04_design/system_architecture.md`: thiết kế luồng xử lý và vai trò các module (Trúc Chi).
- Kết quả chạy thật của `plot_loss.py` và `show_predictions.py` trong `results/figures/`, `data/output/predictions/` (Đức).

## [0.4.0] - 2026-11-07

Chạy được mô hình huấn luyện.

### Added
- `notebooks/02_fashionmnist_training.ipynb`: huấn luyện và đánh giá mô hình; sinh history và checkpoint thật (Huynh).
- `notebooks/01_pytorch_basics.ipynb`: làm quen PyTorch và khám phá dữ liệu (Nguyễn Chi).
- Checkpoint mô hình trong `data/output/checkpoints/` (My, Huynh).

## [0.3.0] - 2026-11-06

Hoàn thành các module chính và bộ test cơ bản.

### Added
- `src/preprocessing/dataset.py`: tải FashionMNIST và tạo DataLoader (Nguyễn Chi).
- `src/models/model_io.py`: lưu và tải mô hình (My).
- `src/training/trainer.py`: vòng lặp huấn luyện Forward → Loss → Backward → Optimize (Huynh).
- `src/visualization/show_predictions.py`: hiển thị ảnh dự đoán so với nhãn thật (Đức).
- `tests/test_transforms.py`, `tests/test_dataset.py`, `tests/test_model.py`, `tests/test_model_io.py`, `tests/test_trainer.py`, `tests/test_metrics.py`.
- `docs/05_testing/test_plan.md`, `docs/05_testing/test_cases.md` (Trúc Chi).

## [0.2.0] - 2026-11-04

Hoàn thành các module nền tảng.

### Added
- `src/preprocessing/transforms.py`: các phép biến đổi ảnh (Nguyễn Chi).
- `src/models/neural_network.py`: kiến trúc mạng nơ-ron (My).
- `src/evaluation/metrics.py`: tính loss và accuracy (Huynh).
- `src/visualization/plot_loss.py`: vẽ biểu đồ loss (Đức).
- `tests/conftest.py`: fixture dùng chung cho pytest (Trúc Chi).
- `docs/01_project_management/`: `project_plan.md`, `task_list.md`, `task_guide.md`, `timeline.md` (Mai).
- `docs/03_research/`: `pytorch_basics.md`, `data_preprocessing.md`, `hyperparameters.md`, `evaluation_methods.md`.

## [0.1.0] - 2026-11-03

Thiết lập nền tảng dự án (Mai).

### Added
- `src/config.py`: đường dẫn dự án, hằng số FashionMNIST (kích thước ảnh, số lớp, tên 10 lớp, mean/std chuẩn hóa), hyperparameters mặc định (`BATCH_SIZE`, `LEARNING_RATE`, `EPOCHS`, `NUM_WORKERS`), `RANDOM_SEED`, tên checkpoint mặc định và các hàm `ensure_dirs()`, `set_seed()`, `get_device()`.
- `requirements.txt`: torch, torchvision, numpy, pandas, matplotlib, tensorboard, scikit-learn, pytest, jupyter.
- `docs/02_requirements/lab_requirements.md`: chuyển yêu cầu slide Practice 1 (Learn, Implement, Tasks, Deliver) thành checklist.
- `docs/02_requirements/experiment_requirements.md`: yêu cầu ghi nhận thử nghiệm kiến trúc và hyperparameters.
- `README.md`: mục tiêu, dữ liệu, cấu trúc thư mục, cài đặt, cách chạy, sản phẩm nộp, nhóm thực hiện.
- `LICENSE`: giấy phép MIT.
- `CHANGELOG.md`: nhật ký thay đổi (file này).

### Changed
- `.gitignore`: bỏ qua cache Python, môi trường ảo, dữ liệu tải về (`data/input/*`), checkpoint, log TensorBoard, kết quả dự đoán và file `.pt`/`.pth`; giữ lại `.gitkeep`.

## [0.0.1] - 2026-11-02

### Added
- Khởi tạo repo và cấu trúc thư mục: `data/`, `notebooks/`, `src/`, `experiments/`, `tests/`, `docs/`, `results/` cùng các file khung (rỗng).
- `Hướng Dẫn Practice_1.md`: hướng dẫn cấu trúc thư mục, bảng phân công, kế hoạch thực hiện chi tiết và các mốc (03/11 – 11/11/2026).
