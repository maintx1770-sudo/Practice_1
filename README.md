# Practice 1 – PyTorch FashionMNIST Classification

Bài thực hành xây dựng mô hình học sâu bằng PyTorch để phân loại ảnh thời trang FashionMNIST thành 10 lớp.

> **Trạng thái (cập nhật 09/11/2026):** mã nguồn, thử nghiệm, tài liệu, báo cáo ngắn (`results/reports/practice1_report.md`) và 3 notebook đã hoàn thành và chạy thật; 34/34 test PASS. Kết quả chính: CNN đạt **91,93%** accuracy trên tập test (10 epoch). Việc còn lại trước hạn 11/11: cả nhóm kiểm tra lần cuối và xác nhận file đã lên GitHub.

## 1. Mục tiêu

Theo Practice 1:
- **Learn:** PyTorch Tensors, Datasets/Loaders, Transforms, Model Building, Autograd, Optimization, Model Saving/Loading.
- **Implement:** tải FashionMNIST, áp dụng transforms, xây mạng nơ-ron, viết training loop (forward, loss, backward, optimize), đánh giá accuracy, lưu và tải model.
- **Tasks:** thử nghiệm kiến trúc/hyperparameters, vẽ biểu đồ loss, hiển thị ảnh dự đoán so với nhãn thật.
- **Deliver:** mã Python, báo cáo ngắn, biểu đồ loss, ảnh dự đoán.

## 2. Dữ liệu

FashionMNIST gồm 60.000 ảnh train và 10.000 ảnh test, ảnh grayscale 28 × 28, 10 lớp: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot.

**Không cần tải thủ công.** Dataset được `torchvision.datasets.FashionMNIST` tự tải về `data/input/` ở lần chạy đầu tiên (cần Internet, khoảng 30 MB). Dữ liệu này không được đưa lên Git.

## 3. Cấu trúc thư mục

```text
Practice_1/
│
├── README.md
├── requirements.txt
├── CHANGELOG.md
├── LICENSE
├── .gitignore
│
├── data/
│   ├── input/
│   └── output/
│       ├── checkpoints/
│       ├── predictions/
│       └── tensorboard/
│
├── notebooks/
│   ├── 01_pytorch_basics.ipynb
│   ├── 02_fashionmnist_training.ipynb
│   └── 03_practice1_full.ipynb
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── transforms.py
│   │   └── dataset.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── neural_network.py
│   │   └── model_io.py
│   ├── training/
│   │   ├── __init__.py
│   │   └── trainer.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── metrics.py
│   └── visualization/
│       ├── __init__.py
│       ├── plot_loss.py
│       └── show_predictions.py
│
├── experiments/
│   ├── __init__.py
│   ├── compare_models.py
│   └── tune_hyperparams.py
│
├── tests/
│   ├── conftest.py
│   ├── test_dataset.py
│   ├── test_transforms.py
│   ├── test_model.py
│   ├── test_model_io.py
│   ├── test_trainer.py
│   └── test_metrics.py
│
├── docs/
│   ├── 01_project_management/
│   │   ├── project_plan.md
│   │   ├── task_list.md
│   │   ├── task_guide.md
│   │   └── timeline.md
│   ├── 02_requirements/
│   │   ├── lab_requirements.md
│   │   └── experiment_requirements.md
│   ├── 03_research/
│   │   ├── pytorch_basics.md
│   │   ├── data_preprocessing.md
│   │   ├── hyperparameters.md
│   │   └── evaluation_methods.md
│   ├── 04_design/
│   │   ├── processing_pipeline.md
│   │   └── system_architecture.md
│   ├── 05_testing/
│   │   ├── test_plan.md
│   │   ├── test_cases.md
│   │   └── test_report.md
│   └── 06_results/
│       ├── experiment_results.md
│       ├── evaluation_results.md
│       └── conclusion.md
│
└── results/
    ├── figures/
    ├── tables/
    └── reports/
```


Chi tiết từng thư mục và người phụ trách xem file `Hướng Dẫn Practice_1.md`.

## 4. Cài đặt

Yêu cầu Python 3.9 trở lên.

```bash
git clone https://github.com/maintx1770-sudo/Practice_1.git
cd Practice_1

python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

Phiên bản PyTorch phù hợp phụ thuộc hệ điều hành và phần cứng (CPU/GPU). Nếu cài lỗi, xem hướng dẫn tại https://pytorch.org/get-started/locally/.

## 5. Cách chạy

Chạy mọi lệnh từ thư mục gốc `Practice_1/`.

**Notebook** (cách dễ nhất):
```bash
jupyter notebook
```
Mở thư mục `notebooks/` và chạy theo thứ tự:
1. `01_pytorch_basics.ipynb`: làm quen PyTorch và khám phá dữ liệu.
2. `02_fashionmnist_training.ipynb`: huấn luyện và đánh giá mô hình.
3. `03_practice1_full.ipynb`: tổng hợp toàn bộ quy trình.

**Script thử nghiệm:**
```bash
python -m experiments.compare_models --epochs 5      # so sánh MLP, CNN, DeepCNN (~8 phút trên CPU)
python -m experiments.tune_hyperparams --epochs 3    # tinh chỉnh learning rate, batch size, epoch, optimizer, dropout (~25 phút trên CPU)
```
Kết quả ghi vào `results/tables/` và `results/figures/`; tổng hợp trong `docs/06_results/experiment_results.md`.

**Kiểm thử:**
```bash
pytest -q
```

**Xem log huấn luyện (nếu có dùng TensorBoard):**
```bash
tensorboard --logdir data/output/tensorboard
```

## 6. Sản phẩm cần nộp

| Sản phẩm | Vị trí |
|---|---|
| Mã Python | `src/`, `experiments/`, `notebooks/` |
| Báo cáo ngắn | `results/reports/practice1_report.md` |
| Biểu đồ loss | `results/figures/` |
| Ảnh dự đoán so với nhãn thật | `results/figures/`, `data/output/predictions/` |
| Bảng kết quả thử nghiệm | `results/tables/` |

## 7. Nhóm thực hiện

| Thành viên | Vai trò |
|---|---|
| Mai | Quản lý và tích hợp |
| Nguyễn Chi | Dữ liệu và kiến thức nền |
| My | Mô hình |
| Huynh | Huấn luyện và đánh giá |
| Đức | Thử nghiệm và trực quan hóa |
| Trúc Chi | Thiết kế, kiểm thử và báo cáo |

## 8. Tài liệu tham khảo

- PyTorch tutorials: https://pytorch.org/tutorials/
- PyTorch docs: https://pytorch.org/docs/
- FashionMNIST (torchvision): https://pytorch.org/vision/stable/generated/torchvision.datasets.FashionMNIST.html

## 9. Giấy phép

MIT. Xem file `LICENSE`.
