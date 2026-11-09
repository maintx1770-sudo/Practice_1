# Quy trình tiền xử lý dữ liệu FashionMNIST

## 1. Bộ dữ liệu

| Thuộc tính | Giá trị |
|---|---|
| Nguồn | Zalando Research, tải qua `torchvision.datasets.FashionMNIST` |
| Tập train / test | 60.000 / 10.000 ảnh |
| Kích thước ảnh | 28 × 28, một kênh xám, giá trị gốc `0–255` (uint8) |
| Số lớp | 10, cân bằng: mỗi lớp 6.000 ảnh train và 1.000 ảnh test |

Dữ liệu được lưu tại `data/input/` (đường dẫn lấy từ `src/config.py`) và không đưa lên Git. Lần chạy đầu tiên `download=True` sẽ tải file `.gz` về nếu máy chưa có.

## 2. Biến đổi dữ liệu (`src/preprocessing/transforms.py`)

| Bước | Tác dụng | Hình dạng / giá trị đầu ra |
|---|---|---|
| `ToTensor()` | Đổi ảnh PIL (uint8 `0–255`) thành `torch.float32` và chia cho 255; đổi chiều `H × W` thành `C × H × W` | `(1, 28, 28)`, giá trị trong `[0, 1]` |
| `Normalize((0.5,), (0.5,))` | Tính `(x − 0.5) / 0.5` | `(1, 28, 28)`, giá trị trong `[-1, 1]` |

Mean và std (`NORMALIZE_MEAN`, `NORMALIZE_STD`) khai báo một lần trong `src/config.py` và dùng chung cho train và test. Chuẩn hóa về quanh 0 giúp gradient ổn định và mô hình hội tụ nhanh hơn so với dữ liệu thô trong `[0, 255]`.

**Tập train và tập test dùng cùng chuỗi biến đổi**, vì project không dùng data augmentation (không lật, xoay hay cắt ngẫu nhiên). Nếu sau này thêm augmentation thì chỉ áp dụng cho tập train; tập test phải giữ nguyên các phép biến đổi xác định.

## 3. DataLoader (`src/preprocessing/dataset.py`)

| Tham số | Giá trị | Lý do |
|---|---|---|
| `batch_size` | 64 (mặc định, có thể đổi khi thử nghiệm) | Cân bằng giữa tốc độ và độ nhiễu của gradient |
| `shuffle` | `True` cho train, `False` cho test | Xáo trộn để mỗi epoch có thứ tự batch khác nhau; test không cần |
| `num_workers` | 0 | Chạy ổn định trên Windows và trên máy ít nhân CPU |

Một batch có hình `(64, 1, 28, 28)` cho ảnh và `(64,)` cho nhãn (`torch.int64`).

**Tách validation.** Khi tinh chỉnh hyperparameters (`experiments/tune_hyperparams.py`), 10% tập train (6.000 ảnh, chọn bằng `random_split` với seed cố định) được dùng làm validation; 54.000 ảnh còn lại để huấn luyện. Tập test chỉ dùng để đánh giá cuối cùng, nhằm tránh chọn cấu hình dựa trên tập test.

## 4. Kiểm thử

Các bước tiền xử lý được kiểm tra bằng `tests/test_transforms.py` (kiểu, hình dạng, khoảng giá trị) và `tests/test_dataset.py` (kích thước tập, hình dạng batch). Kết quả chạy ghi trong `docs/05_testing/test_report.md`.

## Nguồn tham khảo

- PyTorch Tutorials, Datasets & DataLoaders: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
- PyTorch Tutorials, Transforms: https://pytorch.org/tutorials/beginner/basics/transforms_tutorial.html
- torchvision, `FashionMNIST`: https://pytorch.org/vision/stable/generated/torchvision.datasets.FashionMNIST.html
- Xiao, Rasul, Vollgraf (2017), *Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms*: https://arxiv.org/abs/1708.07747
