# Nghiên cứu lý thuyết PyTorch cơ bản

Tài liệu tóm tắt các khái niệm PyTorch cần cho Practice 1 (phân loại ảnh FashionMNIST) và chỉ ra mỗi khái niệm được dùng ở đâu trong project. Ví dụ chạy thử nằm trong `notebooks/01_pytorch_basics.ipynb`.

## 1. Tensor

**Khái niệm.** Tensor là cấu trúc dữ liệu cốt lõi của PyTorch, tương tự mảng nhiều chiều của NumPy nhưng có thể chạy trên GPU và có khả năng tự động tính đạo hàm.

| Thuộc tính | Ý nghĩa | Ví dụ trong project |
|---|---|---|
| `shape` | Kích thước từng chiều | Một ảnh FashionMNIST: `(1, 28, 28)`; một batch: `(64, 1, 28, 28)` |
| `dtype` | Kiểu dữ liệu | Ảnh là `torch.float32`; nhãn là `torch.int64` (`torch.long`) vì `CrossEntropyLoss` yêu cầu nhãn dạng chỉ số lớp |
| `device` | Nơi lưu tensor (CPU/GPU) | `get_device()` trong `src/config.py` chọn thiết bị; `images.to(device)` trong vòng lặp huấn luyện |

**Các thao tác thường dùng:** tạo (`torch.tensor`, `torch.zeros`, `torch.randn`), đổi hình dạng (`view`, `reshape`, `flatten`), nối (`torch.cat`), phép toán theo phần tử, nhân ma trận (`@`), rút gọn (`sum`, `mean`, `argmax`), chuyển đổi từ/sang NumPy (`from_numpy`, `.numpy()`).

**Quy ước chiều của ảnh:** PyTorch dùng `(N, C, H, W)` (batch, kênh, cao, rộng). `transforms.ToTensor()` đổi ảnh PIL `H × W` thành tensor `C × H × W`, nên một ảnh xám 28 × 28 trở thành `(1, 28, 28)`.

## 2. Autograd (tự động tính đạo hàm)

- Tensor có `requires_grad=True` sẽ được PyTorch ghi lại các phép toán để dựng đồ thị tính toán. Tham số của `nn.Module` mặc định đã bật thuộc tính này.
- `loss.backward()` lan truyền ngược trên đồ thị và lưu đạo hàm vào `param.grad`.
- Gradient **được cộng dồn** giữa các lần gọi `backward()`, vì vậy cần `optimizer.zero_grad()` trước mỗi bước (xem `train_one_epoch` trong `src/training/trainer.py`).
- `torch.no_grad()` tắt việc ghi đồ thị, dùng khi đánh giá để tiết kiệm bộ nhớ và thời gian (xem `evaluate` trong `src/evaluation/metrics.py`).

## 3. Dataset và DataLoader

- **`Dataset`** quản lý mẫu và nhãn qua `__len__` và `__getitem__`. `torchvision.datasets.FashionMNIST` là một Dataset dựng sẵn (60.000 ảnh train, 10.000 ảnh test).
- **`DataLoader`** gom mẫu thành batch, có thể xáo trộn (`shuffle=True`) và nạp dữ liệu song song (`num_workers`). Project dùng `batch_size=64`, `shuffle=True` cho tập train và `shuffle=False` cho tập test; `num_workers=0` để chạy ổn định trên mọi hệ điều hành.
- `random_split` được dùng trong `experiments/tune_hyperparams.py` để tách 10% tập train làm validation với seed cố định.

## 4. Transforms

Chi tiết ở `docs/03_research/data_preprocessing.md`. Tóm tắt: `ToTensor()` rồi `Normalize((0.5,), (0.5,))` để đưa pixel về `[-1, 1]`.

## 5. Xây dựng mô hình với `nn.Module`

- Mô hình là lớp con của `nn.Module`: khai báo các lớp trong `__init__`, mô tả luồng tính toán trong `forward`.
- `nn.Sequential` xếp các lớp theo thứ tự; `nn.Flatten`, `nn.Linear`, `nn.Conv2d`, `nn.MaxPool2d`, `nn.ReLU`, `nn.Dropout` là các lớp dùng trong project.
- Mô hình trả về **logits** (chưa qua softmax) có hình `(N, 10)`.
- `model.train()` bật dropout; `model.eval()` tắt dropout khi đánh giá.

Ba kiến trúc của project (`src/models/neural_network.py`): `MLP` (hai lớp ẩn 256 và 128 nơ-ron), `CNN` (hai khối Conv + MaxPool, rồi Linear 128 + Dropout) và `DeepCNN` (ba khối Conv + MaxPool).

## 6. Hàm mất mát và optimizer

- `nn.CrossEntropyLoss()` nhận logits và nhãn nguyên; bên trong đã gồm `LogSoftmax` nên mô hình **không** thêm softmax ở cuối.
- Optimizer cập nhật tham số theo gradient: `Adam` (tự điều chỉnh tốc độ học theo từng tham số) và `SGD` có momentum. Hyperparameters được phân tích ở `docs/03_research/hyperparameters.md`.

## 7. Vòng lặp huấn luyện

Mỗi batch gồm bốn bước: **Forward** (`logits = model(images)`), **Loss** (`loss_fn(logits, labels)`), **Backward** (`loss.backward()`), **Optimize** (`optimizer.step()`), có `optimizer.zero_grad()` đặt trước Backward. Sau mỗi epoch, mô hình được đánh giá bằng `evaluate()` ở chế độ `eval()` và `no_grad()`.

## 8. Lưu và tải mô hình

- Cách khuyến nghị là lưu **`state_dict`** (chỉ trọng số) bằng `torch.save(model.state_dict(), path)`, rồi tải bằng `model.load_state_dict(torch.load(path))` vào một mô hình **cùng kiến trúc**.
- `torch.load(..., weights_only=True)` chỉ cho phép nạp dữ liệu tensor, tránh thực thi mã tùy ý trong file; `load_model` trong `src/models/model_io.py` dùng chế độ này.
- Kiểm tra đúng đắn: accuracy trên tập test của mô hình tải lại phải bằng đúng accuracy trước khi lưu (được kiểm tra trong `tests/test_model_io.py` và notebook 02).

## Nguồn tham khảo

- PyTorch Tutorials, Learn the Basics: https://pytorch.org/tutorials/beginner/basics/intro.html
- PyTorch Docs, `torch.Tensor`: https://pytorch.org/docs/stable/tensors.html
- PyTorch Docs, Autograd mechanics: https://pytorch.org/docs/stable/notes/autograd.html
- PyTorch Docs, `torch.utils.data`: https://pytorch.org/docs/stable/data.html
- PyTorch Docs, `torch.nn`: https://pytorch.org/docs/stable/nn.html
- PyTorch Tutorials, Save and Load the Model: https://pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html
