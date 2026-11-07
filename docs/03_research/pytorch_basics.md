# Nghiên cứu lý thuyết PyTorch Cơ Bản
## 1. Tensor trong PyTorch
- **Khái niệm:** Tensor là cấu trúc dữ liệu cốt lõi trong PyTorch, tương tự như mảng multidimensional của NumPy nhưng hỗ trợ tính toán trên GPU.
- **Tính chất:** Lưu trữ dữ liệu dạng ma trận (VD: tensor 1x28x28 cho ảnh xám FashionMNIST).

## 2. Tự động tính đạo hàm (Autograd)
- `requires_grad=True`: Đánh dấu tensor cần theo dõi các thao tác để tính đạo hàm.
- `backward()`: Thực hiện lan truyền ngược tính gradient tự động cho các tham số.

## 3. Dataset và DataLoader
- **Dataset:** Quản lý tập dữ liệu mẫu và các nhãn tương ứng (`__len__`, `__getitem__`).
- **DataLoader:** Chia nhỏ dữ liệu thành các Batch (`batch_size=64`), hỗ trợ xáo trộn (`shuffle=True`) và nạp dữ liệu đa tiến trình (`num_workers`).