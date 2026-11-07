# Quy Trình Tiền Xử Lý Dữ Liệu FashionMNIST
## 1. Biến đổi dữ liệu (Transforms)
- `ToTensor()`: Chuyển đổi ảnh PIL hoặc mảng NumPy (giá trị 0-255) thành `torch.FloatTensor` nằm trong khoảng `[0.0, 1.0]` và đổi kích thước tensor thành `(C x H x W)`.
- `Normalize(mean=(0.5,), std=(0.5,))`: Chuẩn hóa giá trị pixel về dải `[-1.0, 1.0]` giúp mô hình hội tụ nhanh và ổn định hơn trong quá trình huấn luyện.

## 2. Kiểm thử và Xác nhận
- Tất cả các thao tác tiền xử lý đã được kiểm thử thông qua bài unit test trong `tests/test_transforms.py` và `tests/test_dataset.py`.