## 1. Mục đích tài liệu

Tài liệu này trình bày các hyperparameters cần sử dụng và thử nghiệm trong bài Lab 1 – xây dựng mô hình phân loại ảnh FashionMNIST bằng PyTorch.

Các nội dung chính gồm:

- Learning rate.
- Batch size.
- Number of epochs.
- Optimizer.
- Dropout.
- Weight decay.
- So sánh kiến trúc MLP và CNN.
- Quy trình thử nghiệm hyperparameters công bằng.
- Cách ghi nhận kết quả thực nghiệm.

Mục tiêu là chọn được mô hình và cấu hình phù hợp để phân loại 10 lớp ảnh thời trang của FashionMNIST.

---

## 2. Phân biệt model parameters và hyperparameters

Trong deep learning có hai loại giá trị chính.

### 2.1. Model parameters

Model parameters là các giá trị được mô hình tự học trong quá trình huấn luyện, ví dụ:

- Weight của lớp `Linear`.
- Bias của lớp `Linear`.
- Kernel weight của lớp `Conv2d`.

Các giá trị này được cập nhật bằng quá trình:

```text
Forward → Loss → Backward → Optimizer step
