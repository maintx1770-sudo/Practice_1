# Kết luận – Practice 1: PyTorch FashionMNIST Classification

## 1. Những gì đã thực hiện
Nhóm xây dựng pipeline phân loại ảnh FashionMNIST (10 lớp, ảnh xám 28 × 28) bằng PyTorch: tải dữ liệu và biến đổi, ba kiến trúc mạng (MLP, CNN, DeepCNN), vòng lặp huấn luyện (Forward → Loss → Backward → Optimize), đánh giá, lưu/tải checkpoint, trực quan hóa loss và ảnh dự đoán, cùng các thử nghiệm kiến trúc và hyperparameters. Toàn bộ chạy được trên CPU và có 34 test tự động.

## 2. Kết quả chính

| Nội dung | Kết quả |
|---|---|
| Mô hình chính (CNN, Adam, lr 1e-3, batch 64, 10 epoch) | **91,93%** accuracy trên tập test, test loss 0,2524 |
| So sánh kiến trúc (5 epoch, cùng điều kiện) | MLP 87,90%; CNN 91,00%; DeepCNN 90,83% |
| Cấu hình tốt nhất chọn theo validation (`TUNE-EPOCH-LONG`, 6 epoch, huấn luyện trên 54.000 ảnh) | 91,60% validation; 90,93% test |
| Lưu và tải mô hình | Accuracy sau khi tải khớp tuyệt đối với trước khi lưu |
| Kiểm thử | 34/34 test tự động PASS; 24/24 ca kiểm thử PASS |

## 3. Nhận xét
1. **Kiến trúc có ảnh hưởng lớn nhất.** CNN cao hơn MLP khoảng 3 điểm phần trăm vì khai thác được cấu trúc không gian của ảnh. DeepCNN không cải thiện thêm rõ rệt so với CNN (chênh 0,17 điểm, nhỏ hơn dao động ngẫu nhiên của một lần chạy).
2. **Số epoch và learning rate là hai hyperparameter quan trọng nhất ở giai đoạn đầu huấn luyện.** Tăng từ 1 lên 3 và 6 epoch cải thiện validation accuracy từ 87,87% lên 90,58% và 91,60%. Learning rate 3e-4 học chậm hơn rõ (89,32%), còn 1e-3 và 3e-3 gần như ngang nhau.
3. **Batch size, optimizer và dropout ảnh hưởng nhỏ** trong 3 epoch (chênh dưới 1 điểm). Với số epoch ít, dropout làm chậm học hơn là chống overfitting.
4. **Overfitting bắt đầu xuất hiện sau khoảng epoch 7** ở lần chạy 10 epoch: train accuracy tăng đến 95,60% trong khi test loss ngừng giảm rồi tăng nhẹ. Khoảng cách train–test là 3,67 điểm.
5. **Các lớp bị nhầm chủ yếu là các loại áo.** Shirt có F1 thấp nhất (0,7605) và hay bị nhầm với T-shirt/top, Pullover và Coat. Các lớp Trouser, Bag, Sandal gần như hoàn hảo (F1 trên 0,98).

## 4. Hạn chế
- Mỗi cấu hình chỉ chạy một lần với một seed; chênh lệch dưới khoảng 0,5 điểm phần trăm chưa đủ cơ sở để kết luận.
- Các thử nghiệm hyperparameters chỉ chạy 3 epoch (riêng một cấu hình 6 epoch) do giới hạn thời gian chạy trên CPU, nên mô hình chưa hội tụ hoàn toàn và so sánh dropout chưa phản ánh đúng tác dụng chống overfitting khi huấn luyện lâu.
- Chưa thử weight decay, data augmentation, learning rate scheduler hay dừng sớm bằng validation.
- `TUNE-EPOCH-LONG` thắng chủ yếu nhờ nhiều epoch hơn, nên không so sánh công bằng với các cấu hình 3 epoch.

## 5. Hướng cải thiện
- Huấn luyện nhiều epoch hơn và dùng dừng sớm theo validation để tránh overfitting từ epoch 7.
- Lặp lại các thử nghiệm với nhiều seed rồi báo cáo trung bình và độ lệch chuẩn.
- Thêm data augmentation nhẹ (lật ngang, dịch nhỏ) và weight decay để giảm khoảng cách train–test.
- Thử learning rate scheduler và các giá trị learning rate khác cho SGD.
- Đào sâu cặp lớp khó (Shirt, T-shirt/top, Coat, Pullover) bằng cách xem các ảnh bị dự đoán sai trong `data/output/predictions/`.
