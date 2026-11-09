# Báo cáo ngắn – Practice 1: PyTorch FashionMNIST Classification

Thời gian thực hiện: 03/11/2026 – 11/11/2026. Mã nguồn và dữ liệu kết quả nằm trong repository `Practice_1`.

## 1. Mục tiêu
Xây dựng, huấn luyện và đánh giá mô hình mạng nơ-ron bằng PyTorch để phân loại ảnh FashionMNIST thành 10 lớp, theo bốn phần của đề bài: Learn (tìm hiểu), Implement (cài đặt), Tasks (thử nghiệm và trực quan hóa), Deliver (sản phẩm).

## 2. Dữ liệu và tiền xử lý
FashionMNIST có 60.000 ảnh train và 10.000 ảnh test, ảnh xám 28 × 28, 10 lớp cân bằng. Ảnh được đổi sang tensor (`ToTensor`) rồi chuẩn hóa về [-1, 1] (`Normalize((0.5,), (0.5,))`); train và test dùng cùng phép biến đổi. Khi tinh chỉnh hyperparameters, 10% tập train (6.000 ảnh) được tách làm validation; tập test chỉ dùng để đánh giá cuối.

## 3. Mô hình và huấn luyện

| Mô hình | Cấu trúc | Tham số |
|---|---|---|
| MLP | Flatten → Linear 256 → Linear 128 → Linear 10 (ReLU) | 235.146 |
| CNN | 2 × (Conv 3×3 + ReLU + MaxPool) → Linear 128 → Dropout → Linear 10 | 421.642 |
| DeepCNN | 3 × (Conv 3×3 + ReLU + MaxPool) → Linear 128 → Dropout → Linear 10 | 241.546 |

Huấn luyện bằng `CrossEntropyLoss` trên logits, optimizer Adam (learning rate 1e-3), batch size 64, seed 42, trên CPU. Mỗi batch: Forward → Loss → Backward → Optimize. Mô hình được lưu bằng `state_dict` và tải lại bằng `weights_only=True`; accuracy sau khi tải khớp với trước khi lưu.

## 4. Kết quả

**Mô hình chính** (CNN, 10 epoch, notebook 02): accuracy test **91,93%**, test loss 0,2524, train accuracy 95,60%. Test loss thấp nhất ở epoch 7 rồi tăng nhẹ, cho thấy overfitting bắt đầu từ khoảng epoch 7.

**So sánh kiến trúc** (5 epoch, cùng điều kiện):

| Mô hình | Test accuracy | Test loss | Thời gian |
|---|---|---|---|
| MLP | 87,90% | 0,3421 | 36 giây |
| CNN | 91,00% | 0,2440 | 181 giây |
| DeepCNN | 90,83% | 0,2470 | 230 giây |

**Tinh chỉnh hyperparameters** (11 cấu hình, chọn theo validation accuracy):

| Yếu tố | Giá trị thử → validation accuracy |
|---|---|
| Learning rate | 3e-4: 89,32%; 1e-3: 90,58%; 3e-3: 90,65% |
| Batch size | 32: 90,58%; 64: 90,58%; 128: 90,18% |
| Số epoch | 1: 87,87%; 3: 90,58%; 6: 91,60% |
| Optimizer | Adam: 90,58%; SGD momentum: 89,75% |
| Dropout | 0: 90,95%; 0,20: 90,83%; 0,25: 90,58%; 0,50: 90,03% |

Cấu hình tốt nhất theo validation là CNN, Adam, lr 1e-3, batch 64, 6 epoch: **90,93% accuracy test** khi đánh giá một lần.

**Theo lớp:** Trouser, Bag, Sandal gần như hoàn hảo (F1 > 0,98); Shirt khó nhất (F1 0,7605), thường bị nhầm với T-shirt/top, Pullover và Coat.

## 5. Hình và bảng
- Biểu đồ loss: `results/figures/training_history_real.png`, `results/figures/tune-*_history.png`.
- Ảnh dự đoán so với nhãn thật: `results/figures/best_tuned_predictions.png`; bảng đầy đủ 10.000 dự đoán: `data/output/predictions/best_tuned_predictions.csv`.
- So sánh hyperparameters: `results/figures/hyperparameter_comparison.png`.
- Bảng số liệu: `results/tables/` (`model_comparison.csv`, `hyperparameter_results.csv`, `class_report.csv`, `training_history.csv`).

## 6. Kiểm thử
34 test tự động (`pytest -q`) đều PASS; 24/24 ca kiểm thử trong `docs/05_testing/test_cases.md` PASS. Chi tiết ở `docs/05_testing/test_report.md`.

## 7. Kết luận và hạn chế
CNN vượt MLP khoảng 3 điểm phần trăm; thêm lớp tích chập thứ ba không cải thiện rõ rệt. Số epoch và learning rate ảnh hưởng mạnh nhất ở giai đoạn đầu, còn batch size, optimizer và dropout ảnh hưởng nhỏ. Hạn chế: mỗi cấu hình chỉ chạy một lần với một seed, các thử nghiệm hyperparameters chỉ 3 epoch, và cấu hình tốt nhất thắng chủ yếu nhờ nhiều epoch hơn. Chi tiết ở `docs/06_results/experiment_results.md` và `docs/06_results/conclusion.md`.

## 8. Tài liệu tham khảo
PyTorch Tutorials: https://pytorch.org/tutorials/ ; PyTorch Docs: https://pytorch.org/docs/ ; Xiao, Rasul, Vollgraf (2017), Fashion-MNIST: https://arxiv.org/abs/1708.07747
