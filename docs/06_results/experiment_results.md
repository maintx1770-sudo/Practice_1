# Kết quả thử nghiệm kiến trúc và hyperparameters

Số liệu lấy từ các lần chạy thực tế, đều chạy trên CPU (1 nhân), seed 42, mỗi cấu hình chạy **một lần**. Cách ghi nhận theo `docs/02_requirements/experiment_requirements.md`.

- So sánh kiến trúc: `python -m experiments.compare_models --epochs 5` → `results/tables/model_comparison.csv`, `model_comparison_history.csv`.
- Tinh chỉnh hyperparameters: `python -m experiments.tune_hyperparams --epochs 3` → `results/tables/hyperparameter_results.csv`, từng history `tune-*_history.csv`, biểu đồ `results/figures/tune-*_history.png` và `hyperparameter_comparison.png`.

## 1. So sánh kiến trúc

Cùng điều kiện cho cả ba mô hình: Adam, learning rate 1e-3, batch size 64, 5 epoch, cùng seed, đánh giá trên tập test.

| Mô hình | Tổng tham số | Epoch | Thời gian (giây) | Test loss | Test accuracy |
|---|---|---|---|---|---|
| MLP | 235.146 | 5 | 35.9 | 0.3421 | 87.90% |
| CNN | 421.642 | 5 | 181.0 | 0.2440 | 91.00% |
| DeepCNN | 241.546 | 5 | 229.9 | 0.2470 | 90.83% |

**Nhận xét.**
- CNN và DeepCNN hơn MLP khoảng 3 điểm phần trăm accuracy (91,00% và 90,83% so với 87,90%) và test loss thấp hơn rõ (0,244 và 0,247 so với 0,342). Lớp tích chập tận dụng được cấu trúc không gian của ảnh, trong khi MLP duỗi phẳng ảnh thành vector 784 chiều.
- CNN và DeepCNN chênh nhau 0,17 điểm, nằm trong mức dao động ngẫu nhiên của một lần chạy, nên **không kết luận được mô hình nào tốt hơn**. DeepCNN có ít tham số hơn (241.546 so với 421.642, vì thêm một lần MaxPool nên vector đưa vào lớp Linear nhỏ hơn: 1.152 thay vì 3.136) nhưng chậm hơn (230 giây so với 181 giây) do có thêm một khối tích chập.
- MLP nhanh nhất (36 giây), nhưng accuracy thấp nhất; sau 5 epoch loss vẫn đang giảm ở cả ba mô hình nên kết quả chưa phải giới hạn tối đa.

## 2. Tinh chỉnh hyperparameters

Mô hình CNN, 54.000 ảnh train, 6.000 ảnh validation (tách từ tập train, seed cố định). Mỗi nhóm chỉ đổi một yếu tố so với baseline `TUNE-BASE` (CNN, Adam, lr 1e-3, batch 64, dropout 0,25, 3 epoch). Chỉ số chọn cấu hình là validation accuracy tốt nhất (hòa thì xét validation loss).

| Mã | Mục đích | LR | Batch | Epoch | Optimizer | Dropout | Thời gian (giây) | Train acc (cuối) | Val acc (tốt nhất) | Val loss (tốt nhất) |
|---|---|---|---|---|---|---|---|---|---|---|
| TUNE-BASE | Baseline CNN + Adam | 0.001 | 64 | 3 | adam | 0.25 | 110 | 89.74% | 90.58% | 0.2550 |
| TUNE-LR-LOW | Learning rate thấp | 0.0003 | 64 | 3 | adam | 0.25 | 109 | 87.90% | 89.32% | 0.2994 |
| TUNE-LR-HIGH | Learning rate cao | 0.003 | 64 | 3 | adam | 0.25 | 104 | 90.60% | 90.65% | 0.2525 |
| TUNE-BATCH-32 | Batch size nhỏ | 0.001 | 32 | 3 | adam | 0.25 | 107 | 90.73% | 90.58% | 0.2489 |
| TUNE-BATCH-128 | Batch size lớn | 0.001 | 128 | 3 | adam | 0.25 | 104 | 89.31% | 90.18% | 0.2659 |
| TUNE-EPOCH-SHORT | Ít epoch hơn baseline | 0.001 | 64 | 1 | adam | 0.25 | 31 | 81.66% | 87.87% | 0.3236 |
| TUNE-EPOCH-LONG | Nhiều epoch hơn baseline | 0.001 | 64 | 6 | adam | 0.25 | 225 | 92.49% | 91.60% | 0.2278 |
| TUNE-OPT-SGD | So sánh SGD momentum với Adam | 0.01 | 64 | 3 | sgd | 0.25 | 104 | 88.26% | 89.75% | 0.2888 |
| TUNE-DROP-0 | Không dùng dropout | 0.001 | 64 | 3 | adam | 0 | 120 | 91.25% | 90.95% | 0.2451 |
| TUNE-DROP-20 | Dropout nhẹ | 0.001 | 64 | 3 | adam | 0.2 | 124 | 89.98% | 90.83% | 0.2437 |
| TUNE-DROP-50 | Dropout mạnh | 0.001 | 64 | 3 | adam | 0.5 | 110 | 88.06% | 90.03% | 0.2700 |

Biểu đồ so sánh: `results/figures/hyperparameter_comparison.png`.

### Nhận xét theo từng yếu tố

| Yếu tố | Quan sát | Diễn giải |
|---|---|---|
| Learning rate | 3e-4: 89,32%; 1e-3: 90,58%; 3e-3: 90,65% | lr thấp học chậm nhất (train acc chỉ 87,90% sau 3 epoch, dấu hiệu chưa hội tụ). 1e-3 và 3e-3 gần như bằng nhau (chênh 0,07 điểm), chưa thấy dấu hiệu lr 3e-3 quá lớn |
| Batch size | 32: 90,58%; 64: 90,58%; 128: 90,18% | Batch 128 kém hơn nhẹ vì cùng 3 epoch nhưng ít lần cập nhật trọng số hơn. Batch 32 và 64 bằng nhau về accuracy (val loss của batch 32 thấp hơn chút: 0,2489 so với 0,2550) |
| Số epoch | 1: 87,87%; 3: 90,58%; 6: 91,60% | Tăng epoch cải thiện rõ ràng; ở 6 epoch val loss vẫn đang giảm (0,2278) nên vẫn chưa bão hòa |
| Optimizer | Adam (lr 1e-3): 90,58%; SGD momentum 0,9 (lr 1e-2): 89,75% | Adam hội tụ nhanh hơn trong 3 epoch. So sánh này chỉ thử một learning rate cho SGD nên chưa chứng minh SGD luôn kém hơn |
| Dropout | 0: 90,95%; 0,20: 90,83%; 0,25: 90,58%; 0,50: 90,03% | Với chỉ 3 epoch, mô hình còn đang học dở (train acc 88–91%) nên dropout làm chậm học thay vì chống overfitting. Dropout lớn nhất cho kết quả thấp nhất |

### Cấu hình tốt nhất và đánh giá test
- Tốt nhất theo validation: **`TUNE-EPOCH-LONG`** (val accuracy 91,60%, val loss 0,2278).
- Đánh giá một lần trên tập test: **accuracy 90,93%, loss 0,2511** (`results/tables/best_hyperparameter_summary.json`; checkpoint `data/output/checkpoints/best_tuned_model.pth`; ảnh dự đoán `results/figures/best_tuned_predictions.png`).
- Lưu ý khi đọc: `TUNE-EPOCH-LONG` thắng chủ yếu vì được huấn luyện **6 epoch trong khi các cấu hình còn lại chỉ 3 epoch**, nên đây không phải so sánh công bằng với các nhóm khác. Nếu chỉ xét các cấu hình cùng 3 epoch thì tốt nhất là `TUNE-DROP-0` (90,95%), nhưng chênh lệch giữa nhóm đầu bảng (khoảng 0,3–0,4 điểm) nhỏ so với dao động ngẫu nhiên của một seed.
- Accuracy test (90,93%) thấp hơn validation (91,60%) khoảng 0,7 điểm. Đây là mức chênh bình thường giữa hai tập khác nhau, chưa phải dấu hiệu rò rỉ dữ liệu.

## 3. So sánh với lần chạy 10 epoch trong notebook 02

CNN mặc định (Adam, lr 1e-3, batch 64, dropout 0,25) huấn luyện 10 epoch trên toàn bộ 60.000 ảnh train đạt **91,93% accuracy test** (xem `docs/06_results/evaluation_results.md`), cao hơn 90,93% của `TUNE-EPOCH-LONG` vì được huấn luyện nhiều epoch hơn và trên nhiều dữ liệu hơn (60.000 so với 54.000 ảnh).

## 4. Kết luận từ các thử nghiệm
1. Chuyển từ MLP sang CNN là cải thiện lớn nhất (khoảng +3 điểm); thêm lớp tích chập thứ ba (DeepCNN) không cải thiện thêm rõ rệt.
2. Số epoch và learning rate đủ lớn là hai yếu tố ảnh hưởng mạnh nhất trong 3 epoch đầu; batch size và dropout ảnh hưởng nhỏ.
3. Chưa thử: weight decay, nhiều seed khác nhau, các giá trị learning rate khác cho SGD, và huấn luyện đủ lâu cho dropout phát huy tác dụng. Đây là hướng mở rộng nếu có thêm thời gian.
