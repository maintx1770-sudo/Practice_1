# Kết quả đánh giá mô hình

Tài liệu ghi kết quả huấn luyện và đánh giá thực tế của mô hình chính (CNN) chạy trong `notebooks/02_fashionmnist_training.ipynb`. Phương pháp đánh giá xem ở `docs/03_research/evaluation_methods.md`; so sánh kiến trúc và hyperparameters xem ở `docs/06_results/experiment_results.md`.

## 1. Cấu hình

| Mục | Giá trị |
|---|---|
| Mô hình | `CNN` (2 khối Conv+MaxPool, Linear 128, Dropout 0,25), 421.642 tham số |
| Optimizer / learning rate | Adam, 1e-3 |
| Batch size / số epoch | 64 / 10 |
| Loss | `CrossEntropyLoss` |
| Dữ liệu | FashionMNIST: 60.000 ảnh train, 10.000 ảnh test, chuẩn hóa về [-1, 1] |
| Seed / thiết bị | 42 / CPU |
| Checkpoint | `data/output/checkpoints/fashion_mnist_model.pth` (lưu rồi tải lại: accuracy khớp tuyệt đối, đã kiểm tra bằng `assert` trong notebook) |

## 2. Kết quả cuối (epoch 10, tập test)

| Chỉ số | Train | Test |
|---|---|---|
| Loss | 0.1182 | 0.2524 |
| Accuracy | 95.60% | **91.93%** |

Khoảng cách train–test accuracy là 3.67 điểm phần trăm.

## 3. Đường học theo epoch

| Epoch | Train loss | Train acc | Test loss | Test acc |
|---|---|---|---|---|
| 1 | 0.4758 | 82.87% | 0.3257 | 88.35% |
| 2 | 0.3024 | 89.02% | 0.2945 | 89.18% |
| 3 | 0.2539 | 90.68% | 0.2617 | 90.53% |
| 4 | 0.2245 | 91.72% | 0.2390 | 91.32% |
| 5 | 0.2000 | 92.70% | 0.2353 | 91.16% |
| 6 | 0.1795 | 93.35% | 0.2382 | 91.74% |
| 7 | 0.1622 | 93.99% | 0.2326 | 92.16% |
| 8 | 0.1461 | 94.48% | 0.2362 | 92.09% |
| 9 | 0.1321 | 95.08% | 0.2419 | 91.69% |
| 10 | 0.1182 | 95.60% | 0.2524 | 91.93% |

Biểu đồ: `results/figures/training_history_real.png` (và biểu đồ vẽ trong notebook 02).

**Nhận xét.**
- Train loss giảm đều qua cả 10 epoch (từ 0.4758 xuống 0.1182), nên quá trình học ổn định, learning rate 1e-3 phù hợp.
- Test loss giảm đến epoch 7 (thấp nhất 0.2326, accuracy 92.16%) rồi tăng nhẹ lên 0.2524 ở epoch 10, trong khi train loss vẫn giảm. Đây là dấu hiệu **bắt đầu overfitting** từ khoảng epoch 7; mức độ còn nhẹ (accuracy test dao động quanh 92% chứ không giảm mạnh).
- Số liệu báo cáo là của epoch cuối (10), không chọn epoch 7 theo tập test, vì chọn epoch theo tập test sẽ làm accuracy báo cáo bị lạc quan. Nếu muốn dừng sớm đúng cách thì phải chọn epoch bằng validation set (như cách làm ở `tune_hyperparams.py`).

## 4. Kết quả theo lớp (tập test)

| Lớp | Precision | Recall | F1 | Số ảnh |
|---|---|---|---|---|
| T-shirt/top | 0.8492 | 0.8670 | 0.8580 | 1000 |
| Trouser | 0.9890 | 0.9880 | 0.9885 | 1000 |
| Pullover | 0.8733 | 0.9030 | 0.8879 | 1000 |
| Dress | 0.9376 | 0.9020 | 0.9195 | 1000 |
| Coat | 0.8962 | 0.8550 | 0.8751 | 1000 |
| Sandal | 0.9840 | 0.9850 | 0.9845 | 1000 |
| Shirt | 0.7542 | 0.7670 | 0.7605 | 1000 |
| Sneaker | 0.9678 | 0.9630 | 0.9654 | 1000 |
| Bag | 0.9783 | 0.9900 | 0.9841 | 1000 |
| Ankle boot | 0.9682 | 0.9730 | 0.9706 | 1000 |
| **Trung bình macro** | 0.9198 | 0.9193 | 0.9194 | 10000 |

**Nhận xét.**
- Dễ nhất: Trouser (F1 0.9885), Bag (0.9841), Sandal (0.9845); hình dạng của các lớp này rất khác các lớp còn lại.
- Khó nhất: **Shirt** (F1 0.7605), tiếp theo là T-shirt/top (0.8580), Coat (0.8751) và Pullover (0.8879).
- Các cặp bị nhầm nhiều nhất (số ảnh tập test): Shirt bị dự đoán thành T-shirt/top (101) và ngược lại T-shirt/top thành Shirt (100); Coat thành Shirt (65); Shirt thành Pullover (55); Coat thành Pullover (54). Đây đều là các loại áo có hình dáng gần nhau ở độ phân giải 28 × 28.

## 5. Kiểm tra lưu/tải mô hình
Mô hình được lưu bằng `save_model`, tải lại bằng `load_model` vào một `CNN()` mới rồi đánh giá lại trên tập test: accuracy trước và sau khi lưu bằng nhau (91.93%).

## 6. Hạn chế
- Mỗi kết quả chỉ chạy một lần với một seed, nên chênh lệch nhỏ giữa các lần chạy chưa được kiểm chứng về độ ổn định.
- Trong notebook 02, tập test được đánh giá sau mỗi epoch chỉ để quan sát đường học, không dùng để chọn cấu hình.
