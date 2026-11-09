# Phương pháp đánh giá mô hình

Tài liệu mô tả cách project đo chất lượng mô hình phân loại FashionMNIST và cách đọc kết quả. Số liệu thực tế nằm ở `docs/06_results/evaluation_results.md` và `docs/06_results/experiment_results.md`.

## 1. Các tập dữ liệu và vai trò

| Tập | Kích thước | Dùng để | Không dùng để |
|---|---|---|---|
| Train | 60.000 (hoặc 54.000 khi có validation) | Cập nhật trọng số | Báo cáo chất lượng cuối |
| Validation | 6.000 (10% tập train, tách bằng seed cố định) | Chọn hyperparameters, chọn epoch tốt nhất | Cập nhật trọng số |
| Test | 10.000 | Đánh giá cuối cùng của cấu hình đã chọn | Chọn hyperparameters |

Nguyên tắc: **mọi quyết định chọn cấu hình dùng validation; tập test chỉ chạm vào sau khi đã chọn xong**. Nếu chọn theo tập test thì accuracy báo cáo bị lạc quan (rò rỉ thông tin). Lưu ý: hàm `fit()` trong `src/training/trainer.py` có tính thêm loss/accuracy trên loader được truyền vào sau mỗi epoch. Ở `tune_hyperparams.py`, loader đó là validation; ở `compare_models.py` và notebook 02, đó là tập test và chỉ dùng để quan sát đường học, không dùng để chọn lại cấu hình.

## 2. Chỉ số

### 2.1 Loss (cross-entropy)
`nn.CrossEntropyLoss` trên logits. Với một mẫu có nhãn đúng `y`, loss là `-log(softmax(logits)[y])`; loss trung bình càng thấp thì mô hình càng chắc chắn vào lớp đúng. Loss nhạy hơn accuracy với mức độ "tự tin" của dự đoán, nên là chỉ số chính để theo dõi quá trình học.

### 2.2 Accuracy
`accuracy = số dự đoán đúng / tổng số mẫu`, dự đoán là `argmax` của logits. FashionMNIST cân bằng giữa 10 lớp nên accuracy là chỉ số tổng quát hợp lý; mức đoán ngẫu nhiên là 10%.

### 2.3 Precision, recall, F1 theo lớp
Tính từ confusion matrix `C` (hàng = nhãn thật, cột = nhãn dự đoán):

- precision của lớp k = `C[k,k] / tổng cột k` (trong số ảnh được dự đoán là k, bao nhiêu đúng);
- recall của lớp k = `C[k,k] / tổng hàng k` (trong số ảnh thật sự là k, bao nhiêu được tìm ra);
- F1 = trung bình điều hòa của precision và recall.

Các chỉ số này cho thấy lớp nào bị nhầm nhiều mà accuracy tổng không thể hiện (thường là các lớp hình dạng gần nhau như áo thun, áo sơ mi, áo chui đầu, áo khoác).

### 2.4 Confusion matrix
Ma trận `10 × 10` đếm số ảnh theo cặp (nhãn thật, nhãn dự đoán). Đường chéo là dự đoán đúng; các ô ngoài đường chéo cho biết cặp lớp bị nhầm. Notebook 02 tính trực tiếp bằng tensor từ `predict_all()`.

## 3. Cách project tính

| Hàm | File | Việc làm |
|---|---|---|
| `evaluate(model, loader, loss_fn, device)` | `src/evaluation/metrics.py` | Chuyển mô hình sang `eval()`, dùng `torch.no_grad()`, trả `{'loss', 'accuracy'}` trên toàn bộ loader rồi trả lại chế độ cũ |
| `predict_all(model, loader, device)` | `src/evaluation/metrics.py` | Trả (nhãn thật, nhãn dự đoán) để dựng confusion matrix |
| `fit(...)` | `src/training/trainer.py` | Ghi lại history theo epoch: `train_loss`, `train_accuracy`, và `test_*` (hoặc `val_*`) |

Các kiểm tra tự động: `tests/test_metrics.py` xác nhận accuracy nằm trong `[0, 1]`, `evaluate` không đổi trọng số và không đổi chế độ train/eval của mô hình.

## 4. Cách đọc đường học (loss/accuracy theo epoch)

| Quan sát | Diễn giải | Hướng xử lý |
|---|---|---|
| Train và validation loss cùng giảm, accuracy gần nhau | Học tốt | Giữ cấu hình |
| Train accuracy cao hơn validation/test nhiều; validation loss ngừng giảm hoặc tăng | Overfitting | Tăng dropout, giảm kích thước mô hình, dừng sớm, giảm số epoch |
| Cả hai accuracy thấp, loss giảm chậm | Underfitting | Tăng learning rate, thêm lớp hoặc nơ-ron, tăng epoch |
| Loss dao động mạnh hoặc không giảm | Learning rate quá lớn | Giảm learning rate |

## 5. Tính tái lập và giới hạn

- Cố định `RANDOM_SEED = 42` qua `set_seed()` và dùng cùng seed cho mọi cấu hình khi so sánh để chênh lệch đến từ cấu hình chứ không phải từ khởi tạo ngẫu nhiên.
- Mỗi cấu hình chỉ chạy **một lần với một seed**, nên chênh lệch nhỏ (khoảng vài phần mười điểm phần trăm) có thể nằm trong dao động ngẫu nhiên và không nên coi là khác biệt chắc chắn.
- Số epoch trong các thử nghiệm bị giới hạn bởi thời gian chạy trên CPU, vì vậy kết quả phản ánh mức độ hội tụ ở giai đoạn đầu, không phải giới hạn tối đa của từng kiến trúc.

## Nguồn tham khảo

- PyTorch Docs, `torch.nn.CrossEntropyLoss`: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
- PyTorch Tutorials, Optimizing Model Parameters: https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
- scikit-learn, Model evaluation (precision, recall, F1, confusion matrix): https://scikit-learn.org/stable/modules/model_evaluation.html
