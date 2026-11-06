# Hyperparameters trong Lab 1 – PyTorch FashionMNIST Classification

## 1. Mục đích

Trong Lab 1, nhóm xây dựng mô hình học sâu bằng PyTorch để phân loại ảnh FashionMNIST thành 10 lớp quần áo và phụ kiện.

Mô hình không chỉ phụ thuộc vào kiến trúc mạng nơ-ron mà còn phụ thuộc vào các hyperparameters. Hyperparameters ảnh hưởng đến tốc độ học, độ ổn định của quá trình train, khả năng tổng quát hóa và độ chính xác của mô hình.

Tài liệu này giải thích các hyperparameters quan trọng cần sử dụng trong project:

- Learning rate.
- Batch size.
- Number of epochs.
- Optimizer.
- Dropout.
- Weight decay.
- Kiến trúc MLP.
- Kiến trúc CNN.
- Cách thử nghiệm và chọn cấu hình tốt nhất.

Mục tiêu là hiểu ý nghĩa của từng hyperparameter, biết cách phát hiện cấu hình chưa phù hợp và có kế hoạch thử nghiệm rõ ràng.

---

## 2. Bài toán FashionMNIST

FashionMNIST là bộ dữ liệu phân loại ảnh thời trang. Mỗi ảnh là ảnh grayscale, có một kênh màu và kích thước 28 × 28 pixels.

| Thuộc tính | Giá trị |
|---|---:|
| Số lượng ảnh train | 60.000 |
| Số lượng ảnh test | 10.000 |
| Kích thước ảnh | 28 × 28 |
| Số kênh ảnh | 1 |
| Số lớp phân loại | 10 |
| Dạng bài toán | Multi-class classification |

Mười nhãn của FashionMNIST:

| Nhãn | Tên lớp |
|---:|---|
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

Dữ liệu đầu vào của mô hình có dạng:

```text
(batch_size, 1, 28, 28)
```

Kết quả đầu ra của mô hình phải là logits có dạng:

```text
(batch_size, 10)
```

Mỗi ảnh tương ứng với 10 logits, đại diện cho điểm số dự đoán của 10 lớp.

Khi dùng `nn.CrossEntropyLoss()`, mô hình phải trả về logits trực tiếp. Không thêm lớp `Softmax` ở cuối model vì `CrossEntropyLoss` đã xử lý bước này bên trong.

---

## 3. Phân biệt parameters và hyperparameters

### 3.1. Model parameters

Model parameters là các giá trị do mô hình tự học trong lúc train.

Ví dụ:

- Weight của lớp `Linear`.
- Bias của lớp `Linear`.
- Weight và bias của lớp `Conv2d`.
- Weight của Batch Normalization nếu mô hình có dùng.

Các parameters được cập nhật qua các bước:

```text
Forward
→ Tính loss
→ Backward
→ Optimizer cập nhật weight và bias
```

### 3.2. Hyperparameters

Hyperparameters là các giá trị được người lập trình lựa chọn trước khi train, không phải do model tự học.

Ví dụ:

- Learning rate.
- Batch size.
- Number of epochs.
- Optimizer.
- Dropout rate.
- Weight decay.
- Số lớp hidden layer.
- Số neuron mỗi hidden layer.
- Số filters trong convolution layer.

Các hyperparameters có ảnh hưởng lớn đến kết quả cuối cùng. Một mô hình tốt nhưng hyperparameters không phù hợp vẫn có thể cho kết quả kém.

---

## 4. Cấu hình mặc định của project

Theo file `src/config.py`, project có cấu hình khởi đầu như sau:

| Hyperparameter | Giá trị | Ý nghĩa |
|---|---:|---|
| `BATCH_SIZE` | 64 | Số ảnh trong một batch |
| `LEARNING_RATE` | `1e-3` | Tốc độ cập nhật trọng số |
| `EPOCHS` | 10 | Số lần duyệt toàn bộ train set |
| `NUM_CLASSES` | 10 | Số lớp FashionMNIST |
| `RANDOM_SEED` | 42 | Hỗ trợ tái lập kết quả |
| `NUM_WORKERS` | 0 | Số tiến trình DataLoader |

Cấu hình baseline được đề xuất:

```python
batch_size = 64
learning_rate = 1e-3
epochs = 10
optimizer_name = "Adam"
dropout_rate = 0.3
weight_decay = 1e-4
```

Đây là cấu hình ban đầu để train và so sánh. Sau đó cần thử nghiệm nhiều cấu hình khác nhau để chọn ra kết quả tốt nhất.

---

## 5. Learning rate

### 5.1. Khái niệm

Learning rate, thường viết là `lr`, là tốc độ cập nhật trọng số của mô hình sau mỗi bước train.

Trong Gradient Descent, có thể hiểu đơn giản:

```text
weight mới = weight cũ - learning_rate × gradient
```

Trong PyTorch, learning rate được truyền vào optimizer:

```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3
)
```

Learning rate là một trong các hyperparameters quan trọng nhất vì nó quyết định mô hình học nhanh hay chậm, ổn định hay không ổn định.

### 5.2. Learning rate quá nhỏ

Ví dụ:

```python
learning_rate = 1e-5
```

Khi learning rate quá nhỏ:

- Weight thay đổi rất ít sau mỗi batch.
- Loss giảm chậm.
- Accuracy tăng chậm.
- Cần nhiều epoch hơn để mô hình học.
- Thời gian train lâu.
- Sau số epoch giới hạn, model có thể chưa học đủ.

Dấu hiệu nhận biết:

```text
Train loss vẫn cao.
Validation loss vẫn cao.
Loss giảm rất ít sau mỗi epoch.
Train accuracy và validation accuracy tăng chậm.
```

### 5.3. Learning rate quá lớn

Ví dụ:

```python
learning_rate = 1e-1
```

Khi learning rate quá lớn:

- Weight thay đổi quá mạnh sau mỗi lần update.
- Loss có thể dao động.
- Loss có thể tăng thay vì giảm.
- Accuracy không ổn định.
- Model có thể không hội tụ.

Dấu hiệu nhận biết:

```text
Loss lúc giảm, lúc tăng mạnh.
Validation accuracy dao động nhiều.
Train loss rất lớn hoặc xuất hiện giá trị NaN.
Model train nhiều epoch nhưng accuracy không cải thiện.
```

### 5.4. Giá trị learning rate nên thử

Với Adam, có thể thử:

| Learning rate | Nhận xét |
|---:|---|
| `1e-4` | Học chậm nhưng thường ổn định |
| `3e-4` | Thường ổn định hơn khi `1e-3` dao động |
| `1e-3` | Baseline phù hợp cho Adam |
| `3e-3` | Có thể học nhanh hơn nhưng cần theo dõi loss |
| `1e-2` | Có nguy cơ không hội tụ |

Danh sách thử nghiệm:

```python
learning_rates = [3e-4, 1e-3, 3e-3]
```

Khi thử learning rate, phải giữ nguyên model, batch size, optimizer, dropout và số epoch để so sánh công bằng.

---

## 6. Batch size

### 6.1. Khái niệm

Batch size là số ảnh được đưa vào model trước khi thực hiện một lần cập nhật weight.

Ví dụ:

```python
batch_size = 64
```

FashionMNIST có 60.000 ảnh train. Nếu batch size bằng 64 thì một epoch có xấp xỉ:

```text
60000 / 64 ≈ 938 batches
```

Ở mỗi batch, mô hình thực hiện:

```text
1. Nhận một batch ảnh và nhãn.
2. Chạy forward để tạo logits.
3. Tính loss giữa logits và nhãn thật.
4. Chạy backward để tính gradient.
5. Gọi optimizer.step() để cập nhật weight.
```

### 6.2. Batch size nhỏ

Ví dụ:

```python
batch_size = 32
```

Ưu điểm:

- Dùng ít RAM hoặc VRAM hơn.
- Phù hợp với máy có cấu hình thấp.
- Cập nhật weight thường xuyên hơn.
- Có thể giúp mô hình tổng quát hóa tốt trong một số trường hợp.

Hạn chế:

- Số batch trong mỗi epoch nhiều hơn.
- Train có thể chậm hơn.
- Loss có thể dao động nhiều hơn.
- Thời gian chạy tổng thể có thể dài hơn.

### 6.3. Batch size lớn

Ví dụ:

```python
batch_size = 128
```

Ưu điểm:

- Ít batch hơn trong mỗi epoch.
- Có thể train nhanh hơn nếu dùng GPU.
- Gradient thường ổn định hơn.
- Tận dụng GPU tốt hơn khi GPU có đủ VRAM.

Hạn chế:

- Tốn nhiều RAM hoặc VRAM hơn.
- Có thể gây lỗi hết bộ nhớ.
- Không bảo đảm validation accuracy tốt hơn batch size nhỏ.
- Ít lần cập nhật weight hơn trong một epoch.

### 6.4. Batch size cần thử

| Batch size | Mục đích |
|---:|---|
| 32 | Thử batch nhỏ |
| 64 | Baseline của project |
| 128 | Thử batch lớn |

```python
batch_sizes = [32, 64, 128]
```

Không nên thay đổi batch size và learning rate cùng lúc ở giai đoạn đầu vì rất khó xác định nguyên nhân làm kết quả thay đổi.

---

## 7. Number of epochs

### 7.1. Khái niệm

Một epoch là một lần mô hình đi qua toàn bộ tập train.

Ví dụ:

```python
epochs = 10
```

Điều này nghĩa là 60.000 ảnh train được đưa qua model 10 lần.

### 7.2. Epoch quá ít: underfitting

Underfitting xảy ra khi mô hình chưa học đủ từ dữ liệu train.

Dấu hiệu:

- Train loss cao.
- Validation loss cao.
- Train accuracy thấp.
- Validation accuracy thấp.
- Loss vẫn đang giảm rõ rệt tại epoch cuối cùng.

Nguyên nhân có thể là:

- Số epoch quá ít.
- Learning rate quá nhỏ.
- Mô hình quá đơn giản.
- Số neuron hoặc số filters quá ít.

Cách xử lý:

- Tăng số epoch.
- Điều chỉnh learning rate.
- Dùng kiến trúc mạnh hơn.
- Tăng số neuron hoặc filters nếu cần.

### 7.3. Epoch quá nhiều: overfitting

Overfitting xảy ra khi model học quá kỹ train set nhưng không tổng quát tốt trên dữ liệu chưa thấy.

Dấu hiệu:

- Train loss tiếp tục giảm.
- Train accuracy tiếp tục tăng.
- Validation loss bắt đầu tăng.
- Validation accuracy đứng yên hoặc giảm.
- Train accuracy cao hơn validation accuracy quá nhiều.

Cách xử lý:

- Tăng dropout.
- Dùng weight decay.
- Giảm độ phức tạp của model.
- Dùng early stopping.
- Chọn model có validation loss thấp nhất thay vì model của epoch cuối.

### 7.4. Số epoch nên thử

```python
epochs_to_try = [10, 15, 20]
```

Trong quá trình train, cần lưu history của train loss, validation loss, train accuracy và validation accuracy để vẽ biểu đồ và chọn epoch tốt nhất.

---

## 8. Optimizer

### 8.1. Khái niệm

Optimizer là thuật toán dùng gradient để cập nhật parameters của model, nhằm làm loss giảm dần.

Thứ tự cơ bản trong train loop:

```python
optimizer.zero_grad()

outputs = model(images)
loss = criterion(outputs, labels)

loss.backward()
optimizer.step()
```

Ý nghĩa từng bước:

```text
optimizer.zero_grad()
→ Xóa gradient cũ trước khi train batch mới.

outputs = model(images)
→ Chạy forward, model dự đoán logits.

loss = criterion(outputs, labels)
→ So sánh logits với nhãn thật.

loss.backward()
→ Tính gradient của loss theo từng parameter.

optimizer.step()
→ Cập nhật parameter theo gradient đã tính.
```

### 8.2. SGD

SGD là Stochastic Gradient Descent.

```python
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=1e-2
)
```

Ưu điểm:

- Đơn giản.
- Dễ hiểu.
- Thích hợp để học nguyên lý tối ưu cơ bản.

Hạn chế:

- Nhạy với learning rate.
- Có thể hội tụ chậm.
- Dễ dao động nếu learning rate không phù hợp.

### 8.3. SGD với momentum

Momentum giúp optimizer giữ lại một phần hướng cập nhật trước đó, hỗ trợ train ổn định hơn và nhanh hơn SGD cơ bản.

```python
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=1e-2,
    momentum=0.9
)
```

Giá trị `momentum=0.9` là một cấu hình phổ biến để thử.

### 8.4. Adam

Adam thường được dùng làm baseline vì có khả năng tự điều chỉnh mức cập nhật dựa trên gradient.

```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3
)
```

Ưu điểm:

- Thường hội tụ nhanh.
- Phù hợp để bắt đầu với FashionMNIST.
- Thường ít cần điều chỉnh learning rate hơn SGD.
- Có thể dùng cho cả MLP và CNN.

Hạn chế:

- Vẫn phải theo dõi validation loss.
- Không phải lúc nào cũng tốt hơn SGD.
- Learning rate không phù hợp vẫn có thể làm model không hội tụ.

### 8.5. Optimizer cần thử

| Optimizer | Learning rate đề xuất | Mục đích |
|---|---:|---|
| SGD | `1e-2` | Baseline đơn giản |
| SGD với momentum 0.9 | `1e-2` | So sánh với Adam |
| Adam | `1e-3` | Baseline chính |

---

## 9. Dropout

### 9.1. Khái niệm

Dropout là kỹ thuật regularization dùng để giảm overfitting.

Trong lúc train, dropout sẽ tạm thời đặt ngẫu nhiên một số neuron bằng 0. Điều này khiến mạng không phụ thuộc quá mức vào một số neuron cụ thể.

Ví dụ:

```python
nn.Dropout(p=0.3)
```

Trong đó:

```text
p = xác suất một neuron bị tắt khi train
```

Nếu `p=0.3`, khoảng 30% neuron đầu vào của lớp Dropout bị bỏ qua trong mỗi lần train.

### 9.2. Ví dụ áp dụng dropout trong MLP

```python
self.classifier = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Dropout(p=0.3),

    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Dropout(p=0.3),

    nn.Linear(128, 10),
)
```

### 9.3. Giá trị dropout nên thử

| Dropout rate | Ý nghĩa |
|---:|---|
| `0.0` | Không sử dụng dropout |
| `0.2` | Regularization nhẹ |
| `0.3` | Giá trị baseline đề xuất |
| `0.5` | Regularization mạnh |

```python
dropout_rates = [0.0, 0.2, 0.3, 0.5]
```

### 9.4. Dropout trong train và evaluation

Dropout có hành vi khác nhau tùy chế độ model.

Khi train:

```python
model.train()
```

- Dropout được bật.
- Một số neuron được tạm thời tắt.
- Giúp hạn chế overfitting.

Khi validation hoặc test:

```python
model.eval()
```

- Dropout được tắt.
- Tất cả neuron được dùng để dự đoán.
- Kết quả dự đoán ổn định hơn.

Vì vậy cần luôn gọi `model.train()` trong train loop và `model.eval()` trong validation/test loop.

---

## 10. Weight decay

Weight decay là một kỹ thuật regularization khác. Nó thêm hình phạt khi weight có giá trị quá lớn.

Ví dụ:

```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-4
)
```

Mục đích:

- Giảm overfitting.
- Hạn chế weight tăng quá lớn.
- Cải thiện khả năng tổng quát hóa trong một số trường hợp.

Các giá trị nên thử:

```python
weight_decays = [0.0, 1e-5, 1e-4]
```

Nếu train accuracy cao nhưng validation accuracy thấp, có thể thử tăng dropout hoặc thêm weight decay.

---

## 11. Kiến trúc MLP

### 11.1. Khái niệm

MLP là Multi-Layer Perceptron. MLP thường flatten ảnh thành một vector rồi đưa qua các lớp fully connected.

Ảnh đầu vào:

```text
(batch_size, 1, 28, 28)
```

Sau khi flatten:

```text
(batch_size, 784)
```

Vì:

```text
28 × 28 = 784
```

### 11.2. Kiến trúc MLP đề xuất

```text
Input: (batch_size, 1, 28, 28)
→ Flatten
→ Linear(784, 256)
→ ReLU
→ Dropout(0.3)
→ Linear(256, 128)
→ ReLU
→ Dropout(0.3)
→ Linear(128, 10)
```

Ví dụ ý tưởng code:

```python
class MLP(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super().__init__()

        self.network = nn.Sequential(
            nn.Flatten(),

            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.network(x)
```

### 11.3. Ưu điểm của MLP

- Kiến trúc đơn giản.
- Dễ hiểu.
- Dễ train.
- Phù hợp làm baseline.
- Thời gian train thường nhanh hơn CNN.

### 11.4. Hạn chế của MLP

- Làm mất thông tin không gian của ảnh khi flatten.
- Không tối ưu cho bài toán ảnh.
- Khó học đặc trưng ảnh như cạnh, họa tiết và hình dạng.
- Accuracy thường thấp hơn CNN trên FashionMNIST.

---

## 12. Kiến trúc CNN

### 12.1. Khái niệm

CNN là Convolutional Neural Network. CNN xử lý trực tiếp cấu trúc 2 chiều của ảnh và sử dụng convolution để học các đặc trưng cục bộ.

CNN có thể học:

- Cạnh.
- Đường thẳng.
- Đường cong.
- Họa tiết.
- Hình dạng của quần áo.
- Đặc trưng của giày, túi, áo và váy.

### 12.2. Kiến trúc CNN đề xuất

```text
Input: (batch_size, 1, 28, 28)

→ Conv2d(1, 32, kernel_size=3, padding=1)
→ ReLU
→ MaxPool2d(2)

→ Conv2d(32, 64, kernel_size=3, padding=1)
→ ReLU
→ MaxPool2d(2)

→ Flatten
→ Linear(64 × 7 × 7, 128)
→ ReLU
→ Dropout(0.3)
→ Linear(128, 10)
```

Thay đổi kích thước dữ liệu trong CNN:

```text
Input:
(1, 28, 28)

Sau Conv2d(1, 32, kernel_size=3, padding=1):
(32, 28, 28)

Sau MaxPool2d(2):
(32, 14, 14)

Sau Conv2d(32, 64, kernel_size=3, padding=1):
(64, 14, 14)

Sau MaxPool2d(2):
(64, 7, 7)

Sau Flatten:
64 × 7 × 7 = 3136
```

### 12.3. Ví dụ ý tưởng code CNN

```python
class CNN(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),

            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
```

### 12.4. Ưu điểm của CNN

- Giữ được cấu trúc không gian của ảnh.
- Phù hợp với bài toán phân loại ảnh.
- Học được đặc trưng ảnh tốt hơn MLP.
- Thường đạt accuracy tốt hơn MLP với FashionMNIST.

### 12.5. Hạn chế của CNN

- Phức tạp hơn MLP.
- Cần theo dõi shape của tensor.
- Train có thể lâu hơn.
- Tốn tài nguyên hơn MLP.

---

## 13. So sánh MLP và CNN

| Tiêu chí | MLP | CNN |
|---|---|---|
| Cách xử lý ảnh | Flatten ảnh thành vector | Giữ cấu trúc 2 chiều của ảnh |
| Độ phức tạp | Thấp | Cao hơn |
| Thời gian train | Nhanh hơn | Chậm hơn |
| Khả năng học đặc trưng ảnh | Hạn chế | Tốt hơn |
| Số lớp chính | Linear | Conv2d, MaxPool2d, Linear |
| Phù hợp cho ảnh | Cơ bản | Phù hợp hơn |
| Accuracy kỳ vọng | Thường thấp hơn CNN | Thường tốt hơn MLP |
| Vai trò trong Lab 1 | Mô hình baseline | Mô hình chính để thử nghiệm |

Không được ghi số liệu accuracy giả vào báo cáo. Tất cả loss, accuracy và nhận xét phải lấy từ kết quả chạy thật.

---

## 14. Cách chia dữ liệu để chọn hyperparameters

Không dùng test set để chọn learning rate, batch size, optimizer hoặc dropout.

Nên chia dữ liệu như sau:

| Dataset | Số lượng gợi ý | Mục đích |
|---|---:|---|
| Train set | 54.000 ảnh | Cập nhật weight của model |
| Validation set | 6.000 ảnh | Chọn hyperparameters |
| Test set | 10.000 ảnh | Đánh giá cuối cùng |

Quy trình đúng:

```text
1. Train model trên train set.
2. Đánh giá từng experiment trên validation set.
3. Chọn kiến trúc và hyperparameters có kết quả validation tốt nhất.
4. Train hoặc tải lại checkpoint tốt nhất.
5. Đánh giá cuối cùng đúng một lần trên test set.
```

Không nên sử dụng test set nhiều lần trong khi tuning vì điều này khiến kết quả test bị thiên lệch.

---

## 15. Nguyên tắc thử nghiệm công bằng

Khi so sánh các hyperparameters, phải thay đổi có kiểm soát.

Cần giữ cố định:

- Random seed.
- Train/validation split.
- Transform và normalization.
- Số epoch.
- Loss function.
- Cách tính accuracy.
- Model architecture, nếu đang thử learning rate hoặc batch size.
- Thiết bị train nếu có thể.

Ví dụ đúng: chỉ thay đổi learning rate.

```text
Experiment A:
CNN, batch size 64, Adam, dropout 0.3, lr=3e-4

Experiment B:
CNN, batch size 64, Adam, dropout 0.3, lr=1e-3

Experiment C:
CNN, batch size 64, Adam, dropout 0.3, lr=3e-3
```

Ví dụ không đúng: vừa thay learning rate, vừa thay model, batch size và dropout. Khi đó không thể biết yếu tố nào đã làm kết quả thay đổi.

---

## 16. Kế hoạch thử nghiệm hyperparameters

### 16.1. Giai đoạn 1: So sánh MLP và CNN

| Experiment | Model | Batch size | Learning rate | Optimizer | Dropout | Epochs |
|---|---|---:|---:|---|---:|---:|
| E1 | MLP | 64 | `1e-3` | Adam | 0.3 | 10 |
| E2 | CNN | 64 | `1e-3` | Adam | 0.3 | 10 |

Mục tiêu:

- So sánh validation loss.
- So sánh validation accuracy.
- Chọn kiến trúc tốt hơn để tiếp tục tuning.

### 16.2. Giai đoạn 2: Thử learning rate

Sau khi chọn kiến trúc tốt hơn ở giai đoạn 1:

| Experiment | Learning rate |
|---|---:|
| E3 | `3e-4` |
| E4 | `1e-3` |
| E5 | `3e-3` |

### 16.3. Giai đoạn 3: Thử batch size

| Experiment | Batch size |
|---|---:|
| E6 | 32 |
| E7 | 64 |
| E8 | 128 |

### 16.4. Giai đoạn 4: Thử dropout

| Experiment | Dropout |
|---|---:|
| E9 | 0.0 |
| E10 | 0.2 |
| E11 | 0.3 |
| E12 | 0.5 |

### 16.5. Giai đoạn 5: Thử optimizer

| Experiment | Optimizer | Learning rate |
|---|---|---:|
| E13 | Adam | `1e-3` |
| E14 | SGD | `1e-2` |
| E15 | SGD với momentum 0.9 | `1e-2` |

### 16.6. Giai đoạn 6: Thử weight decay

| Experiment | Weight decay |
|---|---:|
| E16 | `0.0` |
| E17 | `1e-5` |
| E18 | `1e-4` |

Không nhất thiết phải thực hiện toàn bộ thí nghiệm nếu không đủ thời gian. Nhóm nên ưu tiên theo thứ tự:

```text
MLP và CNN
→ Learning rate
→ Batch size
→ Dropout
→ Optimizer
→ Weight decay
```

---

## 17. Các chỉ số cần lưu sau mỗi experiment

Sau mỗi lần chạy, cần lưu:

- Tên experiment.
- Kiến trúc model.
- Batch size.
- Learning rate.
- Optimizer.
- Dropout rate.
- Weight decay.
- Số epoch.
- Epoch tốt nhất.
- Train loss.
- Validation loss.
- Train accuracy.
- Validation accuracy.
- Thời gian train.
- Ghi chú về hiện tượng underfitting hoặc overfitting.

Kết quả tổng hợp sẽ được lưu trong:

```text
results/tables/
```

Biểu đồ loss và accuracy sẽ được lưu trong:

```text
results/figures/
```

---

## 18. Mẫu bảng kết quả thực nghiệm

| Experiment | Model | Batch size | LR | Optimizer | Dropout | Weight decay | Best epoch | Train loss | Val loss | Train accuracy | Val accuracy | Nhận xét |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| E1 | MLP | 64 | `1e-3` | Adam | 0.3 | `1e-4` |  |  |  |  |  |  |
| E2 | CNN | 64 | `1e-3` | Adam | 0.3 | `1e-4` |  |  |  |  |  |  |
| E3 | CNN | 64 | `3e-4` | Adam | 0.3 | `1e-4` |  |  |  |  |  |  |
| E4 | CNN | 64 | `1e-3` | Adam | 0.3 | `1e-4` |  |  |  |  |  |  |
| E5 | CNN | 64 | `3e-3` | Adam | 0.3 | `1e-4` |  |  |  |  |  |  |

Các ô trống chỉ được điền sau khi chạy mô hình thật. Không được điền số liệu minh họa hoặc số liệu giả.

---

## 19. Dấu hiệu và cách điều chỉnh hyperparameters

| Hiện tượng | Nguyên nhân có thể | Hướng xử lý |
|---|---|---|
| Loss giảm rất chậm | Learning rate quá nhỏ | Tăng learning rate nhẹ |
| Loss dao động mạnh | Learning rate quá lớn | Giảm learning rate |
| Loss có giá trị NaN | Learning rate quá lớn hoặc lỗi dữ liệu | Giảm learning rate, kiểm tra dữ liệu |
| Train accuracy thấp và validation accuracy thấp | Underfitting | Train thêm epoch, tăng model capacity hoặc điều chỉnh learning rate |
| Train accuracy cao nhưng validation accuracy thấp | Overfitting | Tăng dropout, thêm weight decay hoặc giảm độ phức tạp model |
| Hết RAM hoặc VRAM | Batch size quá lớn | Giảm batch size |
| Train quá chậm | Batch size nhỏ hoặc model quá lớn | Tăng batch size nếu phần cứng cho phép |
| Validation loss tăng sau nhiều epoch | Overfitting | Dùng early stopping hoặc chọn checkpoint tốt nhất |
| CNN không chạy do lỗi shape | Sai kích thước khi Flatten hoặc Linear | Kiểm tra shape sau Conv2d và MaxPool2d |

---

## 20. Kết luận

Hyperparameters quyết định rất lớn đến chất lượng mô hình FashionMNIST.

Cấu hình baseline đề xuất cho Lab 1 là:

```python
batch_size = 64
learning_rate = 1e-3
epochs = 10
optimizer = "Adam"
dropout_rate = 0.3
weight_decay = 1e-4
random_seed = 42
```

MLP nên được dùng làm mô hình baseline đơn giản. CNN thường phù hợp hơn cho FashionMNIST vì CNN giữ được cấu trúc không gian của ảnh và học đặc trưng ảnh tốt hơn.

Quá trình chọn hyperparameters phải dựa trên validation set. Test set chỉ dùng để đánh giá cuối cùng sau khi đã chọn được model và cấu hình tốt nhất.

Kết quả của các lần thử nghiệm phải được lưu vào `results/tables/`, biểu đồ loss và accuracy được lưu vào `results/figures/`, sau đó dùng các kết quả thực tế này để viết `docs/06_results/experiment_results.md`.

---

