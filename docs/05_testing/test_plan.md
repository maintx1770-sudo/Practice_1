# Kế hoạch kiểm thử – Practice 1

## 1. Thông tin chung

| Mục | Nội dung |
|---|---|
| Tên dự án | **Practice 1 – PyTorch FashionMNIST Classification** |
| Công cụ | `pytest`, PyTorch, Python |
| Trạng thái | Đã thực hiện ngày 09/11/2026; kết quả ghi trong `docs/05_testing/test_report.md` |

## 2. Mục tiêu

Kiểm tra các thành phần của dự án hoạt động đúng trước khi tích hợp và nộp bài: tải/tiền xử lý FashionMNIST, xây dựng mô hình, huấn luyện, đánh giá, lưu/tải checkpoint. Phát hiện lỗi về kiểu dữ liệu, kích thước tensor, nhãn, phép tính loss/accuracy và quy trình huấn luyện.

Việc **chạy test thực tế, ghi số lượng pass/fail và xử lý lỗi** sẽ được tổng hợp trong `docs/05_testing/test_report.md` theo kế hoạch ngày 09/11. Không sử dụng dữ liệu mô phỏng làm kết quả huấn luyện thật.

## 3. Phạm vi kiểm thử

| Thành phần | File kiểm thử | Nội dung cần xác nhận | Thành viên viết test |
|---|---|---|---|
| Biến đổi ảnh | `tests/test_transforms.py` | Tensor, hình dạng, chuẩn hóa | Nguyễn Chi |
| Dữ liệu | `tests/test_dataset.py` | FashionMNIST, DataLoader, ảnh/nhãn/batch | Nguyễn Chi |
| Mạng nơ-ron | `tests/test_model.py` | Forward, logits 10 lớp, kích thước đầu ra | My |
| Lưu/tải mô hình | `tests/test_model_io.py` | Checkpoint tồn tại, nạp lại trọng số chính xác | My |
| Huấn luyện | `tests/test_trainer.py` | Forward → loss → backward → optimizer; lịch sử huấn luyện | Huynh |
| Đánh giá | `tests/test_metrics.py` | Chế độ đánh giá, loss và accuracy hợp lệ | Huynh |
| Dữ liệu dùng chung | `tests/conftest.py` | Cung cấp fixture phục vụ các bài kiểm thử | Trúc Chi |

**Ngoài phạm vi unit test:** so sánh chất lượng thực nghiệm của các kiến trúc, hyperparameter tuning và diễn giải biểu đồ. Những nội dung này được xác minh bằng notebook, script `experiments/` và báo cáo kết quả của nhóm. Không đặt yêu cầu một mức accuracy cố định khi chưa có tiêu chí được nhóm thống nhất.

## 4. Phương pháp kiểm thử

- **Unit testing:** kiểm tra riêng từng hàm/module; ưu tiên dữ liệu giả để chạy nhanh, không phụ thuộc mạng.
- **Integration testing:** kiểm tra các module ghép với nhau: transforms → DataLoader → model → trainer → metrics → save/load.
- **Kiểm tra thủ công:** mở notebook và xem biểu đồ loss, ảnh `predicted vs actual`, bảng kết quả do nhóm tạo.
- **Kiểm tra tính tái lập:** với test dùng dữ liệu ngẫu nhiên, cố định seed khi cần; tránh so sánh một giá trị loss/accuracy cụ thể trong những test không bảo đảm tính xác định.

## 5. Môi trường kiểm thử

| Thành phần | Thiết lập / cách kiểm tra |
|---|---|
| Hệ điều hành | Windows (máy thực tế của nhóm; ghi phiên bản khi chạy) |
| Python | Từ 3.9 trở lên theo yêu cầu dự án; ghi phiên bản thực tế bằng `python --version` |
| Môi trường | Virtual environment `.venv` |
| Framework | PyTorch, torchvision |
| Công cụ | `pytest` |
| Thiết bị | Ưu tiên CPU cho unit test; GPU không bắt buộc |
| Dữ liệu unit test | Fixture tại `tests/conftest.py` |
| Dữ liệu thực | FashionMNIST tại `data/input/` cho integration test nếu cần |

**Ghi nhận khi chạy:** phiên bản Python / PyTorch / torchvision / pytest, hệ điều hành, ngày kiểm thử, commit Git đang kiểm tra.

## 6. Fixture dùng chung

File `tests/conftest.py` đã chuẩn bị các fixture sau (đối chiếu theo phiên bản file đã tạo cho nhóm):

| Fixture | Dữ liệu cung cấp |
|---|---|
| `device` | CPU (`torch.device('cpu')`) |
| `sample_images` | 8 ảnh tensor dạng `(8, 1, 28, 28)`; seed 42 |
| `sample_labels` | 8 nhãn hợp lệ, kiểu `torch.long` |
| `sample_batch` | `(sample_images, sample_labels)` |
| `sample_dataset` | `TensorDataset` từ dữ liệu giả |
| `sample_loader` | `DataLoader`, batch size 4, không shuffle |
| `sample_pil_image` | Ảnh PIL grayscale 28 × 28 |
| `sample_model` | MLP nhỏ chỉ dành cho unit test, **không phải model chính thức** |
| `sample_criterion` | `nn.CrossEntropyLoss()` |
| `sample_optimizer` | `SGD` cho `sample_model` |
| `checkpoint_path` | Đường dẫn tạm do `tmp_path` cung cấp |

**Lưu ý:** `sample_images` là ảnh tensor giả trong khoảng `[0,1]`, không phải tensor đã qua `Normalize((0.5,), (0.5,))`. Các phép kiểm tra chuẩn hóa phải gọi transform tương ứng. Test model chính thức cần import từ `src/models/neural_network.py` khi My hoàn thiện.

## 7. Quy trình chạy kiểm thử

Tại thư mục gốc `Practice_1` trong Terminal VS Code, sau khi kích hoạt `.venv`:

```powershell
python -m pytest tests -q
```

Xem kết quả chi tiết hơn khi cần:

```powershell
python -m pytest tests -v
```

Nếu cần lưu log trên PowerShell:

```powershell
python -m pytest tests -v *> pytest_output.txt
```

Các bước thực hiện:

1. Cập nhật code mới nhất từ nhánh dùng chung; xác nhận các module cần kiểm tra đã được hoàn thiện.
2. Kiểm tra các package có thể import và các fixture trong `conftest.py` được nhận diện.
3. Chạy từng file test liên quan để xác định lỗi; sau đó chạy toàn bộ `tests/`.
4. Ghi lại test case, kết quả thực tế, lỗi/traceback, người xử lý và lần kiểm tra lại.
5. Sau khi các bài kiểm thử đạt yêu cầu, xác minh notebook chạy được và có các hình/bảng cần nộp.
6. Tổng hợp kết quả vào `docs/05_testing/test_report.md`.

## 8. Tiêu chí đạt / không đạt

- **PASS:** kết quả thực tế đúng mong đợi của test case, không phát sinh exception ngoài dự kiến.
- **FAIL:** assertion không đạt, lỗi runtime/import, hoặc kết quả không đúng đặc tả.
- **BLOCKED:** chưa thể chạy do module chưa hoàn thiện, thiếu phụ thuộc hoặc dữ liệu thực không sẵn có; ghi rõ nguyên nhân.
- **NOT RUN:** chưa thực hiện.

**Tiêu chí nghiệm thu kiểm thử:** các test bắt buộc đã được chạy và PASS; lỗi nghiêm trọng (không tải dữ liệu, sai kích thước đầu ra, không train/evaluate, không lưu/tải được model) đã được xử lý. Nếu còn test FAIL/BLOCKED, mô tả rõ trong báo cáo thay vì ghi toàn bộ đã đạt.

## 9. Rủi ro và cách xử lý

| Rủi ro | Ảnh hưởng | Cách xử lý |
|---|---|---|
| Thành viên chưa hoàn thiện module | Test bị BLOCKED | Đối chiếu timeline, thử fixture dữ liệu giả, kiểm tra lại sau |
| Khác biệt CPU/GPU | Kết quả có sai khác nhỏ | Dùng CPU cho unit test; so sánh tensor với tolerance phù hợp |
| Lần đầu tải FashionMNIST cần Internet | Integration test không chạy | Ưu tiên unit test offline; thử lại khi dữ liệu đã tải |
| Test dùng đường dẫn thật | Dễ ghi đè checkpoint | Sử dụng `tmp_path` và `checkpoint_path` |
| Các hàm thực tế đổi tên/chữ ký | Test bị lỗi import/TypeError | Cập nhật test case/test code theo API đã thống nhất với nhóm |
