# Classroom Safety AI - Hệ thống phát hiện bạo lực và té ngã

## Giới thiệu

Classroom Safety AI là hệ thống hỗ trợ giám sát video lớp học, sử dụng **YOLO Pose kết hợp Tracking và mô hình nhận dạng hành động** để phát hiện các tình huống bất thường như bạo lực hoặc té ngã.

Hệ thống phân tích chuyển động cơ thể qua nhiều khung hình, đưa ra cảnh báo để giáo viên hoặc người quản lý kiểm tra. Dự án không nhận diện khuôn mặt và không tự động kết luận vi phạm.

## Tính năng chính

- **Phân tích video:**
  - Nhận video có sẵn hoặc luồng camera.
  - Phát hiện nhiều người trong cùng khung hình.
  - Hiển thị bounding box, skeleton và độ tin cậy.

- **Nhận dạng hành động:**
  - Phân loại ba trạng thái: `normal`, `violence`, `fall`.
  - Theo dõi chuyển động của từng người qua nhiều frame.
  - Sử dụng chuỗi keypoint để phân biệt hành động.

- **Phát hiện té ngã:**
  - Phân tích góc nghiêng cơ thể, vị trí hông và tốc độ thay đổi tư thế.
  - Hạn chế nhầm lẫn với cúi người, ngồi xuống hoặc nằm có chủ đích.

- **Phát hiện bạo lực:**
  - Phân tích chuyển động nhanh của tay, chân và khoảng cách giữa nhiều người.
  - Hỗ trợ nhận biết các hành động như đấm, đá, xô đẩy hoặc vật lộn.

- **Cảnh báo và lưu kết quả:**
  - Chỉ cảnh báo khi sự kiện xuất hiện liên tục trong nhiều frame.
  - Lưu video đã xử lý.
  - Xuất nhật ký sự kiện dưới dạng CSV.
  - Ghi lại thời gian, loại sự kiện, Track ID và confidence.

- **Giao diện demo:**
  - Tải video lên hệ thống.
  - Điều chỉnh ngưỡng confidence.
  - Xem kết quả nhận dạng.
  - Tải video và nhật ký cảnh báo.

## Cấu trúc dự án

- `app/` - Chứa giao diện Streamlit.
- `configs/` - Chứa cấu hình dataset, mô hình và ngưỡng cảnh báo.
- `data/raw/` - Chứa video gốc theo các lớp `normal`, `violence`, `fall`.
- `data/processed/` - Chứa video RGB chuẩn hóa; dữ liệu keypoint được xuất riêng.
- `data/annotations/` - Chứa nhãn hành động và thời điểm xảy ra sự kiện.
- `data/metadata/` - Chứa thông tin nguồn, người tham gia, bối cảnh và giấy phép.
- `src/data/` - Xử lý video, cắt clip, trích xuất frame và chia dữ liệu.
- `src/pose/` - Phát hiện tư thế, tracking và trích xuất đặc trưng.
- `src/models/` - Chứa baseline, mô hình LSTM/GRU và chương trình huấn luyện.
- `src/evaluation/` - Tính metric, confusion matrix và đánh giá tốc độ.
- `src/inference/` - Chạy dự đoán video và quản lý cảnh báo.
- `models/` - Lưu trọng số và checkpoint.
- `outputs/` - Lưu video kết quả, biểu đồ và nhật ký.
- `tests/` - Chứa mã kiểm thử.
- `docs/` - Chứa tài liệu dataset, gán nhãn, thiết kế và Responsible AI.

## Công nghệ sử dụng

- **Ngôn ngữ:** Python 3.10 hoặc 3.11.
- **Computer Vision:** OpenCV, Ultralytics YOLO Pose.
- **Deep Learning:** PyTorch, LSTM hoặc GRU.
- **Tracking:** ByteTrack hoặc tracking tích hợp trong Ultralytics.
- **Xử lý dữ liệu:** NumPy, Pandas.
- **Đánh giá:** scikit-learn, Matplotlib, Seaborn.
- **Giao diện:** Streamlit.
- **Quản lý mã nguồn:** Git và GitHub.
- **Huấn luyện:** Máy tính có GPU NVIDIA hoặc Google Colab.

## Triển khai

Dự án được triển khai theo ba môi trường:

1. **Máy cá nhân:** phát triển, kiểm thử và xử lý video bằng VS Code.
2. **Google Colab hoặc máy có GPU:** huấn luyện mô hình LSTM/GRU.
3. **Streamlit Local:** chạy giao diện trình diễn trên máy tính của nhóm.

Chạy giao diện:

```powershell
streamlit run app/app.py
```

Sau khi mô hình hoạt động ổn định, hệ thống có thể triển khai trên máy chủ nội bộ hoặc thiết bị GPU đặt gần camera. Không nên đưa video lớp học thật lên dịch vụ công cộng khi chưa có cơ chế bảo vệ dữ liệu.

## Hướng dẫn thiết lập

### 1. Tải source code

```powershell
git clone https://github.com/USERNAME/classroom-violence-fall-detection.git
cd classroom-violence-fall-detection
code .
```

Thay `USERNAME` bằng tên tài khoản GitHub của nhóm.

### 2. Tạo môi trường ảo

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Nếu PowerShell chặn kích hoạt:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### 3. Cài đặt thư viện

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Kiểm tra môi trường

```powershell
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
python -c "from ultralytics import YOLO; print('Ultralytics OK')"
python -c "import cv2; print('OpenCV:', cv2.__version__)"
```

### 5. Chuẩn bị dữ liệu

```text
data/raw/
├── normal/
├── violence/
└── fall/
```

Có thể đặt video trực tiếp trong `data/raw/` hoặc trong các thư mục con. Không đưa video thật hoặc dữ liệu cá nhân lên repository công khai.

### 6. Chạy thử YOLO Pose

YOLO có thể chạy với một video hoặc quét đệ quy tất cả video hỗ trợ trong `data/raw/`:

```text
data/raw/
├── sample.mp4
└── fall/
    └── example.mp4
```

```powershell
python -m src.pose.pipeline --source data\raw\sample.mp4
python -m src.pose.pipeline --source data\raw
```

Lệnh thư mục lần lượt xử lý các file `.avi`, `.m4v`, `.mkv`, `.mov`, `.mp4`, `.mpeg`, `.mpg`, `.wmv`, `.webm`, kể cả trong thư mục con. Kết quả từng video được tạo riêng trong `outputs\pose\`; video gốc không bị sửa. Nếu chưa cần chuẩn hóa, có thể chạy YOLO trực tiếp với video gốc.

### 7. Chuẩn hóa video trong `data/raw`

Để chuyển các video sang MP4, 640×360 và 15 FPS (có thể đổi trong `configs/data.json`), chạy:

```powershell
python -m src.data.pipeline normalize-raw
```

Video đã chuẩn hóa được lưu trong `data\processed\raw\` theo cấu trúc thư mục tương ứng; video gốc trong `data\raw\` được giữ nguyên. Kết quả và lỗi (nếu có) được ghi vào `data\metadata\local\raw_normalization.jsonl` và `raw_normalization_errors.jsonl`. Lệnh không ghi đè đầu ra đã tồn tại; nếu muốn chạy lại, chỉ định thư mục đầu ra mới, ví dụ:

```powershell
python -m src.data.pipeline normalize-raw --output-dir data\processed\raw_v2
```

Chuẩn hóa độc lập này không tự gán nhãn hay tạo train/val/test split. Để chuẩn bị manifest dataset có nhãn, checksum, nhóm bản ghi và split, hãy dùng đầy đủ quy trình `download`, `prepare`, `audit` trong [docs/data_workflow.md](docs/data_workflow.md).

Sau đó có thể chạy YOLO trên toàn bộ video chuẩn hóa:

```powershell
python -m src.pose.pipeline --source data\processed\raw
```

### 8. Chạy trên dữ liệu đã chuẩn bị

Sau khi tạo manifest bằng quy trình dữ liệu:

```powershell
python -m src.pose.pipeline --manifest data\metadata\local\manifest.jsonl --split test
```

Repository hiện cung cấp extraction/tracking và baseline candidate qua CLI; giao diện Streamlit và phân loại violence/fall hoàn chỉnh chưa nằm trong phạm vi triển khai này.

## Hướng dẫn sử dụng

1. Khởi động giao diện Streamlit.
2. Tải video cần kiểm tra lên hệ thống.
3. Chọn ngưỡng confidence và số frame xác nhận.
4. Nhấn chạy phân tích.
5. Theo dõi bounding box, skeleton và nhãn hành động.
6. Kiểm tra danh sách cảnh báo.
7. Tải video kết quả hoặc nhật ký CSV.

## Lưu ý dành cho Developer

- **YOLO Pose không trực tiếp kết luận hành động:** YOLO Pose chỉ cung cấp vị trí người và keypoint. Việc nhận dạng `violence` hoặc `fall` cần phân tích chuỗi nhiều frame bằng LSTM, GRU hoặc luật hình học.

- **Chia dữ liệu đúng cách:** Không chia các frame của cùng một video vào cả tập train và test. Nên chia theo người tham gia, video gốc hoặc bối cảnh quay.

- **Xử lý tracking:** Mỗi người cần được duy trì đúng Track ID. Nếu ID bị thay đổi liên tục, chuỗi keypoint sẽ không chính xác.

- **Lọc cảnh báo:** Không tạo cảnh báo dựa trên một frame. Sự kiện cần xuất hiện liên tục trong một số frame và đạt ngưỡng confidence.

- **Dữ liệu bình thường:** Cần bổ sung các hành động dễ gây nhầm như cúi người, ngồi nhanh, chạy, giơ tay, ôm, khoác vai hoặc ngồi dưới sàn.

- **Đánh giá:** Không chỉ sử dụng Accuracy. Cần báo cáo Precision, Recall, F1-score, confusion matrix, FPS và số cảnh báo sai.

- **GPU:** Nếu sử dụng GPU NVIDIA, cần cài đúng phiên bản PyTorch tương thích với CUDA của máy.

- **Dung lượng GitHub:** Không đẩy video, dataset hoặc model lớn trực tiếp lên GitHub. Các thư mục này cần được cấu hình trong `.gitignore`.

- **Quyền riêng tư:** Không nhận diện khuôn mặt, không lưu tên học sinh và không công khai video lớp học khi chưa có sự đồng thuận.

- **An toàn dữ liệu:** Video mô phỏng bạo lực và té ngã phải được thực hiện trong điều kiện kiểm soát, không tạo va chạm thật hoặc yêu cầu người tham gia thực hiện động tác nguy hiểm.

- **Trách nhiệm sử dụng:** Kết quả chỉ là cảnh báo nghi ngờ. Giáo viên hoặc người quản lý phải xem lại video trước khi đưa ra quyết định.

## Quy trình dữ liệu (issue #1)

Bản triển khai dữ liệu có tại [docs/data_workflow.md](docs/data_workflow.md).
Nguồn và điều khoản: [docs/dataset_sources.md](docs/dataset_sources.md).
Quy tắc nhãn và mốc sự kiện: [docs/annotation_guide.md](docs/annotation_guide.md).

```powershell
python -m pip install -r requirements-data.txt
python -m src.data.pipeline download
python -m src.data.pipeline prepare
python -m src.data.pipeline audit
```

Xem data/metadata/local/audit.json trước khi huấn luyện/đánh giá.
Nhãn cấp clip kế thừa từ nguồn; template sự kiện phải được người xem duyệt.
Video được lưu cục bộ và bị loại khỏi Git. Tải trực tiếp từ tác giả theo catalog.

## YOLO Pose và baseline té ngã (issue #2)

Cài phần phụ thuộc:

```powershell
python -m pip install -r requirements-pose.txt
```

Sau khi chuẩn bị dữ liệu, chạy một video hoặc split từ manifest:

```powershell
python -m src.pose.pipeline --source data\processed\fall\<sample_id>.mp4
python -m src.pose.pipeline --manifest data\metadata\local\manifest.jsonl --split test
```

Kết quả chỉ được tạo mới trong `outputs\pose\`; xem [docs/pose_workflow.md](docs/pose_workflow.md) để biết schema keypoint, ngưỡng heuristic và các giới hạn đánh giá. Baseline là bộ luật sàng lọc candidate, không phải bộ phân loại té ngã đã được huấn luyện/kiểm định.



##  Video Inference and Violence Detection

Phần này hướng dẫn cách sử dụng mô hình đã huấn luyện để phân tích một video mới, không cần đưa video vào quá trình training.

### 1. Chuẩn bị thư mục chứa video bên ngoài

Tại thư mục gốc của project, chạy:

```powershell
New-Item -ItemType Directory -Force -Path .\data\external
```

Đưa video cần kiểm tra vào thư mục:

```text
data/
└── external/
    └── test_video.mp4
```

Video trong `data/external/` chỉ dùng cho inference, không tự động được thêm vào dataset training hoặc validation.

### 2. Kích hoạt môi trường Python

Nếu môi trường ảo chưa được kích hoạt, chạy:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Kiểm tra các tham số inference

Chạy:

```powershell
python -m src.inference --help
```

Hệ thống hỗ trợ các tham số:

| Tham số | Chức năng |
|---|---|
| `--source` | Đường dẫn video đầu vào |
| `--model` | Chọn mô hình `mlp` hoặc `lstm` |
| `--checkpoint` | Chỉ định checkpoint cụ thể (không bắt buộc) |
| `--threshold` | Ngưỡng xác suất cảnh báo |
| `--device` | Thiết bị chạy YOLO, ví dụ `0` hoặc `cpu` |
| `--output-dir` | Thư mục lưu kết quả |

### 4. Chạy mô hình trên video mới

Ví dụ sử dụng LSTM:

```powershell
python -m src.inference --source ".\data\external\test_video.mp4" --model lstm
```

Hoặc sử dụng MLP:

```powershell
python -m src.inference --source ".\data\external\test_video.mp4" --model mlp
```

Mặc định, hệ thống sử dụng checkpoint tốt nhất được tìm thấy của mô hình đã chọn và ngưỡng cảnh báo `0.5`.

### 5. Quy trình xử lý tự động

Sau khi chạy command, hệ thống thực hiện:

1. Load classifier và checkpoint đã huấn luyện.
2. Load YOLO Pose model.
3. Chuẩn hóa video về độ phân giải 640×360 và 15 FPS.
4. Phát hiện người và trích xuất pose bằng YOLO Pose.
5. Theo dõi đối tượng bằng ByteTrack.
6. Trích xuất pose features và motion features.
7. Tổng hợp đặc trưng và đưa vào classifier.
8. Phân loại video thành Normal hoặc Violence.
9. Tính xác suất và kiểm tra điều kiện cảnh báo.
10. Xác định các khoảng thời gian nghi vấn.
11. Lưu kết quả inference.

### 6. Xem kết quả

Sau khi hoàn thành, kết quả được lưu tự động trong:

```text
outputs/
└── inference/
    └── video_name_timestamp/
        ├── normalized_input.mp4
        ├── poses.jsonl
        ├── summary.json
        ├── annotated.mp4
        └── inference_result.json
```

**Các file quan trọng:**

- `inference_result.json`: Nhãn dự đoán, xác suất Normal/Violence, trạng thái cảnh báo và timestamp các đoạn nghi vấn.
- `annotated.mp4`: Video đã được chú thích bằng kết quả pose detection.
- `summary.json`: Thông tin tổng hợp quá trình xử lý video.
- `poses.jsonl`: Dữ liệu pose được trích xuất từ video.

### 7. Ví dụ kết quả

```text
Prediction: VIOLENCE
Normal probability:   8.00%
Violence probability: 92.00%
Alert threshold:      50.00%

Potential violence detected in this video.
```

Ví dụ nội dung timestamp trong JSON:

```json
"suspicious_segments": [
    {
        "start_frame": 180,
        "end_frame": 243,
        "max_violence_probability": 0.92,
        "start_seconds": 12.0,
        "end_seconds": 16.267
    }
]
```

Người kiểm tra có thể mở video và xem lại khoảng thời gian nghi vấn.

**Lưu ý:** Kết quả là dự đoán của mô hình, không phải kết luận chắc chắn rằng hành vi bạo lực đã xảy ra. Người kiểm tra cần xem lại video để xác minh.

Video thử nghiệm, checkpoint và kết quả inference không cần đưa lên GitHub. Chỉ cần chia sẻ source code và hướng dẫn sử dụng.