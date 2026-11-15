# Thực hiện issue #1 — Dữ liệu

## Chuẩn hóa video có sẵn trong `data/raw`

Nếu đã có video trong `data/raw` và chỉ cần chuẩn hóa để YOLO xử lý, chạy từ thư mục gốc:

```powershell
python -m src.data.pipeline normalize-raw
```

Pipeline giữ cấu trúc thư mục, chuẩn hóa thành MP4 640×360 ở 15 FPS, không âm thanh, và không thay đổi video gốc. Chế độ này không tự suy đoán nhãn hoặc tạo split.

## Tải và chuẩn bị dataset AIRTLab

Phiên bản catalog này chỉ dùng 350 video AIRTLab. Nhãn gốc là `non-violent` và `violent`; dự án ánh xạ chúng lần lượt thành `normal` và `violence`. Đây là dataset hai lớp và không chứa video fall. Mô-đun phát hiện ngã trong dự án không đồng nghĩa với việc dataset này có mẫu ngã.

```powershell
python -m pip install -r requirements-data.txt
python -m src.data.pipeline download --dataset airtlab
python -m src.data.pipeline prepare
python -m src.data.pipeline audit
python -m src.data.report
```

Chừa dung lượng cho 350 video gốc và các bản chuẩn hóa. Có retry, file `.part`, kiểm checksum AIRTLab, cache và giới hạn mặc định 64 MiB/file. Dùng `--workers 2` để giảm tải máy; mặc định là 4.

Kết quả gồm `data/raw/{normal,violence}`, `data/processed/{normal,violence}`, manifest và downloads trong `data/metadata/local`, cùng audit và thống kê dưới `docs/reports/`. Tổng byte, thời lượng và số frame phải được tạo lại sau khi chạy trên các file thực tế.

## Chia tập, kiểm tra và nghiệm thu

Pipeline chia theo nhóm sự kiện, liên kết bản trùng và ứng viên gần trùng; không chia frame. Đọc cảnh báo audit và xem lại các ứng viên trước khi nghiệm thu. Báo cáo phải mô tả đúng hạn chế của dataset: video được diễn trong một phòng, không đại diện riêng cho lớp học.

Nếu cần huấn luyện/đánh giá lớp fall, cần bổ sung một nguồn được phê duyệt và lập phiên bản dataset mới có nguồn, nhãn, điều khoản và kiểm tra riêng. Không suy diễn lớp fall từ dữ liệu AIRTLab.
