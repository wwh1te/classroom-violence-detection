# Mô tả dữ liệu và kết quả chuẩn bị

Catalog được giới hạn ở AIRTLab. Nhãn nguồn được giữ riêng với nhãn nội bộ: `non-violent` → `normal`, `violent` → `violence`.
Dataset AIRTLab có hai lớp; không có ví dụ fall.

Catalog có **350 video** trong **175 nhóm sự kiện**.

| Nhãn nguồn | Video |
| --- | ---: |
| non-violent | 120 |
| violent | 230 |

## Chia tập

| Tập | normal | violence | Tổng |
| --- | ---: | ---: | ---: |
| train | 84 | 162 | 246 |
| val | 18 | 34 | 52 |
| test | 18 | 34 | 52 |

## Trạng thái và giới hạn

Các tổng dung lượng, thời lượng và số frame chưa được tính lại cho catalog hiện tại; cần tải và chuẩn hóa lại video trước khi báo cáo các chỉ số đó.
Nhãn cấp clip cần được nhóm kiểm duyệt. AIRTLab được diễn trong một phòng, không đại diện riêng cho lớp học.
Dataset này không có ví dụ fall; cần nguồn được phê duyệt trước khi huấn luyện hoặc đánh giá lớp fall.
