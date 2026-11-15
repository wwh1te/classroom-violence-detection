# Phiếu kiểm duyệt dữ liệu — issue #1

Phạm vi hiện tại: AIRTLab, nhãn gốc `non-violent`/`violent`, ánh xạ nội bộ `normal`/`violence`.

## Kiểm duyệt nhãn và chất lượng

Chưa ghi nhận kết quả kiểm duyệt thủ công cho phiên bản catalog hiện tại. Chỉ đánh dấu video đã xem sau khi mở đúng file và ghi nhận xét cụ thể. Không sử dụng kết quả kiểm duyệt của một nguồn dữ liệu đã bị loại để kết luận cho phiên bản hiện tại.

| Video đã xem | Nhãn nguồn | Mở được/chất lượng | Nhận xét nhãn | Quyết định |
| --- | --- | --- | --- | --- |
| Chưa kiểm tra | — | — | — | Chờ kiểm duyệt |

## Kết luận bàn giao

- Dataset được chọn cho issue này có hai nhãn AIRTLab: `non-violent` → `normal`; `violent` → `violence`.
- Dataset hiện tại không có video fall; không dùng nó làm ground truth để huấn luyện hoặc đánh giá ngã.
- Chưa nghiệm thu nhãn thủ công hoặc nhãn thời gian.
- Chạy lại `download`, `prepare`, `audit`; sau đó kiểm tra các ứng viên gần trùng và cập nhật phiếu bằng kết quả thực tế.
