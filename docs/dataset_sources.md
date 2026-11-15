# Nguồn dữ liệu và điều khoản sử dụng

Ngày kiểm tra: 17/10/2026. Phạm vi: nghiên cứu và giáo dục. Điều khoản dữ liệu độc lập với giấy phép mã nguồn.

| Nguồn chính thức | Nhãn gốc | Nhãn nội bộ dự án | Điều khoản đã kiểm tra |
| --- | --- | --- | --- |
| [AIRTLab](https://github.com/airtlab/A-Dataset-for-Automatic-Violence-Detection-in-Videos/tree/1f7747e104301ccaa82ef5a2f6804b51ced1c398) | `non-violent` | `normal` | Dataset Release Agreement cho phép nghiên cứu và giáo dục, yêu cầu trích dẫn. Không gán giấy phép MIT/CC cho video; chưa xác lập quyền thương mại hoặc tái phân phối. |
| AIRTLab | `violent` | `violence` | Cùng điều khoản và nguồn như trên. |
| [RWF-2000](https://github.com/mchengny/RWF2000-Video-Database-for-Violence-Detection) | Chưa dùng | — | Không tải trong phiên bản này; không thay bằng bản mirror không rõ quyền. |

AIRTLab có 350 clip từ hai camera: 120 clip non-violent và 230 clip violent. Cùng nhãn và số clip ở hai camera là cùng sự kiện; metadata ghép chúng bằng `group_id`. Nhóm non-violent có các hành vi dễ gây false positive như ôm, vỗ tay, ăn mừng và cử chỉ mạnh. Dataset được diễn trong một phòng, không phải video lớp học.

`data/metadata/catalog.jsonl` lưu URL, nhãn nội bộ, nhãn gốc, camera, `group_id` và điều khoản. AIRTLab được ghim theo commit `1f7747e104301ccaa82ef5a2f6804b51ced1c398`; pipeline đối chiếu kích thước và SHA-1 Git blob trước khi chấp nhận tải. Không đưa video vào gói mã nguồn công khai; khả năng tải công khai không tự cấp quyền tái phân phối.

## Trích dẫn

Bianculli, M., Falcionelli, N., Sernani, P., Tomassini, S., Contardo, P., Lombardi, M., & Dragoni, A. F. (2020). A dataset for automatic violence detection in videos. *Data in Brief, 33*, 106587. https://doi.org/10.1016/j.dib.2020.106587
