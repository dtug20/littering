# Triển khai phát hiện rác không cần train model

## Kết luận kỹ thuật

Hệ thống dùng hai model pretrained, không fine-tune:

- `yolo11s.engine` trong DeepStream: detect + NvDCF track `person`, `bicycle`,
  `car`, `motorcycle`, `bus`, `truck`.
- `yolov8s-worldv2.onnx`: YOLO-World đã đóng băng text prompt, chỉ chạy ở
  crop ứng viên để phân biệt rác với hard-negative. Runtime không cần
  PyTorch/Ultralytics, chỉ cần ONNX Runtime GPU.

Các prompt positive trong ONNX là `black garbage bag`, `plastic garbage
bag`, `garbage bags on street`, `pile of garbage bags`, `dumped household
waste`, `scattered litter`. Các prompt cạnh tranh gồm balo, vali, túi mua
sắm, thùng carton, thùng rác và người.

> Lưu ý giấy phép: metadata của artifact YOLO-World/Ultralytics ghi
> AGPL-3.0. Trước khi đóng gói sản phẩm thương mại cần xác nhận phương án
> tuân thủ AGPL hoặc license doanh nghiệp với bộ phận pháp chế.

## Hợp đồng quyết định

Một `object_abandoned` chỉ được publish khi đồng thời thỏa:

1. ROI có polygon hợp lệ; ROI rỗng có nghĩa là không phát hiện ở đâu cả.
2. Người được track trong ROI và đứng gần một vị trí ít nhất 2 giây.
3. Người rời vị trí/ROI ít nhất 3 giây. Người chỉ đi ngang không tạo dwell
   zone nên không sinh candidate.
4. Ảnh trước/sau tại vùng mặt đất có thay đổi; contour qua filter kích
   thước, hình học, edge density và MOG tĩnh.
5. Chủ thể rời xa vật đủ `abandonment_dwell_seconds` (hiện tại 5 giây).
6. YOLO-World xác nhận prompt rác ở ít nhất 3/5 crop và thắng prompt âm ít
   nhất `min_target_margin`.

Ứng viên không qua semantic gate chuyển sang `IGNORED`: không publish bbox
rác và không phát event. Lỗi/mất model giữ pending hoặc làm service fail khi
`fail_on_model_error: true`; unknown không bao giờ được đổi thành rác.

Event giữ `event_type: object_abandoned` để tương thích downstream và thêm:

- `incident_type: trash_accumulation`: rác được xác nhận nhưng không có xe
  thỏa điều kiện liên kết.
- `incident_type: illegal_dumping`: có cả `owner_track_id` và xe đã dừng đủ
  lâu trong cùng polygon, gần vị trí rác.
- `related_vehicle_id`, `related_vehicle_class`, `person_dwell_seconds`,
  `vehicle_dwell_seconds` phục vụ truy vết.

Xe chỉ chạy ngang dưới 2 giây không đủ điều kiện liên kết. Lịch sử xe được
giữ 15 giây để vẫn liên kết được khi xe vừa rời khung trước lúc SSIM xác
nhận thay đổi.

## Độ trễ

Với cấu hình hiện tại, phần semantic sau khi đủ dwell chỉ mất khoảng 0,5-1
giây (3 mẫu; inference warm khoảng 6-25 ms/crop trên RTX 3090). Tổng thời
gian cảnh báo sau khi người rời đi xấp xỉ:

```text
departure_confirm_seconds + abandonment_dwell_seconds + semantic_consensus
= 3 + 5 + khoảng 1 = khoảng 9 giây
```

Với cấu hình hiện tại, deployment công bố `X = 10 giây`. Khi thay đổi
`abandonment_dwell_seconds`, phải đo lại false-alert trên video sạch.

## Khởi chạy và dependency

Artifact runtime:

```text
models/yolov8s-worldv2.onnx
SHA256: 21456b406bb35678c7d1f21a696b8e25a5289136007536dc54c2add121f36cff
```

Checksum phải được cập nhật sau mỗi lần export prompt. Image triển khai cần
cài `requirements.txt`; `requirements-export.txt` chỉ dùng ở máy export.
Compose hiện dùng trực tiếp image có sẵn nên nếu container bị recreate, cần
cài dependency vào image/base environment trước khi start.

```bash
docker compose up -d
docker logs -f aod_engine
```

Log đúng phải có `ONNX providers: ['CUDAExecutionProvider', ...]`, load thành
công `yolo11s.engine`, MQTT Connected và số camera active. Dòng `Cập nhật ROI
...: 0 vùng` có nghĩa camera đó đang bị fail-closed và không thể làm test
nghiệm thu cho đến khi web/MQTT cấu hình polygon.

## Tiêu chí nghiệm thu tại camera thật

Không có model zero-shot nào bảo đảm precision/recall chỉ bằng cấu hình.
Logic ở trên bảo đảm điều kiện quyết định; chất lượng nhận dạng phải được đo
ở cấp event trên video của từng camera:

- Setup: ROI sạch/baseline ổn định, đặt túi rác và đống rác đại diện vào ROI.
- Positive: người/xe dừng, đặt rác, rời đi; event trong <= 10 giây sau khi
  rời, crop đúng rác, ID người/xe đúng nếu xe đủ dwell.
- Negative: người và xe đi ngang không dừng; không event.
- Negative: balo, vali, túi mua sắm, carton, bóng đổ, thùng rác cố định;
  không event.
- Soak sạch tối thiểu 24 giờ/camera; mục tiêu false alert <= 1/camera/ngày.
- Mục tiêu ban đầu: event precision >= 95%, recall >= 90%; đo riêng ban đêm,
  mưa, vật nhỏ, che khuất và sát biên ROI.

Nếu không đạt SLA sau prompt/threshold calibration, đó là điểm phải chuyển
sang fine-tune theo domain camera; không nên hạ threshold chỉ để qua một ảnh.

## Kiểm thử logic

```bash
python3 -m unittest tests.test_trash_only_logic -v
python3 tests/test_state_machine.py
```

Các test bao phủ semantic consensus, fail-closed, polygon, reference trước
khi đặt rác, frame throttle, liên kết xe dừng, loại xe đi ngang và 5 lifecycle
case cũ.
