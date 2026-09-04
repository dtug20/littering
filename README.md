# AOD Engine — Hệ thống phát hiện đồ vật bị bỏ lại
### DeepStream SDK (person detection) + MOG2 dual-background (class-agnostic object detection) + MQTT

> **Trash-only strict mode:** phiên bản hiện tại thêm semantic gate bắt buộc
> trước `object_abandoned`: chỉ `garbage_bag`/`loose_waste` đã được model
> custom xác nhận nhiều frame mới được publish. ROI rỗng có nghĩa là không
> detect và model lỗi hoạt động fail-closed. Xem
> [`docs/TRASH_ONLY_DEPLOYMENT.md`](docs/TRASH_ONLY_DEPLOYMENT.md) để đặt model,
> nghiệm thu và tune ngưỡng. Các phần v2 bên dưới được giữ làm mô tả tầng tạo
> ứng viên class-agnostic; chúng không còn là quyết định cảnh báo cuối cùng.

> **v2**: thiết kế lại để giải quyết vấn đề "taxonomy đồ vật quá lớn không
> thể train hết bằng YOLO đa lớp". Xem mục 1b để hiểu lý do & đánh đổi.

## 1. Ý tưởng kiến trúc

```
                 ┌─────────────────────────────────────────┐
 RTSP camera --> │  DeepStream: nvurisrcbin -> nvstreammux  │
                 │  -> nvinfer (YOLO, DUY NHẤT 1 class:     │
                 │     "person") -> nvtracker (NvDCF)       │
                 └───────────────┬───────────────────────────┘
                                 │ probe (mỗi frame): person bbox + object_id
                                 ▼
        ┌─────────────────────────────────────────────────┐
        │ MOG2 dual-background (NGUỒN PHÁT HIỆN VẬT THỂ    │
        │ CHÍNH - class-agnostic, không phân biệt loại)     │
        │  - short-term: bắt foreground mới                 │
        │  - long-term: hấp thụ vật đứng yên lâu vào nền     │
        │  => static foreground = vật mới xuất hiện & đứng  │
        │     yên, CHƯA bị hấp thụ vào nền dài hạn           │
        └───────────────────────┬─────────────────────────┘
                                 ▼
        shape_filters (aspect ratio / solidity / extent)
        - lọc nhiễu KHÔNG CẦN TRAIN: bóng đổ, lá cây, ánh sáng
                                 ▼
        BlobTracker (IoU) - gán object_id ổn định cho blob
                                 ▼
        PersonHistoryBuffer - suy ra "ai vừa đặt đồ xuống"
        bằng lịch sử vị trí person vài giây gần nhất
                                 ▼
        ┌─────────────────────────────────────────────────┐
        │ ObjectStateTracker v2 (state machine per blob_id) │
        │  CONFIRMING -> STATIC_WITH_OWNER/NO_OWNER ->      │
        │  ABANDONED | PENDING_CLAIM -> CLAIMED | BACKGROUND│
        └───────────────────────┬─────────────────────────┘
                                 ▼
        (tuỳ chọn, CHỈ khi có event) zero-shot labeling
        - gắn nhãn mô tả cho UI, KHÔNG ảnh hưởng logic cảnh báo
                                 ▼
                    MQTT publish (bbox realtime + event)
```

## 1b. Vì sao KHÔNG dùng YOLO đa lớp để nhận diện đồ vật

**Vấn đề gốc:** danh sách đồ vật có thể bị bỏ quên là open-set (balo, vali,
hộp, túi ni-lông, mũ bảo hiểm, chai nước, laptop, dù, xe đạp, valy...) —
không thể liệt kê hết, và mỗi lần phát sinh loại đồ vật mới lại phải thu thập
dữ liệu + train lại + đánh giá lại độ chính xác. Đây là bài toán **không phù
hợp với detector closed-set**.

**Giải pháp — tách "phát hiện" khỏi "phân loại":**

| Việc cần làm | Cách tiếp cận cũ (v1) | Cách tiếp cận mới (v2) |
|---|---|---|
| Phát hiện có vật thể bị bỏ lại hay không | YOLO đa lớp (phải train theo từng loại) | MOG2 dual-background — **class-agnostic**, dựa vào tín hiệu chuyển động → đứng yên, áp dụng cho MỌI loại vật kể cả chưa từng thấy |
| Loại nhiễu (bóng đổ, lá cây, ánh sáng) | Ẩn trong confidence threshold của YOLO | `shape_filters.py` — heuristic hình học (aspect ratio/solidity/extent), **không cần train**, áp dụng chung cho mọi loại nhiễu |
| Ai là chủ của vật | bbox overlap giữa object-class và person | `PersonHistoryBuffer` — chỉ cần person (1 class, đã chín) |
| Vật đó cụ thể là gì (để hiển thị UI) | Phải nằm trong tập class đã train | **Zero-shot/open-vocabulary** (CLIP zero-shot, YOLO-World...), sửa `candidate_labels` trong config để thêm loại đồ vật mới, không cần train lại |

**Đánh đổi cần biết:**
- MOG2 không phân biệt được "đồ vật" với "vật cản khác" thuần theo hình dạng
  (vd: một chiếc xe đẩy hàng dừng lại lâu, hoặc rác đọng lại) — `shape_filters`
  giảm nhưng không loại bỏ hoàn toàn false positive dạng này; nếu cần độ
  chính xác cao hơn, có thể bật thêm zero-shot labeling để lọc lần cuối trước
  khi publish alert (đổi lấy độ trễ vài trăm ms/event, chấp nhận được vì tần
  suất event rất thấp).
- MOG nhạy với thay đổi ánh sáng đột ngột / rung camera — vẫn cần camera cố
  định, ít nhiễu như yêu cầu gốc trong bảng test case (rất phù hợp vì hệ
  thống của bạn vốn đã yêu cầu "camera cố định, vùng quan sát rõ").
- Nếu vẫn muốn dùng YOLO đa lớp cho MỘT SỐ loại đồ vật ưu tiên cao (vd:
  balo/vali ở khu vực an ninh trọng điểm) để tăng độ tin cậy, có thể kết hợp
  song song: YOLO detect các class ưu tiên + MOG bắt phần còn lại (open-set) —
  kiến trúc `abandonment_rules.py` đã tách rời nguồn object nên ghép thêm
  nguồn thứ 2 không phá vỡ state machine hiện có.

## 2. Ánh xạ 5 test case trong bảng yêu cầu (v2 - class-agnostic)

| # | Test case | Cơ chế xử lý | Kết quả |
|---|-----------|--------------|---------|
| 1 | Đặt đồ vật xuống rồi rời khỏi khu vực | MOG sinh blob sau khi vật đứng yên → `CONFIRMING` → `STATIC_NO_OWNER` → vượt `abandonment_dwell_seconds` | **Cảnh báo** (`object_abandoned`) |
| 2 | Đặt đồ vật xuống rồi quay lại lấy | người chạm/che vị trí vật → blob biến mất → `PENDING_CLAIM` → không tái xuất hiện → `CLAIMED` | Không cảnh báo |
| 3 | Đứng cạnh đồ vật sau khi đặt xuống | luôn ở `STATIC_WITH_OWNER` (khoảng cách chủ < `owner_distance_threshold_px`) | Không cảnh báo |
| 4 | Đi ngang qua với đồ vật trên tay | vật luôn di chuyển cùng người → **MOG không bao giờ sinh static blob** → không có track nào được tạo | Không cảnh báo |
| 5 | Đồ vật cố định từ trước trong khu vực | blob xuất hiện trong `baseline_learning_seconds` khi khởi động → `BACKGROUND` ngay | Không cảnh báo |

So với v1, case 4 giờ được giải quyết **tự nhiên bởi chính cơ chế MOG**
(không cần logic "carried" tường minh nữa) — vì MOG chỉ sinh static foreground
khi vùng ảnh THỰC SỰ ngừng chuyển động; vật đang được cầm/mang theo luôn di
chuyển cùng người nên không bao giờ thoả điều kiện này.

Logic đầy đủ nằm trong `src/tracking/object_state_tracker.py` +
`src/logic/abandonment_rules.py`. File `tests/test_state_machine.py` mô
phỏng cả 5 case (không cần DeepStream/OpenCV) — chạy
`python3 tests/test_state_machine.py` để tự kiểm chứng, hiện tại
**5/5 PASS**.

Yêu cầu độ chính xác ≥85% và điều kiện "quan sát rõ / kích thước đủ lớn /
không bị che liên tục" được ánh xạ vào các threshold cấu hình được (không
hard-code):
- `min_object_area_ratio`, `min_confidence` (detection) — lọc vật quá nhỏ/mờ.
- `max_occlusion_gap_seconds` — cho phép đám đông che khuất tạm thời mà
  không làm gãy track (đúng test case "Người đi ngang qua với đồ vật trên
  tay" và crowd occlusion nói chung).
- `roi_polygons` (vẽ vùng ảo) — chỉ áp dụng logic cho khu vực hay bị quên đồ.

## 3. Cấu trúc thư mục

```
aod_deepstream/
├── configs/
│   ├── app_config.yaml       # camera, ROI, threshold, shape_filters, ownership, labeling
│   ├── mqtt_config.yaml      # broker host/port, topic, payload template
│   ├── pgie_config.txt       # DeepStream nvinfer (YOLO - CHỈ 1 class "person")
│   └── tracker_config.txt    # DeepStream nvtracker (NvDCF, cho person)
├── src/
│   ├── main.py                        # entry point
│   ├── pipeline/
│   │   ├── deepstream_pipeline.py     # dựng Gst pipeline, probe metadata (person)
│   │   └── probe_callbacks.py         # nối probe -> rule engine -> MQTT
│   ├── detection/
│   │   ├── mog_static_detector.py     # MOG2 dual-background (nguồn object chính)
│   │   ├── shape_filters.py           # lọc hình học class-agnostic, không train
│   │   ├── blob_tracker.py            # gán object_id ổn định cho blob (IoU)
│   │   └── optional_labeling.py       # gắn nhãn zero-shot, chỉ khi có event
│   ├── tracking/
│   │   └── object_state_tracker.py    # state machine per object (v2)
│   ├── logic/
│   │   ├── abandonment_rules.py       # orchestrator
│   │   └── ownership.py               # PersonHistoryBuffer (suy luận chủ sở hữu)
│   ├── mqtt/
│   │   ├── mqtt_client.py             # publish/subscribe tổng quát
│   │   └── cmd_handlers.py            # xử lý lệnh web (get_camera, update_roi...)
│   └── utils/
│       ├── config_loader.py           # YAML + hot-reload
│       └── geometry.py                # IoU, centroid, point-in-polygon
├── tests/
│   └── test_state_machine.py          # mô phỏng 5 test case, không cần DeepStream
└── requirements.txt
```

## 4. Cấu hình MQTT — chỉnh sửa không cần build lại code

Toàn bộ `broker host/port/auth`, `topic`, và **cấu trúc field trong payload**
đều nằm trong `configs/mqtt_config.yaml`, dùng cú pháp `str.format` với
placeholder có sẵn:

```yaml
broker:
  host: "192.168.1.10"      # đổi broker chỉ cần sửa ở đây
  port: 1883

topics:
  publish_bbox: "aod/{camera_id}/bbox"
  publish_event: "aod/{camera_id}/event"
  subscribe_cmd: "aod/{camera_id}/cmd"      # web gửi lệnh xuống (get_camera, update_roi...)
```

Ví dụ đổi cấu trúc payload event (không sửa code) — chỉ cần sửa
`payload_templates.event` trong YAML, ví dụ đổi tên field `object_id` thành
`item_id` và bỏ bớt field không cần:

```yaml
payload_templates:
  event:
    id: "{event_id}"
    cam: "{camera_id}"
    type: "{event_type}"
    item_id: "{object_id}"
    bbox: ["{x1}", "{y1}", "{x2}", "{y2}"]
```

Placeholder khả dụng: `event_id, camera_id, event_type, object_id,
owner_track_id, x1, y1, x2, y2, label, label_confidence,
first_seen_static_at, triggered_at, snapshot_path` (event) và `camera_id,
frame_timestamp, object_id, class_name, x1, y1, x2, y2, state` (bbox, lặp
theo từng object — `class_name` luôn là `"object"` hoặc `"person"` vì tầng
realtime không phân loại chi tiết; `label` trong event mới là nhãn mô tả
chi tiết, chỉ có khi bật `labeling.backend`).

**Lệnh web gửi xuống** (`aod/{camera_id}/cmd`, payload JSON `{"cmd": "...",
"request_id": "..."}`) đã hỗ trợ sẵn: `get_camera`, `update_roi`,
`ack_event` — thêm handler mới trong `src/mqtt/cmd_handlers.py`.

## 5. Chạy thử

```bash
pip install -r requirements.txt --break-system-packages
# pyds/gi cài theo DeepStream SDK đã setup sẵn trên server (không qua pip)

python3 -m src.main --app-config configs/app_config.yaml \
                     --mqtt-config configs/mqtt_config.yaml
```

Chạy test logic độc lập (không cần GPU/DeepStream):
```bash
python3 tests/test_state_machine.py
```

## 6. Việc cần hoàn thiện trước khi lên production

- `configs/pgie_config.txt`: trỏ đúng `.onnx`/`.engine`/`labels_person.txt` —
  chỉ cần model detect person (có thể tái sử dụng model person đã có sẵn từ
  các pipeline khác như vungcam_standard/speed-alert, hoặc pretrained COCO
  person-only), **không cần train riêng cho AOD nữa**.
- `configs/tracker_config.txt` cần file `nvdcf_tracker_config.yml` chuẩn theo
  bản DeepStream SDK đang cài.
- `frame_size_by_camera` trong `main.py` hiện hard-code 1920x1080 — nên lấy
  theo caps thực tế của từng source (`pad.get_current_caps()`).
- MOG chạy trên CPU (OpenCV) qua `pyds.get_nvds_buf_surface` mỗi frame — chi
  phí map GPU→CPU chấp nhận được vì nvinfer giờ chỉ chạy 1 class nhẹ hơn
  nhiều; nếu cần tối ưu thêm, có thể hạ tần suất chạy MOG (mỗi N frame) qua
  cấu hình bổ sung `mog.run_every_n_frames`.
- Snapshot khi bắn `object_abandoned` (field `snapshot_path`) chưa được sinh
  ra trong code mẫu — có thể tái dùng pattern lưu MinIO đã làm ở project VQD.
- `labeling.backend: "clip_zero_shot"` trong `optional_labeling.py` là khung
  sườn tham khảo — cần `pip install open_clip_torch torch` và kiểm thử độ trễ
  thực tế trên GPU trước khi bật cho production; nếu độ trễ không chấp nhận
  được cho luồng cảnh báo realtime, có thể chuyển sang chạy bất đồng bộ
  (publish alert trước, gắn nhãn + publish update sau).
- Giới hạn baseline theo *camera khởi động lần đầu*; nếu muốn baseline theo
  *mỗi lần vẽ lại ROI mới*, publish `update_roi` nên kèm reset lại
  `_CameraMogState.started_at` cho camera đó.
- Nếu cần độ chính xác cao hơn cho một số loại đồ vật ưu tiên (an ninh trọng
  điểm), xem đánh đổi ở mục 1b — có thể ghép thêm YOLO đa lớp cho các class
  đó song song với MOG mà không cần đổi kiến trúc state machine.
