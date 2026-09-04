"""
Xây dựng pipeline DeepStream cho hệ thống AOD (Abandoned Object Detection).

Kiến trúc pipeline (multi-source):

  nvurisrcbin (mỗi camera) --> nvstreammux (batch) --> nvinfer (pgie: YOLO
  person + object classes) --> nvtracker (NvDCF, gán object_id ổn định
  xuyên suốt để phân biệt CARRIED/DROPPED_PENDING) --> nvvideoconvert -->
  nvdsosd --> sink

  Probe được gắn ở src pad của nvtracker (sau khi có object_id) để:
    1. Trích xuất NvDsObjectMeta (bbox, class, confidence, object_id)
    2. (Tuỳ chọn) lấy frame BGR từ NvBufSurface để chạy MOG bổ trợ
    3. Gọi AbandonmentRuleEngine.process_frame(...) (qua on_batch_meta_cb)
    4. Publish MQTT (bbox realtime + event khi có)

Lưu ý: module này giả định môi trường đã cài DeepStream SDK + pyds + gi
(giống stack hiện có trong các pipeline DeepStream khác - vungcam_standard,
weapon-detection, speed-alert...).
"""
import sys
import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib  # noqa: E402

import pyds  # noqa: E402

Gst.init(None)


class DeepStreamAodPipeline:
    def __init__(self, cameras: list, pgie_config_path: str, tracker_config_path: str,
                 on_batch_meta_cb, batched_push_timeout=40000, width=1920, height=1080):
        """
        cameras: list[dict] {camera_id, uri}
        on_batch_meta_cb: callable(camera_id, frame_num, detections, source_id,
                          gst_buffer, batch_id) - do main.py truyền vào, nối
                          tới FrameHandler (rule engine + MQTT).
        """
        self.cameras = cameras
        self.on_batch_meta_cb = on_batch_meta_cb
        self.pipeline = Gst.Pipeline.new("aod-pipeline")
        self.source_id_to_camera = {}

        self.streammux = Gst.ElementFactory.make("nvstreammux", "streammux")
        self.streammux.set_property("batch-size", max(1, len(cameras)))
        self.streammux.set_property("width", width)
        self.streammux.set_property("height", height)
        self.streammux.set_property("batched-push-timeout", batched_push_timeout)
        self.streammux.set_property("live-source", 1)
        self.streammux.set_property("nvbuf-memory-type", 3)
        self.pipeline.add(self.streammux)

        for idx, cam in enumerate(cameras):
            self._add_source(idx, cam)

        self.pgie = self._make("nvinfer", "pgie", {"config-file-path": pgie_config_path})
        self.tracker = self._make("nvtracker", "tracker")
        self._configure_tracker(tracker_config_path)

        self.tee = self._make("tee", "tee")
        self.queue_main = self._make("queue", "queue_main")
        self.queue_mog = self._make("queue", "queue_mog")

        # Nhánh detector phụ: 960x540 giữ đủ chi tiết cho túi nhỏ; YOLO-World
        # sẽ chia tile vuông trước inference thay vì ép toàn cảnh xuống 384.
        self.nvvidconv_mog = self._make("nvvideoconvert", "convertor_mog")
        self.nvvidconv_mog.set_property("nvbuf-memory-type", 3)
        self.capsfilter_mog = self._make("capsfilter", "capsfilter_mog")
        caps_mog = Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA, width=960, height=540")
        self.capsfilter_mog.set_property("caps", caps_mog)
        self.sink_mog = self._make("fakesink", "sink_mog", {"sync": 0, "async": 0})
        
        self.nvvidconv = self._make("nvvideoconvert", "convertor")
        self.nvvidconv.set_property("nvbuf-memory-type", 3) # NVBUF_MEM_CUDA_UNIFIED
        
        self.capsfilter = self._make("capsfilter", "capsfilter")
        caps = Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA")
        self.capsfilter.set_property("caps", caps)
        
        self.nvosd = self._make("nvdsosd", "onscreendisplay")
        self.sink = self._make("fakesink", "sink", {"sync": 0})

        for el in (self.tee, self.queue_main, self.queue_mog, self.nvvidconv_mog, self.capsfilter_mog, self.sink_mog):
            self.pipeline.add(el)

        for el in (self.pgie, self.tracker, self.nvvidconv, self.capsfilter, self.nvosd, self.sink):
            self.pipeline.add(el)

        self.streammux.link(self.pgie)
        self.pgie.link(self.tracker)
        self.tracker.link(self.tee)

        # Link nhánh chính
        self.tee.link(self.queue_main)
        self.queue_main.link(self.nvvidconv)
        self.nvvidconv.link(self.capsfilter)
        self.capsfilter.link(self.nvosd)
        self.nvosd.link(self.sink)

        # Link nhánh MOG
        self.tee.link(self.queue_mog)
        self.queue_mog.link(self.nvvidconv_mog)
        self.nvvidconv_mog.link(self.capsfilter_mog)
        self.capsfilter_mog.link(self.sink_mog)

        # gắn probe vào src pad của capsfilter_mog để lấy frame nhỏ 640x360
        caps_src_pad = self.capsfilter_mog.get_static_pad("src")
        caps_src_pad.add_probe(Gst.PadProbeType.BUFFER, self._tracker_src_pad_probe, 0)

        self.loop = None

    # ------------------------------------------------------------------
    def _make(self, factory, name, props=None):
        el = Gst.ElementFactory.make(factory, name)
        if el is None:
            raise RuntimeError(f"Không tạo được element '{factory}' ({name})")
        for k, v in (props or {}).items():
            el.set_property(k, v)
        return el

    def _add_source(self, idx, cam):
        source_bin = Gst.ElementFactory.make("nvurisrcbin", f"source-bin-{idx}")
        if source_bin is None:
            raise RuntimeError(f"Không tạo được nvurisrcbin cho {cam['camera_id']}")
        source_bin.set_property("uri", cam["uri"])
        source_bin.set_property("rtsp-reconnect-interval", 10)
        source_bin.set_property("cudadec-memtype", 0)
        # Giảm frame rate giải mã: drop 1 frame sau mỗi frame giữ lại (tức là giảm 1 nửa FPS, khoảng 10-15fps)
        source_bin.set_property("drop-frame-interval", 2)
        self.pipeline.add(source_bin)

        sinkpad = self.streammux.get_request_pad(f"sink_{idx}")
        source_bin.connect("pad-added", self._on_source_pad_added, sinkpad)
        self.source_id_to_camera[idx] = cam["camera_id"]

    @staticmethod
    def _on_source_pad_added(bin_, pad, sinkpad):
        caps = pad.query_caps(None)
        if caps.to_string().startswith("video"):
            pad.link(sinkpad)

    def _configure_tracker(self, tracker_config_path):
        import configparser
        cp = configparser.ConfigParser()
        cp.read(tracker_config_path)
        section = cp["tracker"]
        self.tracker.set_property("tracker-width", int(section["tracker-width"]))
        self.tracker.set_property("tracker-height", int(section["tracker-height"]))
        self.tracker.set_property("gpu-id", int(section.get("gpu-id", 0)))
        self.tracker.set_property("ll-lib-file", section["ll-lib-file"])
        if "ll-config-file" in section:
            import os
            config_path = section["ll-config-file"]
            if not os.path.isabs(config_path):
                base_dir = os.path.dirname(os.path.abspath(tracker_config_path))
                config_path = os.path.join(base_dir, config_path)
            self.tracker.set_property("ll-config-file", os.path.abspath(config_path))

    # ------------------------------------------------------------------
    def _tracker_src_pad_probe(self, pad, info, user_data):
        gst_buffer = info.get_buffer()
        if not gst_buffer:
            return Gst.PadProbeReturn.OK

        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        l_frame = batch_meta.frame_meta_list
        while l_frame is not None:
            try:
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            except StopIteration:
                break

            camera_id = self.source_id_to_camera.get(frame_meta.source_id, "unknown")

            detections = []
            l_obj = frame_meta.obj_meta_list
            while l_obj is not None:
                try:
                    obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                except StopIteration:
                    break
                rect = obj_meta.rect_params
                detections.append({
                    "object_id": str(obj_meta.object_id),
                    "class_name": obj_meta.obj_label,
                    "confidence": obj_meta.confidence,
                    "bbox": (
                        rect.left, rect.top,
                        rect.left + rect.width, rect.top + rect.height,
                    ),
                })
                try:
                    l_obj = l_obj.next
                except StopIteration:
                    break

            # frame_bgr = None theo mặc định; nếu cần MOG, FrameHandler sẽ lấy qua
            # pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id) rồi
            # cv2.cvtColor(..., cv2.COLOR_RGBA2BGR) - chi phí map GPU->CPU nên chỉ
            # bật khi mog.enable = true trong app_config.yaml.
            self.on_batch_meta_cb(camera_id, frame_meta.frame_num, detections,
                                   frame_meta.source_id, gst_buffer, frame_meta.batch_id)

            try:
                l_frame = l_frame.next
            except StopIteration:
                break

        return Gst.PadProbeReturn.OK

    # ------------------------------------------------------------------
    def run(self):
        self.pipeline.set_state(Gst.State.PLAYING)
        self.loop = GLib.MainLoop()
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._on_bus_message)
        try:
            self.loop.run()
        except KeyboardInterrupt:
            pass
        finally:
            self.pipeline.set_state(Gst.State.NULL)

    def _on_bus_message(self, bus, message):
        t = message.type
        if t == Gst.MessageType.EOS:
            self.loop.quit()
        elif t == Gst.MessageType.ERROR:
            err, dbg = message.parse_error()
            print(f"[Pipeline ERROR] {err}: {dbg}", file=sys.stderr)
            self.loop.quit()
        return True
