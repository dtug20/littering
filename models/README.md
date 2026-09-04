# Models Directory

Thư mục này chứa các tệp mô hình cho hệ thống AOD (Abandonment Object Detection).

Do dung lượng file weights lớn (> 20-50MB) và file `.engine` phụ thuộc chặt chẽ vào kiến trúc GPU cụ thể, các file sau được loại khỏi Git repository:
- `*.pt`
- `*.onnx`
- `*.engine`

---

## 1. Các mô hình sử dụng trong hệ thống

1. **YOLO11s (PGIE - DeepStream)**:
   - File cấu hình: `configs/pgie_config.txt`
   - File nhãn: `models/labels.txt` (hoặc `models/DeepStream-Yolo/labels.txt`)
   - File engine: `models/yolo11s.engine`
   - Tải pretrained:
     ```bash
     wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt
     ```
   - Export ONNX theo tài liệu trong `models/DeepStream-Yolo/docs/YOLO11.md`:
     ```bash
     python3 models/DeepStream-Yolo/utils/export_yolo11.py -w yolo11s.pt --dynamic
     mv yolo11s.onnx models/
     ```

2. **YOLO-World (Open-vocabulary / Semantic Gate)**:
   - File model ONNX: `models/yolov8s-worldv2-bags-640.onnx` hoặc `models/yolov8s-worldv2.onnx`
   - Xem chi tiết tại [`docs/TRASH_ONLY_DEPLOYMENT.md`](../docs/TRASH_ONLY_DEPLOYMENT.md).
   - Export offline theo `requirements-export.txt`:
     ```bash
     pip install -r requirements-export.txt
     ```

3. **Biên dịch DeepStream-Yolo custom library**:
   ```bash
   cd models/DeepStream-Yolo/nvdsinfer_custom_impl_Yolo
   CUDA_VER=12.x make   # Thay 12.x bằng CUDA version hiện tại
   ```
