import cv2
print("OpenCV version:", cv2.__version__)
try:
    count = cv2.cuda.getCudaEnabledDeviceCount()
    print("CUDA enabled devices:", count)
except Exception as e:
    print("CUDA error:", e)
