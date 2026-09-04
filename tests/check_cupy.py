try:
    import cupy as cp
    print("Cupy:", cp.__version__)
except Exception as e:
    print("Cupy error:", e)
try:
    import cv2
    if hasattr(cv2, 'cudabgsegm'):
        print("cv2.cudabgsegm exists!")
    else:
        print("cv2.cudabgsegm DOES NOT exist!")
except Exception as e:
    print("OpenCV error:", e)
