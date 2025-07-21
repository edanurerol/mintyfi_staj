import sys
import torch
import numpy
import cv2  # opencv
import platform

# Python sürümü
print(f"Python sürümü: {sys.version.split()[0]}")

# CUDA ve PyTorch bilgileri
cuda_available = torch.cuda.is_available()
print(f"PyTorch sürümü: {torch.__version__}")
print(f"CUDA kullanılabilir mi?: {cuda_available}")
if cuda_available:
    print(f"CUDA sürümü: {torch.version.cuda}")
    print(f"Kullanılan GPU: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA kullanılamıyor.")

# NumPy sürümü
print(f"NumPy sürümü: {numpy.__version__}")

# OpenCV sürümü
print(f"OpenCV sürümü: {cv2.__version__}")

# İşletim sistemi bilgileri
print(f"İşletim Sistemi: {platform.system()} {platform.release()} ({platform.version()})")
