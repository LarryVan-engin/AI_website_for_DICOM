import pydicom
import numpy as np
import cv2
import os

# Đọc file DICOM
def read_dicom(file_path):
    """Đọc ảnh từ file DICOM và trả về pixel array"""
    ds = pydicom.dcmread(file_path)
    pixel_array = ds.pixel_array
    return pixel_array

# Tạo mask dựa trên ngưỡng
def create_mask(image, threshold=128):
    """Tạo mask nhị phân dựa trên ngưỡng"""
    norm_image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    _, mask = cv2.threshold(norm_image, threshold, 255, cv2.THRESH_BINARY)
    return mask

# Lưu mask thành file
def save_mask(output_path, mask):
    """Lưu mask dưới dạng file ảnh PNG"""
    cv2.imwrite(output_path, mask)
    return output_path  # Trả về đường dẫn để hiển thị trên web
