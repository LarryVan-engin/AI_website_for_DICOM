import pydicom
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from glob import glob

# Đọc file DICOM
def read_dicom(file_path):
    ds = pydicom.dcmread(file_path)
    pixel_array = ds.pixel_array
    return pixel_array

# Tạo mask dựa trên ngưỡng
def create_mask(image, threshold=128):
    norm_image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    _, mask = cv2.threshold(norm_image, threshold, 255, cv2.THRESH_BINARY)
    return mask

# Hiển thị một cặp ảnh gốc và mask
def display_image(original, mask):
    plt.figure(figsize=(10, 5))

    # Ảnh gốc
    plt.subplot(1, 2, 1)
    plt.imshow(original, cmap="gray")
    plt.title("Original DICOM Image")
    plt.axis("off")

    # Mask nhị phân
    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap="gray")
    plt.title("Generated Mask")
    plt.axis("off")

    plt.show()

# Lưu mask thành file
def save_mask(output_path, mask):
    cv2.imwrite(output_path, mask)
    print(f"Mask saved to {output_path}")

# Đường dẫn thư mục chứa file DICOM
input_dicom_dir = 'train\CT_dicom\dicom_dir_full'
output_mask_dir = "train\mask"

# Tạo thư mục lưu mask nếu chưa tồn tại
if not os.path.exists(output_mask_dir):
    os.makedirs(output_mask_dir)

# Lấy danh sách tất cả các file DICOM trong thư mục đầu vào
dicom_files = glob(os.path.join(input_dicom_dir, "*.dcm"))

# Đọc và xử lý tất cả các file DICOM
for dicom_file_path in dicom_files:
    dicom_image = read_dicom(dicom_file_path)
    mask = create_mask(dicom_image, threshold=128)

    # Tạo tên file mask đầu ra
    file_name = os.path.basename(dicom_file_path)
    mask_output_path = os.path.join(output_mask_dir, f"{os.path.splitext(file_name)[0]}_mask.png")

    # Lưu mask vào file
    save_mask(mask_output_path, mask)

    # Hiển thị từng cặp ảnh gốc và mask
   #display_image(dicom_image, mask)
