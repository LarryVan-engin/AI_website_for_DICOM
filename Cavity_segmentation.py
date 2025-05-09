import pydicom
import nibabel as nib
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import os

# Hàm xây dựng mô hình U-Net
def build_unet(input_shape=(256, 256, 1), num_classes=4):
    inputs = Input(input_shape)
    
    # Encoder
    c1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    c1 = Conv2D(64, 3, activation='relu', padding='same')(c1)
    p1 = MaxPooling2D(pool_size=(2, 2))(c1)
    
    c2 = Conv2D(128, 3, activation='relu', padding='same')(p1)
    c2 = Conv2D(128, 3, activation='relu', padding='same')(c2)
    p2 = MaxPooling2D(pool_size=(2, 2))(c2)
    
    c3 = Conv2D(256, 3, activation='relu', padding='same')(p2)
    c3 = Conv2D(256, 3, activation='relu', padding='same')(c3)
    p3 = MaxPooling2D(pool_size=(2, 2))(c3)
    
    # Bottleneck
    c4 = Conv2D(512, 3, activation='relu', padding='same')(p3)
    c4 = Conv2D(512, 3, activation='relu', padding='same')(c4)
    
    # Decoder
    u5 = UpSampling2D(size=(2, 2))(c4)
    u5 = concatenate([u5, c3])
    c5 = Conv2D(256, 3, activation='relu', padding='same')(u5)
    c5 = Conv2D(256, 3, activation='relu', padding='same')(c5)
    
    u6 = UpSampling2D(size=(2, 2))(c5)
    u6 = concatenate([u6, c2])
    c6 = Conv2D(128, 3, activation='relu', padding='same')(u6)
    c6 = Conv2D(128, 3, activation='relu', padding='same')(c6)
    
    u7 = UpSampling2D(size=(2, 2))(c6)
    u7 = concatenate([u7, c1])
    c7 = Conv2D(64, 3, activation='relu', padding='same')(u7)
    c7 = Conv2D(64, 3, activation='relu', padding='same')(c7)
    
    outputs = Conv2D(num_classes, 1, activation='softmax')(c7)
    
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Hàm đọc và tiền xử lý hình ảnh DICOM hoặc NIfTI
def read_image(file_path):
    try:
        if file_path.endswith(('.dcm', '.DCM')):
            # Đọc file DICOM
            dicom_file = pydicom.dcmread(file_path)
            image = dicom_file.pixel_array
            # Nếu ảnh có nhiều slice, lấy slice giữa
            if len(image.shape) > 2:
                image = image[image.shape[0] // 2]
        elif file_path.endswith(('.nii', '.nii.gz')):
            # Đọc file NIfTI
            nii_file = nib.load(file_path)
            image = nii_file.get_fdata()
            # Nếu ảnh 3D, lấy slice giữa
            if len(image.shape) > 2:
                image = image[:, :, image.shape[2] // 2]
            # Xoay ảnh nếu cần (NIfTI thường có định hướng khác)
            image = np.rot90(image)
        else:
            raise ValueError("Định dạng file không được hỗ trợ. Sử dụng .dcm hoặc .nii/.nii.gz")
        
        # Chuẩn hóa và resize ảnh
        image = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-6)  # Chuẩn hóa [0,1]
        image = cv2.resize(image, (256, 256))  # Resize về kích thước U-Net
        image = np.expand_dims(image, axis=-1)  # Thêm chiều channel
        return image
    except Exception as e:
        messagebox.showerror("Lỗi", f"Không thể đọc file: {str(e)}")
        return None

# Hàm giả lập phân đoạn (vì không có mô hình đã huấn luyện)
def simulate_segmentation(image):
    # Giả lập kết quả phân đoạn: tạo mask với 4 lớp (background, gan, dạ dày, tụy)
    mask = np.zeros((256, 256, 4))
    # Ví dụ: gán ngẫu nhiên các vùng
    mask[50:150, 50:150, 1] = 1  # Gan
    mask[150:200, 100:200, 2] = 1  # Dạ dày
    mask[80:120, 160:200, 3] = 1  # Tụy
    mask[:, :, 0] = 1 - np.sum(mask[:, :, 1:], axis=-1)  # Background
    return mask

# Hàm hiển thị kết quả phân đoạn
def display_results(image, mask):
    # Mô tả các nội tạng
    organ_descriptions = {
        0: "Background: Vùng không thuộc nội tạng.",
        1: "Gan: Cơ quan lớn nhất trong ổ bụng, tham gia chuyển hóa và giải độc.",
        2: "Dạ dày: Cơ quan tiêu hóa, phân hủy thức ăn bằng axit và enzyme.",
        3: "Tụy: Sản xuất insulin và các enzyme tiêu hóa."
    }
    
    # Tạo cửa sổ Tkinter
    window = tk.Tk()
    window.title("Segmentation Results")
    window.geometry("800x600")
    
    # Hiển thị hình ảnh gốc
    image_display = (image[:, :, 0] * 255).astype(np.uint8)
    img_pil = Image.fromarray(image_display)
    img_tk = ImageTk.PhotoImage(img_pil)
    tk.Label(window, text="Hình ảnh gốc (DICOM/NIfTI)", font=("Arial", 12)).pack()
    tk.Label(window, image=img_tk).pack()
    
    # Hiển thị mask phân đoạn
    mask_display = np.argmax(mask, axis=-1).astype(np.uint8) * 80  # Nhân để dễ nhìn
    mask_pil = Image.fromarray(mask_display)
    mask_tk = ImageTk.PhotoImage(mask_pil)
    tk.Label(window, text="Kết quả phân đoạn", font=("Arial", 12)).pack()
    tk.Label(window, image=mask_tk).pack()
    
    # Hiển thị mô tả nội tạng
    tk.Label(window, text="Mô tả các nội tạng:", font=("Arial", 12, "bold")).pack(pady=10)
    for idx, desc in organ_descriptions.items():
        tk.Label(window, text=f"{desc}", font=("Arial", 10), wraplength=700).pack(anchor="w", padx=10)
    
    # Nút đóng cửa sổ
    tk.Button(window, text="Đóng", command=window.destroy).pack(pady=20)
    
    window.mainloop()

def main():
    # Đường dẫn đến file DICOM hoặc NIfTI (thay đổi theo file của bạn)
    file_path = "train\Abdominal CT scans\images\image_002.nii\image_002_0000.nii"  # Hoặc .nii/.nii.gz
    
    if not os.path.exists(file_path):
        messagebox.showerror("Lỗi", f"File {file_path} không tồn tại.")
        return
    
    # Đọc và tiền xử lý hình ảnh
    image = read_image(file_path)
    if image is None:
        return
    
    # Xây dựng mô hình U-Net (chưa huấn luyện)
    model = build_unet()
    
    # Giả lập phân đoạn (thay thế bằng model.predict(image) nếu có mô hình đã huấn luyện)
    mask = simulate_segmentation(image)
    
    # Hiển thị kết quả
    display_results(image, mask)

if __name__ == "__main__":
    main()