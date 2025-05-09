# Thư viện pydicom: Dùng để đọc và xử lý các file DICOM (định dạng ảnh y tế, ví dụ: ảnh CT, MRI).
import pydicom

# Thư viện NumPy: Dùng để xử lý mảng đa chiều, thực hiện các phép toán số học trên dữ liệu ảnh.
import numpy as np

# Thư viện OpenCV (cv2): Dùng để xử lý ảnh (đọc, ghi, resize, chuẩn hóa ảnh) và tạo mask.
import cv2

# Thư viện Matplotlib (pyplot): Dùng để vẽ biểu đồ và hiển thị ảnh (ví dụ: hiển thị ảnh DICOM và mask).
import matplotlib.pyplot as plt

# Thư viện os: Dùng để tương tác với hệ điều hành (tạo thư mục, kiểm tra file tồn tại, xử lý đường dẫn).
import os

# Thư viện glob: Dùng để tìm kiếm file theo mẫu (ví dụ: tìm tất cả file *.dcm trong thư mục).
from glob import glob

# Thư viện TensorFlow: Framework học sâu, dùng để xây dựng, huấn luyện và sử dụng mô hình U-Net.
import tensorflow as tf

# Thư viện Keras từ TensorFlow: Cung cấp các lớp (layers) và mô hình (Model) để xây dựng mạng U-Net.
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
# - Input: Tạo lớp đầu vào cho mô hình.
# - Conv2D: Tạo lớp tích chập (convolution) để trích xuất đặc trưng từ ảnh.
# - MaxPooling2D: Tạo lớp gộp (pooling) để giảm kích thước ảnh trong U-Net.
# - UpSampling2D: Tạo lớp mở rộng (upsampling) để khôi phục kích thước ảnh trong U-Net.
# - concatenate: Kết hợp các đặc trưng từ các lớp khác nhau (skip connections) trong U-Net.

# Thư viện Keras (optimizers): Cung cấp bộ tối ưu Adam để huấn luyện mô hình U-Net.
from tensorflow.keras.optimizers import Adam

# Thư viện scikit-learn: Cung cấp hàm train_test_split để chia dữ liệu thành tập huấn luyện và kiểm tra.
from sklearn.model_selection import train_test_split

# Đọc file DICOM
def read_dicom(file_path):
    try:
        ds = pydicom.dcmread(file_path)
        if hasattr(ds, 'pixel_array'):
            return ds.pixel_array
        else:
            print(f"Warning: No pixel data in {file_path}")
            return None
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

# Đọc và tiền xử lý ảnh và mask
def preprocess_image(image, target_size=(256, 256)):
    if image is None:
        return None
    # Chuẩn hóa và resize ảnh
    image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    image = cv2.resize(image, target_size)
    return image[..., np.newaxis]  # Thêm chiều kênh

def preprocess_mask(mask_path, target_size=(256, 256)):
    if not os.path.exists(mask_path):
        print(f"Mask not found: {mask_path}")
        return None
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"Error reading mask: {mask_path}")
        return None
    mask = cv2.resize(mask, target_size)
    mask = (mask > 0).astype(np.float32)  # Chuyển thành nhị phân (0 hoặc 1)
    return mask[..., np.newaxis]

# Xây dựng mô hình U-Net
def build_unet(input_shape=(256, 256, 1)):
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
    
    outputs = Conv2D(1, 1, activation='sigmoid')(c7)
    
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Chuẩn bị dữ liệu huấn luyện
def prepare_dataset(dicom_dir, mask_dir, target_size=(256, 256)):
    dicom_files = glob(os.path.join(dicom_dir, "*.dcm"))
    images, masks = [], []
    
    for dicom_path in dicom_files:
        file_name = os.path.basename(dicom_path)
        mask_path = os.path.join(mask_dir, f"{os.path.splitext(file_name)[0]}_mask.png")
        
        image = read_dicom(dicom_path)
        if image is None:
            continue
        image = preprocess_image(image, target_size)
        mask = preprocess_mask(mask_path, target_size)
        if mask is None:
            continue
        
        images.append(image)
        masks.append(mask)
    
    return np.array(images), np.array(masks)

# Tạo mask bằng U-Net
def create_mask_unet(image, model, target_size=(256, 256)):
    if image is None:
        return None
    # Tiền xử lý ảnh
    input_image = preprocess_image(image, target_size)
    if input_image is None:
        return None
    # Dự đoán mask
    pred = model.predict(np.expand_dims(input_image, axis=0))[0]
    mask = (pred > 0.5).astype(np.uint8) * 255  # Chuyển thành nhị phân (0 hoặc 255)
    # Resize lại kích thước gốc nếu cần
    if image.shape != target_size:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask

# Hiển thị một cặp ảnh gốc và mask
def display_image(original, mask, max_displays=5):
    if original is None or mask is None:
        return
    global display_count
    if display_count >= max_displays:
        return
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(original, cmap="gray")
    plt.title("Original DICOM Image")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap="gray")
    plt.title("U-Net Generated Mask")
    plt.axis("off")
    plt.show()
    display_count += 1

# Lưu mask thành file
def save_mask(output_path, mask):
    if mask is None:
        return
    try:
        cv2.imwrite(output_path, mask)
        print(f"Mask saved to {output_path}")
    except Exception as e:
        print(f"Error saving mask to {output_path}: {e}")

# Hàm chính để xử lý DICOM files
def process_dicom_files(dicom_dir, mask_dir, output_mask_dir, model_path=None, train=True, epochs=20, batch_size=8, max_displays=5):
    global display_count
    display_count = 0
    
    # Tạo thư mục lưu mask
    os.makedirs(output_mask_dir, exist_ok=True)
    
    # Khởi tạo mô hình U-Net
    model = build_unet()
    
    if train:
        # Chuẩn bị dữ liệu
        print("Preparing dataset...")
        images, masks = prepare_dataset(dicom_dir, mask_dir)
        if len(images) == 0:
            print("No valid data found for training.")
            return
        
        # Chia dữ liệu thành train và validation
        X_train, X_val, y_train, y_val = train_test_split(images, masks, test_size=0.2, random_state=42)
        
        # Huấn luyện mô hình
        print("Training U-Net model...")
        model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, verbose=1)
        
        # Lưu mô hình
        model.save(model_path)
        print(f"Model saved to {model_path}")
    else:
        # Tải mô hình đã huấn luyện
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path)
            print(f"Loaded model from {model_path}")
        else:
            print(f"Model path {model_path} does not exist.")
            return
    
    # Xử lý các file DICOM
    dicom_files = glob(os.path.join(dicom_dir, "*.dcm"))
    for dicom_file_path in dicom_files:
        dicom_image = read_dicom(dicom_file_path)
        mask = create_mask_unet(dicom_image, model)
        
        if mask is not None:
            # Tạo tên file mask đầu ra
            file_name = os.path.basename(dicom_file_path)
            mask_output_path = os.path.join(output_mask_dir, f"{os.path.splitext(file_name)[0]}_mask.png")
            
            # Lưu và hiển thị mask
            save_mask(mask_output_path, mask)
            display_image(dicom_image, mask, max_displays)

# Đường dẫn và tham số
input_dicom_dir = os.path.join('train\CT_dicom\dicom_dir_full')
mask_dir = os.path.join('train\mask')  # Thư mục chứa ground truth masks
output_mask_dir = 'mask'
model_path = 'unet_model.h5'
train_model = True  # Đặt False để chỉ sử dụng mô hình đã huấn luyện
epochs = 20
batch_size = 8
max_displays = 5

# Chạy chương trình
process_dicom_files(input_dicom_dir, mask_dir, output_mask_dir, model_path, train=train_model, epochs=epochs, batch_size=batch_size, max_displays=max_displays)