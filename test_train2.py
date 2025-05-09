# Xây dựng mô hình
import tensorflow as tf
from tensorflow.keras import layers, Model

def unet_model(input_size=(256, 256, 1)):
    inputs = tf.keras.Input(input_size)

    # Encoder
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = layers.Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = layers.Conv2D(512, 3, activation='relu', padding='same')(conv4)
    drop4 = layers.Dropout(0.5)(conv4)
    pool4 = layers.MaxPooling2D(pool_size=(2, 2))(drop4)

    # Bottleneck
    conv5 = layers.Conv2D(1024, 3, activation='relu', padding='same')(pool4)
    conv5 = layers.Conv2D(1024, 3, activation='relu', padding='same')(conv5)
    drop5 = layers.Dropout(0.5)(conv5)

    # Decoder
    up6 = layers.Conv2DTranspose(512, 2, strides=(2, 2), padding='same')(drop5)
    up6 = layers.concatenate([up6, drop4])
    conv6 = layers.Conv2D(512, 3, activation='relu', padding='same')(up6)
    conv6 = layers.Conv2D(512, 3, activation='relu', padding='same')(conv6)

    up7 = layers.Conv2DTranspose(256, 2, strides=(2, 2), padding='same')(conv6)
    up7 = layers.concatenate([up7, conv3])
    conv7 = layers.Conv2D(256, 3, activation='relu', padding='same')(up7)
    conv7 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv7)

    up8 = layers.Conv2DTranspose(128, 2, strides=(2, 2), padding='same')(conv7)
    up8 = layers.concatenate([up8, conv2])
    conv8 = layers.Conv2D(128, 3, activation='relu', padding='same')(up8)
    conv8 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv8)

    up9 = layers.Conv2DTranspose(64, 2, strides=(2, 2), padding='same')(conv8)
    up9 = layers.concatenate([up9, conv1])
    conv9 = layers.Conv2D(64, 3, activation='relu', padding='same')(up9)
    conv9 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv9)

    # Output layer
    outputs = layers.Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model

model = unet_model()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
# Chuẩn bị dữ liệu
import numpy as np
import matplotlib.pyplot as plt

# Giả sử bạn đã có dữ liệu X_train, y_train, X_val, y_val
# X_train: Hình ảnh MRI
# y_train: Mask tương ứng

# Ví dụ:
# X_train = np.load('X_train.npy')
# y_train = np.load('y_train.npy')

# Tạo dữ liệu giả lập với kích thước phù hợp
X_train = np.random.rand(100, 256, 256, 1)  # 100 hình ảnh MRI ngẫu nhiên
y_train = np.random.randint(0, 2, (100, 256, 256, 1))  # 100 mặt nạ (mask) nhị phân

X_val = np.random.rand(20, 256, 256, 1)  # 20 hình ảnh validation
y_val = np.random.randint(0, 2, (20, 256, 256, 1))  # 20 mặt nạ validation

# Hiển thị một số hình ảnh và mặt nạ
plt.figure(figsize=(10, 10))
for i in range(1, 5):
    plt.subplot(2, 2, i)
    plt.imshow(X_train[i], cmap='gray')
    plt.title('MRI Image')
    plt.axis('off')
    plt.show()

plt.figure(figsize=(10, 10))
for i in range(1, 5):
    plt.subplot(2, 2, i)
    plt.imshow(y_train[i], cmap='gray')
    plt.title('Mask')
    plt.axis('off')
    plt.show()
 # Huấn luyện
history = model.fit(X_train, y_train, batch_size=16, epochs=50, validation_data=(X_val, y_val))
# Đánh giá
# Vẽ đồ thị loss và accuracy
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
 
 # Dự đoán
# Giả sử bạn có một hình ảnh MRI mới
new_mri = np.expand_dims(X_val[0], axis=0)
predicted_mask = model.predict(new_mri)

# Hiển thị kết quả dự đoán
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(X_val[0], cmap='gray')
plt.title('Original MRI')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(np.squeeze(predicted_mask), cmap='gray')
plt.title('Predicted Mask')
plt.axis('off')
plt.show()
model.save('brain_tumor_segmentation_model.h5')