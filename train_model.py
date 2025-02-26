import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import cv2

# تحميل البيانات من CSV
df = pd.read_csv('fer2013.csv')

# تحويل البيانات إلى مصفوفات NumPy
X = np.array([np.fromstring(row, dtype=np.uint8, sep=' ').reshape(48, 48, 1) for row in df['pixels']])
y = tf.keras.utils.to_categorical(df['emotion'], num_classes=7)  # 7 مشاعر

# تقسيم البيانات إلى تدريب واختبار
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# إعادة تشكيل الصور لتناسب MobileNetV2
X_train = np.stack([cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) for img in X_train]) / 255.0
X_test = np.stack([cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) for img in X_test]) / 255.0

# تحميل MobileNetV2 كقاعدة للنموذج
base_model = MobileNetV2(input_shape=(48, 48, 3), include_top=False, weights='imagenet')
base_model.trainable = False  # تجميد الطبقات الأساسية

# إنشاء النموذج
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')  # 7 مشاعر
])

# تجميع النموذج
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# تحسينات التدريب
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

# تدريب النموذج
model.fit(X_train, y_train, epochs=30, batch_size=64, validation_data=(X_test, y_test),
          callbacks=[early_stopping, reduce_lr])

# حفظ النموذج
model.save('emotion_detection_model.keras')

print("✅ النموذج تم تدريبه وحفظه بنجاح!")
