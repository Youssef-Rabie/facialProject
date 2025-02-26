import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# تحميل النموذج المدرب
model = load_model('emotion_detection_model.keras')

# تعريف أسماء المشاعر
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# فتح الكاميرا
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # تحويل الصورة إلى تدرج الرمادي للبحث عن الوجوه فقط
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]  # قص الوجه
        roi = cv2.resize(roi, (48, 48))  # إعادة تحجيم الصورة
        roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2RGB)  # تحويل الصورة إلى 3 قنوات
        roi = np.array(roi, dtype='float32') / 255.0  # تطبيع الصورة
        roi = np.expand_dims(roi, axis=0)  # إضافة بعد الدُفعة

        # التنبؤ بالمشاعر
        prediction = model.predict(roi)
        emotion = emotion_labels[np.argmax(prediction)]

        # رسم مستطيل حول الوجه
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # رسم النص فوق المستطيل
        text_size = cv2.getTextSize(emotion, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
        text_x = x
        text_y = y - 20 if y - 20 > 20 else y + 20  # تحريك النص للأعلى أو الأسفل حسب الموقع

        cv2.rectangle(frame, (text_x, text_y - text_size[1] - 10), (text_x + text_size[0] + 10, text_y), (0, 255, 0), -1)
        cv2.putText(frame, emotion, (text_x + 5, text_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

    cv2.imshow('Emotion Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
