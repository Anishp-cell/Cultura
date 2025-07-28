import os
import cv2
import numpy as np
import tensorflow as tf
import time
from datetime import datetime

# Suppress TensorFlow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

# Load emotion detection model (Keras)
model = tf.keras.models.load_model("emo_model/emotion_model_v6.keras")

# Emotion labels
emotion_labels = ["Angry", "Happy", "Neutral", "Sad", "Surprised"]

# Load OpenCV Haar cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# === Helpers ===
def apply_clahe(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))
    if len(faces) == 0:
        return None, None
    x, y, w, h = faces[0]
    face = gray[y:y+h, x:x+w]
    face = cv2.resize(face, (48, 48))
    face = apply_clahe(face)
    face = (face - np.mean(face)) / np.std(face)
    face = np.expand_dims(face, axis=0).reshape(-1, 48, 48, 1)
    return face, (x, y, w, h)

# === Main Camera Loop ===
cap = cv2.VideoCapture(0)

emotion_history = []
emotion_label = "None"

print("[INFO] Starting emotion detection... Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    processed_face, face_coords = preprocess_image(frame)

    if processed_face is not None:
        predictions = model.predict(processed_face, verbose=0)
        predicted_emotion = emotion_labels[np.argmax(predictions)]
        confidence = np.max(predictions)

        # Display detected emotion
        x, y, w, h = face_coords
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label_text = f"{predicted_emotion} ({confidence*100:.1f}%)"
        cv2.putText(frame, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Store stable emotion (use majority voting or confidence)
        emotion_history.append(predicted_emotion)
        if len(emotion_history) > 20:
            emotion_history.pop(0)
        # Majority vote every N frames
        emotion_label = max(set(emotion_history), key=emotion_history.count)

    # Display the frame
    cv2.imshow("Emotion Detection - Cultura", frame)

    # Save final result for next module (e.g., Streamlit reads it)
    with open("emotion_result.txt", "w") as f:
        f.write(emotion_label)

    # Optional: save timestamped emotion logs
    with open("emotion_log.txt", "a") as f:
        f.write(f"{datetime.now()}: {emotion_label}\n")

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print(f"[INFO] Final Detected Emotion: {emotion_label}")
