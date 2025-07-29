import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import requests
from PIL import Image
import os
from dotenv import load_dotenv

# === Load environment variables ===
load_dotenv()
QLOO_API_KEY = os.getenv("QLOO_API_KEY") or "AOlP87vLgupHXo0z_0w-LvjyMyToinHmC_xnE01hPqU"

# === Load Emotion Recognition Model ===
try:
    model = load_model("D:/python/CULTURA/emotion_model_v10.h5")
    print("✅ Emotion model loaded.")
except Exception as e:
    st.error(f"❌ Error loading emotion model: {e}")
    model = None

# === Emotion Labels ===
emotion_labels = ["Angry", "Happy", "Neutral", "Sad", "Surprised"]

# === Face Detection ===
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# === Preprocessing Functions ===
def apply_clahe(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img)

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 5, minSize=(30, 30))
    if len(faces) == 0:
        return None, None
    x, y, w, h = faces[0]
    face = gray[y:y+h, x:x+w]
    face = apply_clahe(cv2.resize(face, (64, 64)))
    face = (face - np.mean(face)) / np.std(face)
    face = np.expand_dims(face, axis=-1).reshape(1, 64, 64, 1)
    return face, (x, y, w, h)

# === Friendly Emotion Mapping ===
friendly_emotion_map = {
    "Angry": "frustrated or overwhelmed",
    "Happy": "joyful and energetic",
    "Neutral": "calm and centered",
    "Sad": "a little down or reflective",
    "Surprised": "excited or caught off guard"
}

# === LLaMA API Call ===
def query_llama(emotion, music_list, film_list):
    reworded_emotion = friendly_emotion_map.get(emotion, emotion.lower())
    prompt = (
        f"The user is feeling {reworded_emotion}. "
        f"Give warm, uplifting advice to support their emotional state. "
        f"Then recommend 2-3 songs and movies they might enjoy. "
        f"Songs: {', '.join(music_list)}. Films: {', '.join(film_list)}. "
        f"Write as a friendly virtual companion."
    )
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "llama3", "prompt": prompt, "stream": False},
            timeout=30
        )
        result = response.json().get("response", None)
        if not result or "I can't help" in result:
            return "⚠️ LLaMA returned an incomplete or blocked response. Try rephrasing."
        return result
    except Exception as e:
        return f"❌ Error communicating with LLaMA: {e}"

# === Qloo API Integration ===
def get_qloo_recommendations(emotion):
    tag_map = {
        "Happy": "urn:tag:music:happy",
        "Sad": "urn:tag:music:sad",
        "Angry": "urn:tag:music:aggressive",
        "Neutral": "urn:tag:music:chill",
        "Surprised": "urn:tag:music:experimental"
    }
    tag = tag_map.get(emotion, "urn:tag:music:experimental")

    headers = {
        "accept": "application/json",
        "x-api-key": QLOO_API_KEY
    }

    # Get music recommendations
    music_params = {
        "filter.type": "urn:entity:artist",
        "signal.interests.tags": tag,
        "take": 5
    }
    music_response = requests.get("https://hackathon.api.qloo.com/v2/insights", headers=headers, params=music_params)
    music_data = music_response.json()
    music_recommendations = [item['name'] for item in music_data.get("data", [])]

    # Get movie recommendations
    movie_params = {
        "filter.type": "urn:entity:movie",
        "signal.interests.tags": tag,
        "take": 5
    }
    movie_response = requests.get("https://hackathon.api.qloo.com/v2/insights", headers=headers, params=movie_params)
    movie_data = movie_response.json()
    movie_recommendations = [item['name'] for item in movie_data.get("data", [])]

    return music_recommendations, movie_recommendations

# === Streamlit Interface ===
st.set_page_config(page_title="CULTURA - Emotion-Based Recommendations")
st.title("🎭 Cultura: Real-Time Emotion-Based Recommendations")
st.markdown("Turn on your webcam. Based on your emotion, I’ll recommend music, films, and a message from LLaMA AI.")

img_file_buffer = st.camera_input("📷 Take a selfie to analyze your emotion:")

if img_file_buffer and model is not None:
    image = Image.open(img_file_buffer)
    frame = np.array(image)

    with st.spinner("🧠 Analyzing your emotion..."):
        processed_face, face_coords = preprocess_image(frame)

        if processed_face is not None:
            prediction = model.predict(processed_face, verbose=0)
            predicted_label = emotion_labels[np.argmax(prediction)]
            st.success(f"🧠 Detected Emotion: `{predicted_label}`")

            music, films = get_qloo_recommendations(predicted_label)
            llama_message = query_llama(predicted_label, music, films)

            st.markdown("### 🎧 Recommended Music")
            st.write(" • " + "\n • ".join(music))

            st.markdown("### 🎬 Recommended Films")
            st.write(" • " + "\n • ".join(films))

            st.markdown("### 🤖 Message from LLaMA")
            st.info(llama_message)
        else:
            st.warning("😕 No face detected. Try again with better lighting or clearer view.")

elif model is None:
    st.error("🚫 Emotion model failed to load. Please check path or format.")
