import streamlit as st
import random
import requests

from emo_model import emotion_model_v6 # <-- Use your emotion model

# --- LLaMA Local Ollama Integration ---
def query_llama(prompt):
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "llama3", "prompt": prompt, "stream": False}
        )
        return response.json()["response"]
    except Exception as e:
        return f"Error from LLaMA: {e}"

# --- Mock Qloo API Replacements ---
MOCK_QLOO_MUSIC = {
    "happy": ["Coldplay - Viva La Vida", "Pharrell - Happy", "Daft Punk - Get Lucky"],
    "sad": ["Adele - Someone Like You", "Radiohead - Creep", "Lana Del Rey - Summertime Sadness"],
    "stressed": ["Lo-fi Chill Beats", "Sigur Rós - Sæglópur", "Bon Iver - Holocene"]
}

MOCK_QLOO_FILMS = {
    "happy": ["The Secret Life of Walter Mitty", "La La Land", "Paddington 2"],
    "sad": ["The Pursuit of Happyness", "Blue Valentine", "Her"],
    "stressed": ["The Intouchables", "Chef", "Inside Out"]
}

# --- Streamlit App UI ---
st.set_page_config(page_title="CULTURA - Mood Based Recommendations")
st.title("🎭 Cultura: Mood-to-Recommendation Assistant")
st.write("Tell me how you feel, and I'll find music and movies that match your vibe.")

mood_input = st.text_input("📝 Describe your mood in one or two sentences:")

if st.button("🎯 Analyze & Recommend"):
    with st.spinner("Analyzing your emotion..."):
        predicted_emotion = emotion_model_v6(mood_input)
        st.success(f"🧠 Detected Emotion: `{predicted_emotion}`")

        # Call LLaMA with your input
        prompt = f"Suggest personalized thoughts and activities for someone feeling {predicted_emotion}."
        llama_response = query_llama(prompt)

        music = MOCK_QLOO_MUSIC.get(predicted_emotion, [])
        films = MOCK_QLOO_FILMS.get(predicted_emotion, [])

    st.markdown("### 🤖 LLaMA Response")
    st.info(llama_response)

    st.markdown("### 🎧 Music Recommendations")
    for track in music:
        st.write(f"• {track}")

    st.markdown("### 🎬 Film Recommendations")
    for film in films:
        st.write(f"• {film}")

    st.markdown("### 🔁 Was this helpful?")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("👍 Yes"):
            st.success("Thanks for the feedback!")
    with col2:
        if st.button("👎 No"):
            st.warning("We'll try to improve that!")
