import streamlit as st
import speech_recognition as sr
import joblib
from sklearn.preprocessing import StandardScaler

# Fungsi untuk mengklasifikasikan sentimen emosi
def classify_sentiment(text, model, scaler):
    # Preprocess audio text (misalnya, ekstraksi fitur audio jika diperlukan)
    # ...

    # Lalu, skalakan fitur audio jika diperlukan
    scaled_features = scaler.transform(preprocessed_features)

    # Lakukan prediksi sentimen emosi
    prediction = model.predict(scaled_features)

    return prediction

# Tampilan aplikasi Streamlit
st.title("Aplikasi Klasifikasi Sentimen Emosi (SER) dari Audio")

# Input audio dari pengguna
uploaded_file = st.file_uploader("Upload file audio (MP3, WAV, atau lainnya):", type=["mp3", "wav"])

if uploaded_file:
    audio = sr.AudioFile(uploaded_file)
    recognizer = sr.Recognizer()

    try:
        with audio as source:
            audio_text = recognizer.record(source)
        st.audio(uploaded_file)
        st.write("Teks dari audio:")
        st.write(recognizer.recognize_google(audio_text))

        # Load model KNN dan skalar dari file
        model = joblib.load('modelknn.pkl')
        scaler = joblib.load('scaler.pkl')

        sentiment = classify_sentiment(recognizer.recognize_google(audio_text), model, scaler)
        st.write(f"Sentimen Emosi: {sentiment}")

    except sr.UnknownValueError:
        st.error("Tidak dapat mengenali teks dari audio.")
    except sr.RequestError:
        st.error("Permintaan kepada Google Speech Recognition gagal.")
