# accent_detector_app.py

import streamlit as st
import os
import tempfile
import subprocess
import yt_dlp
import torch
import whisper
import librosa
import numpy as np
import joblib

# Load Whisper
model = whisper.load_model("base")  # Or 'small' / 'medium' based on GPU availability

# Load pretrained accent classifier (dummy for now)
def classify_accent(audio_path):
    # Load audio and extract MFCCs
    y, sr = librosa.load(audio_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    features = np.mean(mfcc.T, axis=0).reshape(1, -1)

    # Dummy classifier (replace with real model)
    accents = ['American', 'British', 'Australian']
    probs = np.random.dirichlet(np.ones(len(accents)), size=1)[0]
    pred = accents[np.argmax(probs)]
    return pred, round(float(np.max(probs) * 100), 2)

# Download video using yt_dlp
def download_video(url):
    temp_dir = tempfile.mkdtemp()
    video_path = os.path.join(temp_dir, "video.mp4")
    ydl_opts = {
        'format': 'best[ext=mp4]',
        'outtmpl': video_path,
        'quiet': True
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    return video_path

# Extract audio using ffmpeg
def extract_audio(video_path):
    audio_path = video_path.replace(".mp4", ".wav")
    command = [
        "ffmpeg", "-i", video_path,
        "-ar", "16000", "-ac", "1", audio_path,
        "-y"
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return audio_path

# Transcribe using Whisper
def transcribe(audio_path):
    result = model.transcribe(audio_path)
    return result["text"]

# Streamlit UI
st.title("English Accent Detector")

video_url = st.text_input("Paste a public video URL (MP4 or YouTube)")

if st.button("Analyze") and video_url:
    with st.spinner("Downloading video..."):
        try:
            video_path = download_video(video_url)
        except Exception as e:
            st.error(f"Error downloading video: {e}")
            st.stop()

    with st.spinner("Extracting audio..."):
        audio_path = extract_audio(video_path)

    with st.spinner("Transcribing speech..."):
        transcript = transcribe(audio_path)

    with st.spinner("Classifying accent..."):
        accent, confidence = classify_accent(audio_path)

    st.success("Analysis complete!")
    st.markdown(f"**Accent:** {accent}")
    st.markdown(f"**Confidence Score:** {confidence}%")
    st.markdown("**Transcript:**")
    st.info(transcript)

    # Cleanup
    os.remove(video_path)
    os.remove(audio_path)
