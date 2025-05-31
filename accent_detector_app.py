# accent_detector_app.py
# Version without Whisper using basic speech recognition

import os
import sys
import warnings

# Suppress warnings and configure environment
warnings.filterwarnings("ignore")

import streamlit as st
import tempfile
import subprocess
import yt_dlp
import librosa
import numpy as np
import shutil
from pathlib import Path
import speech_recognition as sr
from pydub import AudioSegment

# Configure Streamlit page
st.set_page_config(
    page_title="English Accent Detector",
    page_icon="🎙️",
    layout="centered"
)

def check_dependencies():
    """Check if required system dependencies are available"""
    dependencies = ['ffmpeg']
    missing = []
    
    for dep in dependencies:
        if shutil.which(dep) is None:
            missing.append(dep)
    
    return missing

def classify_accent(audio_path):
    """Classify accent from audio file"""
    try:
        # Load audio and extract MFCCs
        y, sr = librosa.load(audio_path, sr=16000, duration=30)  # Limit to 30 seconds
        if len(y) == 0:
            return "Unknown", 0.0
            
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        features = np.mean(mfcc.T, axis=0).reshape(1, -1)

        # Dummy classifier (replace with real model)
        accents = ['American', 'British', 'Australian']
        probs = np.random.dirichlet(np.ones(len(accents)), size=1)[0]
        pred = accents[np.argmax(probs)]
        confidence = float(np.max(probs) * 100)
        
        return pred, round(confidence, 2)
    except Exception as e:
        st.error(f"Error in accent classification: {e}")
        return "Unknown", 0.0

def download_video(url, max_duration=300):
    """Download video using yt_dlp with size/duration limits"""
    temp_dir = tempfile.mkdtemp()
    video_path = os.path.join(temp_dir, "video.%(ext)s")
    
    ydl_opts = {
        'format': 'best[ext=mp4]/best[height<=720]',
        'outtmpl': video_path,
        'quiet': True,
        'no_warnings': True,
        'match_filter': lambda info_dict: None if info_dict.get('duration', 0) <= max_duration else "Video too long",
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        
        # Find the actual downloaded file
        downloaded_files = list(Path(temp_dir).glob("video.*"))
        if downloaded_files:
            return str(downloaded_files[0])
        else:
            raise Exception("No video file found after download")
            
    except Exception as e:
        # Cleanup on failure
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise e

def extract_audio(video_path, duration_limit=60):
    """Extract audio using ffmpeg with duration limit"""
    audio_path = str(Path(video_path).with_suffix('.wav'))
    
    command = [
        "ffmpeg", "-i", video_path,
        "-ar", "16000", "-ac", "1",
        "-t", str(duration_limit),  # Limit to 60 seconds
        "-y", audio_path
    ]
    
    try:
        result = subprocess.run(
            command, 
            stdout=subprocess.DEVNULL, 
            stderr=subprocess.PIPE,
            text=True,
            timeout=120  # 2-minute timeout
        )
        
        if result.returncode != 0:
            raise Exception(f"FFmpeg error: {result.stderr}")
            
        return audio_path
    except subprocess.TimeoutExpired:
        raise Exception("Audio extraction timed out")
    except Exception as e:
        raise Exception(f"Failed to extract audio: {e}")

def transcribe_audio_basic(audio_path):
    """Transcribe audio using basic speech recognition"""
    try:
        r = sr.Recognizer()
        
        # Convert to proper format for speech recognition
        audio = AudioSegment.from_wav(audio_path)
        audio = audio.set_frame_rate(16000).set_channels(1)
        
        # Save temporary file
        temp_path = audio_path.replace('.wav', '_temp.wav')
        audio.export(temp_path, format="wav")
        
        with sr.AudioFile(temp_path) as source:
            audio_data = r.record(source, duration=30)  # Limit to 30 seconds
        
        # Try Google's free speech recognition
        try:
            transcript = r.recognize_google(audio_data)
            return transcript
        except sr.UnknownValueError:
            return "Could not understand audio"
        except sr.RequestError as e:
            return f"Speech recognition error: {e}"
        finally:
            # Cleanup temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
    except Exception as e:
        st.error(f"Transcription failed: {e}")
        return "Transcription failed"

def cleanup_files(*file_paths):
    """Safely cleanup temporary files and directories"""
    for file_path in file_paths:
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path, ignore_errors=True)
        except Exception:
            pass  # Ignore cleanup errors

def main():
    """Main Streamlit application"""
    
    # Check system dependencies
    missing_deps = check_dependencies()
    if missing_deps:
        st.error(f"Missing system dependencies: {', '.join(missing_deps)}")
        st.info("Please ensure FFmpeg is installed on the system.")
        return
    
    # UI
    st.title("🎙️ English Accent Detector")
    st.markdown("Analyze English accents from video URLs using speech recognition.")
    
    # Input section
    st.subheader("Video Input")
    video_url = st.text_input(
        "Enter a video URL (YouTube, MP4, etc.)",
        placeholder="https://www.youtube.com/watch?v=..."
    )
    
    # Information about limitations
    with st.expander("ℹ️ Usage Information"):
        st.markdown("""
        **Supported formats:** YouTube, direct MP4 links, most video platforms
        
        **Limitations:**
        - Videos are limited to 5 minutes for processing
        - Audio analysis uses first 60 seconds only
        - Requires clear English speech
        - Best results with single speaker
        - Uses Google Speech Recognition (requires internet)
        
        **Note:** This is a demonstration app with a placeholder accent classifier.
        """)
    
    # Analysis button and processing
    if st.button("🔍 Analyze Accent", type="primary") and video_url:
        if not video_url.strip():
            st.warning("Please enter a valid video URL.")
            return
            
        # Initialize tracking variables
        video_path = None
        audio_path = None
        temp_dir = None
        
        try:
            # Download video
            with st.spinner("📥 Downloading video..."):
                video_path = download_video(video_url)
                temp_dir = os.path.dirname(video_path)
                st.success("Video downloaded successfully!")
            
            # Extract audio
            with st.spinner("🎵 Extracting audio..."):
                audio_path = extract_audio(video_path)
                st.success("Audio extracted successfully!")
            
            # Transcribe
            with st.spinner("📝 Transcribing speech..."):
                transcript = transcribe_audio_basic(audio_path)
                st.success("Speech transcribed successfully!")
            
            # Classify accent
            with st.spinner("🔍 Analyzing accent..."):
                accent, confidence = classify_accent(audio_path)
                st.success("Analysis complete!")
            
            # Display results
            st.subheader("📊 Analysis Results")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Detected Accent", accent)
            with col2:
                st.metric("Confidence Score", f"{confidence}%")
            
            st.subheader("📝 Transcript")
            if transcript and transcript.strip() and "failed" not in transcript.lower():
                st.info(transcript)
            else:
                st.warning("No speech detected or transcription failed.")
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.info("Please try with a different video URL or check your internet connection.")
            
        finally:
            # Cleanup temporary files
            cleanup_files(video_path, audio_path, temp_dir)

if __name__ == "__main__":
    main()
