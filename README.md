# 🎙️ English Accent Detector

This is a simple Streamlit web app that detects the **English accent** of a speaker from a public video URL. It extracts audio from the video, transcribes the speech using OpenAI Whisper, and classifies the accent (e.g., American, British, Australian) with a confidence score.

---

## 🚀 Features

- Accepts **public video URLs** (e.g., Loom, YouTube, MP4).
- Extracts and processes audio using `ffmpeg`.
- Transcribes speech using **Whisper** (OpenAI).
- Classifies English accents with a confidence score.
- Displays transcript for reference.

---

## 🛠️ Tech Stack

- [Streamlit](https://streamlit.io/)
- [OpenAI Whisper](https://github.com/openai/whisper)
- [FFmpeg](https://ffmpeg.org/)
- [Librosa](https://librosa.org/)
- `yt_dlp` for video download

---

## 🖥️ Usage

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/accent-detector.git
cd accent-detector
```

### 2. Install Dependencies

We recommend using a virtual environment (e.g., `conda` or `venv`).

```bash
pip install -r requirements.txt
```

Or manually install:

```bash
pip install streamlit yt_dlp ffmpeg-python openai-whisper librosa torch torchaudio
```

### 3. Run the App

```bash
streamlit run accent_detector_app.py
```

---

## 🌐 Deployment

To deploy this app on [Streamlit Cloud](https://streamlit.io/cloud):

1. Push this repo to GitHub.
2. Go to Streamlit Cloud and sign in.
3. Deploy your repo with `accent_detector_app.py` as the entry point.

---

## 📌 Notes

- This app currently uses a **dummy classifier** for accent detection based on random MFCC features. For production use, replace it with a trained ML model.
- Only English accents (American, British, Australian) are supported.
- Video must contain **clearly spoken English** for accurate results.

---

## 📄 Example Output

```
Accent: American  
Confidence: 84.5%  
Transcript: "Hello, my name is Sarah and I’m here to explain the hiring process..."
```

---

## 🧠 Future Improvements

- Integrate a real pretrained accent classification model.
- Add more supported accents (e.g., Indian, South African).
- Improve transcript quality and grammar check.
- Enable file upload and YouTube playlists.

---
