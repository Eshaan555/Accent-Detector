# English Accent Detector 🎙️

A Streamlit web application that analyzes English accents from video URLs using OpenAI Whisper for speech recognition and machine learning for accent classification.

## Features

- **Video URL Support**: Works with YouTube URLs and direct MP4 links
- **Speech Recognition**: Uses OpenAI Whisper for accurate transcription
- **Accent Classification**: Detects American, British, and Australian accents
- **Confidence Scoring**: Provides percentage confidence in classification
- **Real-time Processing**: Complete analysis pipeline in a web interface

## Quick Start

### Prerequisites

- Python 3.8 or higher
- FFmpeg (for audio extraction)
- Git (optional, for cloning)

### Installation

1. **Clone or download the repository**
   ```bash
   git clone <your-repo-url>
   cd accent-detector
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install FFmpeg**
   
   **Windows:**
   - Download from https://ffmpeg.org/download.html
   - Add to your system PATH
   
   **macOS:**
   ```bash
   brew install ffmpeg
   ```
   
   **Ubuntu/Debian:**
   ```bash
   sudo apt update
   sudo apt install ffmpeg
   ```

4. **Run the application**
   ```bash
   streamlit run accent_detector_app.py
   ```

5. **Open your browser** to `http://localhost:8501`

## Usage

1. **Enter a video URL** in the text input field
   - YouTube: `https://www.youtube.com/watch?v=...`
   - Direct MP4: `https://example.com/video.mp4`
   - Any public video URL with audio

2. **Click "Analyze"** to start the processing

3. **Wait for analysis** - the app will:
   - Download the video
   - Extract audio
   - Transcribe speech using Whisper
   - Classify the accent

4. **View results**:
   - Detected accent (American/British/Australian)
   - Confidence score (0-100%)
   - Full speech transcription

## System Requirements

### Hardware
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 2GB free space for models and temporary files
- **CPU**: Multi-core processor recommended for faster processing

### Software
- **Python**: 3.8, 3.9, 3.10, or 3.11
- **FFmpeg**: Latest stable version
- **Internet**: Required for downloading videos and models

## Supported Formats

### Video URLs
- YouTube videos
- Direct MP4 links
- Most video hosting platforms
- Public URLs only (no authentication required)

### Audio Requirements
- Clear speech audio
- English language
- Single speaker preferred
- Minimal background noise

## Technical Details

### Components
- **Streamlit**: Web interface framework
- **yt-dlp**: Video downloading from URLs
- **Whisper**: OpenAI's speech recognition model
- **librosa**: Audio processing and feature extraction
- **FFmpeg**: Audio format conversion

### Processing Pipeline
1. **Download**: Video downloaded using yt-dlp
2. **Audio Extraction**: FFmpeg converts to 16kHz WAV
3. **Speech Recognition**: Whisper transcribes audio
4. **Feature Extraction**: MFCC features extracted using librosa
5. **Classification**: ML model predicts accent type
6. **Results**: Accent, confidence, and transcript displayed

## Troubleshooting

### Common Issues

**"Error downloading video"**
- Check if URL is publicly accessible
- Try a different video URL
- Verify internet connection

**"FFmpeg not found"**
- Install FFmpeg and add to system PATH
- Restart terminal/command prompt after installation

**"CUDA out of memory"**
- Use CPU-only mode: `torch.cuda.is_available() = False`
- Reduce video length or quality

**"Module not found"**
- Install missing dependencies: `pip install -r requirements.txt`
- Use virtual environment to avoid conflicts

**"Permission denied"**
- Run with administrator privileges
- Check file permissions in temp directory

### Performance Tips

1. **Use shorter videos** (under 5 minutes) for faster processing
2. **Ensure good audio quality** for better transcription
3. **Close other applications** to free up memory
4. **Use SSD storage** for faster file operations

## Development

### Project Structure
```
accent-detector/
├── accent_detector_app.py    # Main application
├── requirements.txt          # Python dependencies
├── README.md                 # This file
└── models/                   # Downloaded models (auto-created)
```

### Extending the Classifier

The current classifier is a placeholder. To implement a real classifier:

1. **Train a model** on accent datasets
2. **Save the model** using joblib or pickle
3. **Replace the dummy classifier** in `classify_accent()` function
4. **Update feature extraction** as needed

Example replacement:
```python
# Replace dummy classifier with real model
def classify_accent(audio_path):
    # Load your trained model
    model = joblib.load('path/to/your/model.pkl')
    
    # Extract features
    y, sr = librosa.load(audio_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    features = np.mean(mfcc.T, axis=0).reshape(1, -1)
    
    # Predict
    prediction = model.predict(features)[0]
    confidence = model.predict_proba(features)[0].max() * 100
    
    return prediction, confidence
```

### Adding New Accents

To support additional accents:

1. **Update the accents list** in `classify_accent()`
2. **Retrain your model** with new accent data
3. **Test thoroughly** with samples from each accent

## Deployment

### Local Deployment
The app runs locally by default on `http://localhost:8501`

### Cloud Deployment

**Streamlit Community Cloud:**
1. Push code to GitHub
2. Connect at https://share.streamlit.io
3. Deploy automatically

**Heroku:**
1. Add `Procfile`: `web: streamlit run accent_detector_app.py --server.port=$PORT --server.address=0.0.0.0`
2. Add `system.txt`: `ffmpeg`
3. Deploy via Heroku CLI

**Docker:**
```dockerfile
FROM python:3.9-slim

RUN apt-get update && apt-get install -y ffmpeg

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY accent_detector_app.py .

EXPOSE 8501

CMD ["streamlit", "run", "accent_detector_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

## License

This project is intended for educational and research purposes.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## Support

For issues and questions:
- Check the troubleshooting section
- Review error messages carefully
- Ensure all dependencies are installed correctly

## Acknowledgments

- OpenAI for the Whisper speech recognition model
- The librosa team for audio processing tools
- Streamlit for the web framework
- yt-dlp for video downloading capabilities

---

**Note**: This is a demonstration application. For production use, consider implementing proper error handling, user authentication, rate limiting, and a robust accent classification model trained on comprehensive datasets.

---

## requirements.txt

```txt
streamlit>=1.28.0
yt-dlp>=2023.9.0
torch>=2.0.0
openai-whisper>=20231117
librosa>=0.10.0
numpy>=1.24.0
joblib>=1.3.0
```

---

## Additional Files for Deployment

### Procfile (for Heroku)
```
web: streamlit run accent_detector_app.py --server.port=$PORT --server.address=0.0.0.0
```

### runtime.txt (for Heroku)
```
python-3.9.18
```

### Aptfile (for Heroku - system dependencies)
```
ffmpeg
```

### .streamlit/config.toml
```toml
[server]
headless = true
port = 8501
enableCORS = false

[browser]
gatherUsageStats = false
```

### setup.sh (for Streamlit Cloud)
```bash
#!/bin/bash

# Install system dependencies
apt-get update
apt-get install -y ffmpeg

# Create .streamlit directory
mkdir -p ~/.streamlit/

# Setup Streamlit config
echo "\
[server]\n\
headless = true\n\
port = \$PORT\n\
enableCORS = false\n\
\n\
[browser]\n\
gatherUsageStats = false\n\
" > ~/.streamlit/config.toml
```
