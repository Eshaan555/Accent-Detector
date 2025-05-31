# English Accent Detector 🎙️

A Streamlit web application that analyzes English accents from video URLs using Google Speech Recognition and machine learning for accent classification.

## Features

- **Video URL Support**: Works with YouTube URLs and direct MP4 links
- **Speech Recognition**: Uses Google Speech Recognition for accurate transcription
- **Accent Classification**: Detects American, British, and Australian accents
- **Confidence Scoring**: Provides percentage confidence in classification
- **Real-time Processing**: Complete analysis pipeline in a web interface
- **Cloud-Optimized**: Designed for reliable deployment on Streamlit Cloud

## Quick Start

### Prerequisites

- Python 3.8 or higher
- FFmpeg (for audio extraction)
- Internet connection (for speech recognition)
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
   - Transcribe speech using Google Speech Recognition
   - Classify the accent

4. **View results**:
   - Detected accent (American/British/Australian)
   - Confidence score (0-100%)
   - Full speech transcription

## System Requirements

### Hardware
- **RAM**: 2GB minimum, 4GB recommended
- **Storage**: 1GB free space for temporary files
- **CPU**: Any modern processor
- **Internet**: Required for video downloads and speech recognition

### Software
- **Python**: 3.8, 3.9, 3.10, 3.11, or 3.13
- **FFmpeg**: Latest stable version
- **Internet**: Required for downloading videos and Google Speech Recognition

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
- Good internet connection for transcription

## Technical Details

### Components
- **Streamlit**: Web interface framework
- **yt-dlp**: Video downloading from URLs
- **Google Speech Recognition**: Free speech-to-text service
- **librosa**: Audio processing and feature extraction
- **FFmpeg**: Audio format conversion
- **SpeechRecognition**: Python library for speech recognition
- **pydub**: Audio manipulation

### Processing Pipeline
1. **Download**: Video downloaded using yt-dlp
2. **Audio Extraction**: FFmpeg converts to 16kHz WAV
3. **Speech Recognition**: Google API transcribes audio
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

**"Speech recognition error"**
- Check internet connection
- Ensure audio has clear speech
- Try with a different video with better audio quality

**"Could not understand audio"**
- Audio quality may be too poor
- Try videos with clearer speech
- Ensure minimal background noise

**"Module not found"**
- Install missing dependencies: `pip install -r requirements.txt`
- Use virtual environment to avoid conflicts

**"Permission denied"**
- Run with administrator privileges
- Check file permissions in temp directory

### Performance Tips

1. **Use shorter videos** (under 3 minutes) for faster processing
2. **Ensure good audio quality** for better transcription
3. **Single speaker videos** work best
4. **Stable internet connection** for reliable speech recognition
5. **Close other applications** to free up memory

## Development

### Project Structure
```
accent-detector/
├── accent_detector_app.py    # Main application
├── requirements.txt          # Python dependencies
├── packages.txt              # System dependencies
├── config.toml              # Streamlit configuration
├── python-version.txt       # Python version specification
└── README.md                # This file
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

### Streamlit Community Cloud

**Automatic Deployment:**
1. Push code to GitHub
2. Connect at https://share.streamlit.io
3. Deploy automatically

**Required Files:**
- `requirements.txt` - Python dependencies
- `packages.txt` - System dependencies (ffmpeg)
- `config.toml` - Streamlit configuration
- `python-version.txt` - Python version specification

### Other Platforms

**Heroku:**
```
# Add to requirements.txt:
streamlit>=1.28.0,<1.46.0
yt-dlp>=2023.9.0
librosa>=0.10.0
numpy>=1.24.0,<2.0.0
joblib>=1.3.0
ffmpeg-python>=0.2.0
soundfile>=0.12.0
scipy>=1.10.0
SpeechRecognition>=3.10.0
pydub>=0.25.0

# Procfile:
web: streamlit run accent_detector_app.py --server.port=$PORT --server.address=0.0.0.0

# Aptfile:
ffmpeg
```

**Docker:**
```dockerfile
FROM python:3.10-slim

RUN apt-get update && apt-get install -y ffmpeg

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY accent_detector_app.py .

EXPOSE 8501

CMD ["streamlit", "run", "accent_detector_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

## Current Dependencies

```txt
streamlit>=1.28.0,<1.46.0
yt-dlp>=2023.9.0
librosa>=0.10.0
numpy>=1.24.0,<2.0.0
joblib>=1.3.0
ffmpeg-python>=0.2.0
soundfile>=0.12.0
scipy>=1.10.0
SpeechRecognition>=3.10.0
pydub>=0.25.0
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
- Verify internet connection for speech recognition

## Acknowledgments

- Google for the Speech Recognition API
- The librosa team for audio processing tools
- Streamlit for the web framework
- yt-dlp for video downloading capabilities
- The SpeechRecognition library maintainers

---

**Note**: This is a demonstration application using Google's free speech recognition service. For production use, consider implementing proper error handling, user authentication, rate limiting, and a robust accent classification model trained on comprehensive datasets.

## Architecture Differences

This implementation differs from typical Whisper-based solutions:

- **Lighter Dependencies**: No PyTorch or heavy ML frameworks required
- **Cloud Optimized**: Designed specifically for Streamlit Cloud compatibility
- **Internet Dependent**: Requires internet connection for speech recognition
- **Free Service**: Uses Google's free speech recognition API
- **Faster Startup**: Quicker initial loading without large model downloads

---

## Live Demo

Visit the deployed application: https://accent-detector-kpahfmxippbaleojwqjv8l.streamlit.app/

---

**Last Updated**: May 2025
