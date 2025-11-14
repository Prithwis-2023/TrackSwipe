# MoodFlix 🎬

![WhatsApp Image 2025-11-09 at 11 10 20 PM](https://github.com/user-attachments/assets/1a31f8e1-1fae-4544-a279-5d9a61e3f983)

## Overview

MoodFlix is an AI-powered movie recommendation system that analyzes your emotional state, voice tone, location, weather, and contextual information to suggest personalized movie recommendations. The system uses computer vision, audio analysis, and machine learning to understand user preferences and improve recommendations over time.

## Features

### Multi-Modal Emotion Detection
- **Facial Emotion Recognition**: Captures 20 frames and analyzes emotions using DeepFace with confidence-weighted smoothing
- **Voice Tone Analysis**: Records 10 seconds of audio and extracts features (RMS energy, pitch, zero-crossing rate, spectral centroid) using librosa
- **Supported Emotions**: angry, disgust, fear, happy, sad, surprise, neutral

### Context-Aware Recommendations
- **Location-Based**: Uses geocoding to determine city and coordinates
- **Weather Integration**: Fetches current weather conditions and temperature via OpenWeatherMap API
- **Temporal Context**: Considers day of week, weekday/weekend status, and upcoming holidays
- **Voice Tone**: Classifies voice as happy, sad, or neutral

### Intelligent Recommendation Engine
- **Primary Recommendations**: Uses Ollama (llama2 model) to generate context-aware movie suggestions
- **Machine Learning**: Random Forest classifier trained on user selection history from [user_logs.csv](user_logs.csv)
- **Hybrid Approach**: Combines LLM recommendations with learned user preferences

### Additional Features
- **Eye Tracking** ([eyetracker.py](eyetracker.py)): MediaPipe-based gaze detection for engagement monitoring
- **User Feedback Loop**: Logs selections to continuously improve recommendations
- **Interactive UI**: Jupyter widget-based movie selection interface

## Project Structure

```
TrackSwipe/
├── TrackSwipe.ipynb          # Main notebook with recommendation system
├── aggregate.py              # Aggregation script for emotion + weather data
├── eyetracker.py            # Eye tracking implementation
├── other_params.py          # Location, weather, and temporal utilities
├── user_logs.csv            # User selection history for ML training
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

2. Install Ollama:
```bash
# Follow instructions at https://ollama.ai
ollama pull llama2
```

3. Set up webcam and microphone access

## Usage

### Running the Recommendation System

Open and run [TrackSwipe.ipynb](TrackSwipe.ipynb) in Jupyter:

```bash
jupyter notebook TrackSwipe.ipynb
```

The system will:
1. Detect your location and weather conditions
2. Capture facial emotions (20 frames)
3. Record and analyze voice tone (10 seconds)
4. Generate 5 personalized movie recommendations
5. Display interactive buttons for selection
6. Log your choice to improve future recommendations

### Running Eye Tracker

```bash
python eyetracker.py
```

Monitors gaze direction (Left, Right, Up, Down, Center) and alerts when user loses focus.

### Running Aggregation Script

```bash
python aggregate.py
```

Combines emotion detection with environmental context for real-time analysis.

## Technologies

- **Computer Vision**: OpenCV, MediaPipe, DeepFace, MTCNN
- **Audio Processing**: sounddevice, librosa
- **Machine Learning**: scikit-learn (Random Forest)
- **LLM**: Ollama (llama2)
- **APIs**: OpenWeatherMap, geocoder
- **UI**: ipywidgets, Jupyter

## Data Logged

Each interaction is logged to [user_logs.csv](user_logs.csv) with:
- Timestamp
- Location (city, latitude, longitude)
- Temporal context (weekday, today/tomorrow status)
- Weather (description, temperature)
- Emotional state (mood, voice tone)
- Selected movie

## Configuration

Key parameters in [TrackSwipe.ipynb](TrackSwipe.ipynb):
- `CONFIDENCE_THRESHOLD = 0.4`: Minimum emotion confidence
- `NUM_FRAMES = 20`: Frames to capture for emotion
- `DURATION = 10`: Audio recording duration (seconds)
- `OLLAMA_MODEL = "llama2"`: LLM model for recommendations

## Future Enhancements

- Real-time eye tracking integration for engagement scoring
- Voice emotion model fine-tuning
- Deep learning-based preference prediction
- Multi-user support
- Streaming platform integration

## Credits

Developed as part of Sejong University coursework (S3).