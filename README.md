# Multimodal Emotion Recognition & AI Interview Coach (Tkinter Desktop App)

A lightweight Python desktop application built with **Tkinter** and **TensorFlow/Keras** for real-time multimodal emotion recognition and mock interview coaching.

## Features (4 Tabs)
- **Video:** Real-time facial emotion detection from your webcam (7 emotions via Xception CNN).
- **Audio:** Real-time speech emotion recognition from your microphone using a rolling 3-second buffer + live mic level meter, plus `.wav` file analysis (7 emotions via Time-Distributed 2D CNN + LSTM).
- **Text:** Live Big-Five personality trait analysis updating as you type, plus `.txt` file loading (1D CNN + 3-layer LSTM).
- **Interview Coach:** Combines live webcam + live microphone + scrolling Composure vs. Stress timeline + Face–Voice sync check + Readiness Score (0–100) + coaching tips + Big-Five transcript analysis + one-click **2-page PDF & CSV report export**.

## Quick Start
```bash
pip install -r requirements.txt
python3 04-WebApp/main.py
```

*(Optional: install `SpeechRecognition` for automatic speech-to-text inside the Interview Coach tab.)*

## Project Structure
```text
.
├── 00-Presentation/             # Project slides, charts & visual assets
├── 01-Audio/                    # RAVDESS audio preprocessing & CNN-LSTM notebooks (.ipynb)
├── 02-Text/                     # NLP Big-Five personality training notebooks (.ipynb)
├── 03-Video/                    # FER2013 facial expression Xception notebooks (.ipynb)
├── Models/                      # Shared vision model weights
├── 04-WebApp/
│   ├── main.py                  # Desktop app entry launcher
│   ├── app.py                   # 4-tab Tkinter GUI & live Video/Audio/Text pipelines
│   ├── interview_coach.py       # Live Interview Coach fusion, timeline & 2-page PDF export
│   ├── requirements_tkinter.txt # Desktop dependencies
│   ├── Models/                  # Pre-trained weights (video.h5, audio.hdf5, Personality_traits_NN.h5, padding.pickle, etc.)
│   ├── tmp/                     # Attempt history JSON & temporary audio recordings
│   └── reports/                 # Exported 2-page PDF reports & CSV timelines
├── requirements.txt
├── LICENSE
└── Multimodal_Emotion_Recognition_Project_Report.pdf
```

## Credits & License
The pre-trained model weights in `04-WebApp/Models/` are derived from open-source models trained on the FER2013, RAVDESS, and Pennebaker Essays datasets under the Apache License 2.0 (see `LICENSE`).
