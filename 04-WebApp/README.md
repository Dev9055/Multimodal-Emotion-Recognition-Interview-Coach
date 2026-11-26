# Multimodal Emotion Recognition & AI Interview Coach (Tkinter)

Desktop GUI application with 4 live tabs:

- **Video**: live facial emotion from your webcam (`Models/video.h5`)
- **Audio**: live speech emotion from your microphone or a `.wav` file (`Models/audio.hdf5`)
- **Text**: live Big-Five personality traits as you type or from a `.txt` file (`Models/Personality_traits_NN.h5` & `Models/padding.pickle`)
- **Interview Coach**: simultaneous webcam + mic fusion, scrolling emotion timeline, Readiness Score (0–100), coaching tips, attempt history, and 2-page PDF + CSV report export

## Run
```bash
pip install -r requirements_tkinter.txt
python3 main.py   # or: python3 app.py
```

On macOS, allow **Camera** and **Microphone** access for your Terminal / IDE in System Settings → Privacy & Security.
