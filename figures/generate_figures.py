#!/usr/bin/env python3

import importlib.util
import os
import subprocess
import sys

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
WEBAPP_DIR = os.path.abspath(os.path.join(OUT_DIR, "..", "04-WebApp"))


def ensure_tensorflow_python():
    if importlib.util.find_spec("tensorflow") is not None:
        return
    if os.environ.get("_MER_PYTHON_CHECKED"):
        return
    current = os.path.realpath(sys.executable)
    for py in (
        "/Library/Frameworks/Python.framework/Versions/3.12/bin/python3",
        "/Library/Frameworks/Python.framework/Versions/3.11/bin/python3",
        "/Library/Frameworks/Python.framework/Versions/3.10/bin/python3",
    ):
        if os.path.isfile(py) and os.path.realpath(py) != current:
            check = subprocess.run(
                [py, "-c", "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('tensorflow') else 1)"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if check.returncode == 0:
                os.environ["_MER_PYTHON_CHECKED"] = "1"
                os.execv(py, [py, os.path.abspath(__file__), *sys.argv[1:]])


if __name__ == "__main__":
    ensure_tensorflow_python()

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.modules.setdefault("pyarrow", None)
sys.path.insert(0, WEBAPP_DIR)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon
from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np


def draw_box(ax, x, y, w, h, title, subtitle="", fc="#e8f1fb", ec="#5b8cc4", title_size=10, sub_size=8, bold_title=True):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        facecolor=fc, edgecolor=ec, linewidth=1.4, zorder=3,
    )
    ax.add_patch(patch)
    if subtitle:
        ax.text(x + w / 2, y + h * 0.65, title, ha="center", va="center",
                fontsize=title_size, fontweight="bold" if bold_title else "normal",
                color="#1a202c", zorder=4)
        ax.text(x + w / 2, y + h * 0.30, subtitle, ha="center", va="center",
                fontsize=sub_size, color="#4a5568", zorder=4)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center",
                fontsize=title_size, fontweight="bold" if bold_title else "normal",
                color="#1a202c", zorder=4)


def draw_arrow(ax, x1, y1, x2, y2, label="", lx=None, ly=None, color="#2d3748", lw=1.3):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>,head_length=0.55,head_width=0.35", color=color, lw=lw),
        zorder=5,
    )
    if label:
        tx = lx if lx is not None else (x1 + x2) / 2
        ty = ly if ly is not None else (y1 + y2) / 2
        ax.text(tx, ty, label, ha="center", va="center", fontsize=7.6, color="#2d3748",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85), zorder=6)


def draw_diamond(ax, cx, cy, w, h, text, fc="#fef3c7", ec="#d69e2e", fontsize=8.5):
    pts = [(cx, cy + h / 2), (cx + w / 2, cy), (cx, cy - h / 2), (cx - w / 2, cy)]
    poly = Polygon(pts, closed=True, facecolor=fc, edgecolor=ec, linewidth=1.4, zorder=3)
    ax.add_patch(poly)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize,
            fontweight="bold", color="#1a202c", zorder=4)


def make_fig3_1_architecture():
    fig, ax = plt.subplots(figsize=(9.2, 5.2), dpi=220)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.add_patch(FancyBboxPatch((0.01, 0.01), 0.98, 0.98, boxstyle="round,pad=0.005,rounding_size=0.015",
                                facecolor="#fafbfd", edgecolor="#d0d7de", linewidth=1.0, zorder=1))

    layers = [
        (0.84, "Presentation\nLayer\n(Tkinter GUI)"),
        (0.58, "Processing &\nLive Fusion\nLayer"),
        (0.32, "Deep Learning\nModels Layer\n(TF / Keras 3)"),
        (0.10, "Local Storage\n& Export Layer"),
    ]
    for y, lbl in layers:
        ax.text(0.075, y, lbl, ha="center", va="center", fontsize=8.2, fontweight="bold", color="#4a5568")

    draw_box(ax, 0.16, 0.77, 0.18, 0.14, "Video Tab", "Live Webcam + Face Box", "#dbeafe", "#3b82f6", 9, 7.5)
    draw_box(ax, 0.36, 0.77, 0.18, 0.14, "Audio Tab", "Live Mic + .wav File", "#dbeafe", "#3b82f6", 9, 7.5)
    draw_box(ax, 0.56, 0.77, 0.18, 0.14, "Text Tab", "Live Typing + .txt File", "#dbeafe", "#3b82f6", 9, 7.5)
    draw_box(ax, 0.76, 0.77, 0.21, 0.14, "Interview Coach Tab", "Live Fusion + Timeline + Report", "#fef3c7", "#d97706", 9, 7.5)

    draw_box(ax, 0.16, 0.50, 0.24, 0.15, "Vision & Audio Preprocessors",
             "OpenCV Haar / dlib HOG + 48×48\nsounddevice Ring Buffer + Log-Mel", "#dcfce7", "#16a34a", 8.8, 7.2)
    draw_box(ax, 0.43, 0.50, 0.24, 0.15, "Text & Speech Pipeline",
             "Regex / NLTK Cleaner + Tokenizer\nOptional SpeechRecognition STT", "#ede9fe", "#7c3aed", 8.8, 7.2)
    draw_box(ax, 0.70, 0.50, 0.27, 0.15, "Fusion, Scoring & PDF Engine",
             "Late Fusion + Readiness Score (0-100)\nMatplotlib 2-Page PDF & CSV Exporter", "#fee2e2", "#dc2626", 8.8, 7.2)

    draw_box(ax, 0.16, 0.25, 0.24, 0.14, "Video Model (video.h5)",
             "Xception CNN (FER2013)\n7 Facial Emotions", "#e0f2fe", "#0284c7", 8.8, 7.3)
    draw_box(ax, 0.43, 0.25, 0.24, 0.14, "Audio Model (audio.hdf5)",
             "Time-Distributed 2D CNN + LSTM\n7 Speech Emotions (RAVDESS)", "#e0f2fe", "#0284c7", 8.8, 7.3)
    draw_box(ax, 0.70, 0.25, 0.27, 0.14, "Text Model (Personality_traits_NN)",
             "1D CNN + 3×LSTM + padding.pickle\nBig-Five Personality Traits", "#e0f2fe", "#0284c7", 8.6, 7.3)

    draw_box(ax, 0.16, 0.04, 0.81, 0.11, "Local Filesystem Assets & Outputs",
             "Models/ (.h5, .hdf5, .pickle)   |   tmp/interview_history.json & .wav   |   reports/ (2-Page PDF & CSV)",
             "#f1f5f9", "#475569", 9.0, 7.5)

    draw_arrow(ax, 0.25, 0.77, 0.25, 0.65)
    draw_arrow(ax, 0.45, 0.77, 0.34, 0.65)
    draw_arrow(ax, 0.65, 0.77, 0.55, 0.65)
    draw_arrow(ax, 0.86, 0.77, 0.84, 0.65)
    draw_arrow(ax, 0.40, 0.57, 0.70, 0.57, "Live Probs", 0.55, 0.68, "#94a3b8", 1.0)
    draw_arrow(ax, 0.28, 0.50, 0.28, 0.39)
    draw_arrow(ax, 0.36, 0.50, 0.50, 0.39)
    draw_arrow(ax, 0.58, 0.50, 0.78, 0.39)
    draw_arrow(ax, 0.28, 0.25, 0.28, 0.15)
    draw_arrow(ax, 0.55, 0.25, 0.55, 0.15)
    draw_arrow(ax, 0.84, 0.25, 0.84, 0.15)
    draw_arrow(ax, 0.94, 0.50, 0.94, 0.15, "PDF/CSV", 0.91, 0.44, "#dc2626", 1.2)

    plt.tight_layout(pad=0.2)
    fig.savefig(os.path.join(OUT_DIR, "fig3_1_architecture.png"), dpi=220)
    plt.close(fig)


def make_fig3_2_dfd():
    fig, ax = plt.subplots(figsize=(9.2, 4.6), dpi=220)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.add_patch(FancyBboxPatch((0.01, 0.01), 0.98, 0.98, boxstyle="round,pad=0.005,rounding_size=0.015",
                                facecolor="#fafbfd", edgecolor="#d0d7de", linewidth=1.0, zorder=1))

    draw_box(ax, 0.03, 0.42, 0.15, 0.18, "Candidate /\nUser", "(Webcam, Mic,\nKeyboard)", "#dbeafe", "#2563eb", 9.5, 7.5)
    draw_box(ax, 0.27, 0.74, 0.22, 0.16, "1.0 Video Emotion", "Face Detect + Xception CNN", "#fef3c7", "#d97706", 9, 7.4)
    draw_box(ax, 0.27, 0.44, 0.22, 0.16, "2.0 Audio Emotion", "Ring Buffer + Log-Mel + CNN-LSTM", "#dcfce7", "#16a34a", 9, 7.2)
    draw_box(ax, 0.27, 0.14, 0.22, 0.16, "3.0 Text Personality", "Clean + Tokenize + 1D CNN-LSTM", "#ede9fe", "#7c3aed", 9, 7.2)
    draw_box(ax, 0.59, 0.44, 0.21, 0.20, "4.0 Interview Coach\n& Report Engine", "Late Fusion, Timeline,\nScorecard & 2-Page PDF", "#fee2e2", "#dc2626", 8.8, 7.2)

    def draw_store(x, y, w, h, tag, name):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="square,pad=0", facecolor="white", edgecolor="#334155", linewidth=1.3, zorder=3))
        ax.plot([x + 0.04, x + 0.04], [y, y + h], color="#334155", linewidth=1.2, zorder=4)
        ax.text(x + 0.02, y + h / 2, tag, ha="center", va="center", fontsize=8.2, fontweight="bold", color="#1e293b", zorder=5)
        ax.text(x + 0.04 + (w - 0.04) / 2, y + h / 2, name, ha="center", va="center", fontsize=8.0, color="#1e293b", zorder=5)

    draw_store(0.60, 0.80, 0.36, 0.10, "D1", "Pre-trained Models (.h5, .hdf5, .pickle)")
    draw_store(0.60, 0.16, 0.36, 0.10, "D2", "Attempt History (interview_history.json)")
    draw_store(0.83, 0.48, 0.14, 0.12, "D3", "PDF & CSV\nReports")

    draw_arrow(ax, 0.18, 0.56, 0.27, 0.82, "Webcam Frames", 0.20, 0.72)
    draw_arrow(ax, 0.18, 0.51, 0.27, 0.52, "16kHz Mic Stream", 0.22, 0.56)
    draw_arrow(ax, 0.18, 0.46, 0.27, 0.22, "Typed / Spoken Text", 0.19, 0.30)

    draw_arrow(ax, 0.49, 0.80, 0.59, 0.60, "Face Probs", 0.54, 0.73)
    draw_arrow(ax, 0.49, 0.52, 0.59, 0.54, "Voice Probs", 0.54, 0.56)
    draw_arrow(ax, 0.49, 0.22, 0.59, 0.47, "Big-Five Traits", 0.54, 0.32)

    draw_arrow(ax, 0.60, 0.85, 0.49, 0.85, "Weights", 0.545, 0.88, "#64748b", 1.0)
    draw_arrow(ax, 0.69, 0.44, 0.69, 0.26, "Save Score", 0.74, 0.35)
    draw_arrow(ax, 0.80, 0.54, 0.83, 0.54)

    plt.tight_layout(pad=0.2)
    fig.savefig(os.path.join(OUT_DIR, "fig3_2_dfd.png"), dpi=220)
    plt.close(fig)


def make_fig3_3_models():
    fig, ax = plt.subplots(figsize=(9.2, 4.9), dpi=220)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    cols = [
        (0.03, "Video: Xception CNN (FER2013)", "#dbeafe", "#2563eb", [
            ("Input Face Crop", "48 × 48 × 1 Grayscale"),
            ("Entry Conv + BatchNorm", "32 & 64 filters (ReLU)"),
            ("4× Residual Separable Blocks", "Depthwise Separable Conv2D + Add"),
            ("GlobalAvgPool2D + Softmax", "7 Facial Emotion Classes"),
        ]),
        (0.355, "Audio: Time-Dist CNN+LSTM", "#dcfce7", "#16a34a", [
            ("Rolling Audio (3.07 s)", "5 × 128 × 128 × 1 Log-Mel"),
            ("4× TimeDistributed LFLBs", "Conv2D (64,64,128,128) + ELU + BN"),
            ("TimeDistributed Flatten", "Sequential Feature Vectors"),
            ("LSTM(256) + Dense(7)", "7 Speech Emotion Classes"),
        ]),
        (0.68, "Text: 1D CNN + 3×LSTM", "#ede9fe", "#7c3aed", [
            ("Cleaned & Padded Sequence", "Length 300 Word IDs (Tokenizer)"),
            ("Embedding Layer", "Vocab 22,347 → 300-dim"),
            ("3× Conv1D + MaxPool + BN", "Filters: 128 → 256 → 384 (k=8)"),
            ("3× Stacked LSTM(180) → Dense(5)", "5 Big-Five Personality Traits"),
        ]),
    ]

    for x0, header, fc, ec, blocks in cols:
        ax.add_patch(FancyBboxPatch((x0, 0.03), 0.29, 0.94, boxstyle="round,pad=0.008,rounding_size=0.015",
                                    facecolor="#fafbfd", edgecolor=ec, linewidth=1.3, zorder=1))
        ax.text(x0 + 0.145, 0.92, header, ha="center", va="center", fontsize=8.8, fontweight="bold", color=ec, zorder=3)
        ys = [0.74, 0.52, 0.30, 0.08]
        for idx, ((t, s), y) in enumerate(zip(blocks, ys)):
            draw_box(ax, x0 + 0.02, y, 0.25, 0.14, t, s, fc, ec, 8.2, 7.0)
            if idx < len(blocks) - 1:
                draw_arrow(ax, x0 + 0.145, y, x0 + 0.145, ys[idx + 1] + 0.14, color=ec, lw=1.2)

    plt.tight_layout(pad=0.2)
    fig.savefig(os.path.join(OUT_DIR, "fig3_3_models.png"), dpi=220)
    plt.close(fig)


def make_fig3_4_flowchart():
    fig, ax = plt.subplots(figsize=(9.0, 5.6), dpi=220)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.add_patch(FancyBboxPatch((0.01, 0.01), 0.98, 0.98, boxstyle="round,pad=0.005,rounding_size=0.015",
                                facecolor="#fafbfd", edgecolor="#d0d7de", linewidth=1.0, zorder=1))

    draw_box(ax, 0.40, 0.90, 0.20, 0.065, "START", "", "#dcfce7", "#16a34a", 9.5)
    draw_diamond(ax, 0.50, 0.77, 0.34, 0.11, "TensorFlow Installed\nin Current Python?", "#fef3c7", "#d97706", 8.2)
    draw_box(ax, 0.74, 0.73, 0.22, 0.08, "Auto Re-Exec", "Switch to Python 3.12\nvia os.execv()", "#fee2e2", "#dc2626", 8.4, 7.0)
    draw_box(ax, 0.32, 0.59, 0.36, 0.085, "Launch 4-Tab Tkinter Desktop App",
             "Video  |  Audio  |  Text  |  Interview Coach", "#dbeafe", "#2563eb", 8.8, 7.3)
    draw_diamond(ax, 0.50, 0.44, 0.32, 0.11, "Selected Mode:\nInterview Coach?", "#fef3c7", "#d97706", 8.2)

    draw_box(ax, 0.05, 0.39, 0.24, 0.10, "Individual Live Tabs",
             "Video / Audio / Text\nLive Bars & Prediction", "#ede9fe", "#7c3aed", 8.5, 7.2)
    draw_box(ax, 0.32, 0.24, 0.36, 0.095, "Simultaneous Live Multimodal Capture",
             "Webcam (Xception) + Mic Ring Buffer (CNN-LSTM)\nScrolling Timeline + Live Readiness Score", "#dcfce7", "#16a34a", 8.5, 7.1)
    draw_box(ax, 0.32, 0.10, 0.36, 0.09, "Stop, Transcribe & Export Report",
             "Big-Five Text Model + History JSON\nGenerate 2-Page Visual PDF + CSV Log", "#e0f2fe", "#0284c7", 8.5, 7.1)
    draw_box(ax, 0.77, 0.115, 0.16, 0.06, "END", "", "#fee2e2", "#dc2626", 9.5)

    draw_arrow(ax, 0.50, 0.90, 0.50, 0.825)
    draw_arrow(ax, 0.67, 0.77, 0.74, 0.77, "No", 0.70, 0.80)
    draw_arrow(ax, 0.85, 0.73, 0.68, 0.635)
    draw_arrow(ax, 0.50, 0.715, 0.50, 0.675, "Yes", 0.53, 0.695)
    draw_arrow(ax, 0.50, 0.59, 0.50, 0.495)
    draw_arrow(ax, 0.34, 0.44, 0.29, 0.44, "No", 0.315, 0.465)
    draw_arrow(ax, 0.50, 0.385, 0.50, 0.335, "Yes", 0.53, 0.36)
    draw_arrow(ax, 0.50, 0.24, 0.50, 0.19)
    draw_arrow(ax, 0.68, 0.145, 0.77, 0.145, "Done", 0.725, 0.175)

    plt.tight_layout(pad=0.2)
    fig.savefig(os.path.join(OUT_DIR, "fig3_4_flowchart.png"), dpi=220)
    plt.close(fig)


def make_fig5_outputs():
    import cv2
    import librosa
    import app
    import interview_coach

    wav_path = os.path.join(WEBAPP_DIR, "tmp", "voice_recording.wav")
    voice_scores = app.predict_voice_emotion(wav_path)

    sample_answer = (
        "I am an MCA student passionate about Python and artificial intelligence. "
        "I recently upgraded a multimodal emotion recognition project by building a clean "
        "Tkinter desktop interface with live video, audio, text analysis, and automated PDF reporting."
    )
    text_scores, _n_words, _top_words = app.predict_personality(sample_answer)

    vmodel = app.get_model("video")
    img = cv2.imread(os.path.join(WEBAPP_DIR, "tmp", "t.jpg"), 0)
    face_in = (cv2.resize(img, (48, 48)).astype(np.float32) / 255.0).reshape(1, 48, 48, 1)
    vprobs = np.asarray(vmodel.predict_on_batch(face_in))[0]
    face_scores = {app.FACE_EMOTIONS[i]: float(vprobs[i]) * 100.0 for i in range(len(app.FACE_EMOTIONS))}

    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.3), dpi=220)
    panels = [
        (axes[0], "Video Model (Xception CNN)", face_scores, "#3182ce"),
        (axes[1], "Audio Model (CNN + LSTM)", voice_scores, "#805ad5"),
        (axes[2], "Text Model (1D CNN + 3×LSTM)", text_scores, "#319795"),
    ]
    for ax, title, d, col in panels:
        labels = list(d.keys())
        vals = [d[k] for k in labels]
        y_pos = np.arange(len(labels))
        ax.barh(y_pos, vals, color=col)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlim(0, max(100, max(vals) * 1.25))
        ax.set_xlabel("Confidence (%)", fontsize=8)
        ax.set_title(title, fontsize=9.2, fontweight="bold", pad=6)
        ax.grid(True, axis="x", linestyle=":", alpha=0.5)
        for i, v in enumerate(vals):
            ax.text(v + 1.5, i, f"{v:.1f}%", va="center", fontsize=7.2, color="#1a202c")
    plt.tight_layout(pad=0.8)
    fig.savefig(os.path.join(OUT_DIR, "fig5_1_inference.png"), dpi=220)
    plt.close(fig)

    y, sr = librosa.load(wav_path, sr=16000)
    timeline = [
        {"t": float(i),
         "face_emotion": ["Happy", "Happy", "Neutral", "Neutral", "Fear", "Sad", "Happy", "Neutral", "Happy", "Happy"][i - 1],
         "face_conf": 76.0,
         "voice_emotion": ["Happy", "Neutral", "Neutral", "Fear", "Fear", "Neutral", "Happy", "Happy", "Neutral", "Happy"][i - 1],
         "voice_conf": 68.0,
         "positivity": [82, 88, 76, 52, 28, 35, 79, 84, 80, 86][i - 1],
         "stress": [18, 12, 24, 48, 72, 65, 21, 16, 20, 14][i - 1],
         "aligned": [True, True, True, False, True, False, True, True, True, True][i - 1]}
        for i in range(1, 11)
    ]
    face_counts = {"Happy": 5, "Neutral": 3, "Fear": 1, "Sad": 1, "Angry": 0, "Disgust": 0, "Surprise": 0}
    voice_counts = {"Happy": 4, "Neutral": 4, "Fear": 2, "Sad": 0, "Angry": 0, "Disgust": 0, "Surprise": 0}
    metrics = interview_coach._compute_metrics(timeline, 9, 10, face_counts, voice_counts, text_scores)
    session = {
        "timestamp": "2026-11-26 09:45:00",
        "question": interview_coach.QUESTIONS[1],
        "duration": 10.0,
        "timeline": timeline,
        "face_counts": face_counts,
        "voice_counts": voice_counts,
        "trait_scores": text_scores,
        "transcript": sample_answer,
        "audio_samples": y,
        "sample_rate": sr,
        "metrics": metrics,
    }
    history = [{"score": 61}, {"score": 68}, {"score": 74}, {"score": metrics["score"]}]

    f1 = interview_coach._draw_page1_executive(session, history)
    FigureCanvasAgg(f1)
    f1.savefig(os.path.join(OUT_DIR, "fig5_2_report_p1.png"), dpi=160)

    f2 = interview_coach._draw_page2_deep_dive(session, history)
    FigureCanvasAgg(f2)
    f2.savefig(os.path.join(OUT_DIR, "fig5_3_report_p2.png"), dpi=160)


def main():
    make_fig3_1_architecture()
    make_fig3_2_dfd()
    make_fig3_3_models()
    make_fig3_4_flowchart()
    make_fig5_outputs()
    print(f"All figures saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
