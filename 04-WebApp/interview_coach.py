
import csv
import datetime
import json
import os
import re
import threading
import time
from collections import Counter
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext

import cv2
import numpy as np
from PIL import Image, ImageTk

QUESTIONS = [
    "Tell me about yourself and your technical background.",
    "Describe a challenging project you built and how you solved problems.",
    "Why should we hire you for this role?",
    "Tell me about a time you handled pressure or tight deadlines.",
    "What are your biggest strengths and one area you are improving?",
]

POSITIVE_SET = {"Happy", "Neutral", "Surprise"}
STRESS_SET = {"Angry", "Disgust", "Fear", "Sad"}

CAM_SIZE = (300, 210)
MAX_AUDIO_RECORD_SECONDS = 180

def _sentiment_group(emotion):
    if emotion in POSITIVE_SET:
        return "positive"
    if emotion in STRESS_SET:
        return "stress"
    return None

def _compute_metrics(timeline, face_present, face_total, face_counts, voice_counts, trait_scores):
    if timeline:
        composure = float(np.mean([row["positivity"] for row in timeline]))
        max_stress_row = max(timeline, key=lambda r: r["stress"])
    else:
        composure = 0.0
        max_stress_row = None

    presence = (100.0 * face_present / face_total) if face_total else 0.0

    aligned_vals = [row["aligned"] for row in timeline if row["aligned"] is not None]
    if aligned_vals:
        alignment = 100.0 * sum(1 for v in aligned_vals if v) / len(aligned_vals)
    else:
        alignment = composure

    score = int(round(0.50 * composure + 0.25 * presence + 0.25 * alignment))

    top_face = max(face_counts, key=face_counts.get) if sum(face_counts.values()) else "N/A"
    top_voice = max(voice_counts, key=voice_counts.get) if sum(voice_counts.values()) else "N/A"
    top_trait = max(trait_scores, key=trait_scores.get) if trait_scores else "N/A"

    tips = []
    if face_total and presence < 75:
        tips.append("Keep your face centered toward the webcam for steadier eye contact.")
    if max_stress_row and max_stress_row["stress"] >= 60:
        sec = int(round(max_stress_row["t"]))
        tips.append(f"Tension peaked around {sec // 60}:{sec % 60:02d} — pause and breathe on harder points.")
    if aligned_vals and alignment < 65:
        tips.append("Face and voice tone disagreed at times — match your vocal warmth to your expression.")
    if composure >= 70 and timeline:
        tips.append("Strong composure and positive energy across your answer.")
    elif timeline and composure < 50:
        tips.append("Try relaxing your shoulders and smiling lightly to project more confidence.")
    if trait_scores:
        tips.append(f"Your wording highlighted {top_trait} ({trait_scores[top_trait]:.0f}%).")
    if not tips:
        tips.append("Complete a live session to receive personalized coaching tips.")

    return {
        "score": score,
        "composure": round(composure, 1),
        "presence": round(presence, 1),
        "alignment": round(alignment, 1),
        "top_face": top_face,
        "top_voice": top_voice,
        "top_trait": top_trait,
        "tips": tips[:3],
    }

def _try_transcribe_wav(wav_path):
    try:
        import speech_recognition as sr
    except ImportError:
        return None, "no_pkg"

    try:
        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_path) as source:
            audio = recognizer.record(source)
        text = recognizer.recognize_google(audio)
        return text.strip(), "ok"
    except Exception as err:
        return None, str(err)

REPORT_STOPWORDS = {
    "a", "about", "after", "all", "also", "am", "an", "and", "any", "are", "as", "at",
    "be", "because", "been", "being", "but", "by", "can", "could", "did", "do", "does",
    "for", "from", "get", "got", "had", "has", "have", "he", "her", "here", "him", "his",
    "how", "i", "if", "in", "into", "is", "it", "its", "just", "me", "more", "my", "no",
    "not", "now", "of", "on", "one", "only", "or", "other", "our", "out", "over", "really",
    "she", "so", "some", "than", "that", "the", "their", "them", "then", "there", "these",
    "they", "this", "those", "to", "too", "up", "us", "very", "was", "we", "were", "what",
    "when", "where", "which", "while", "who", "will", "with", "would", "you", "your",
}

FILLER_WORDS = {"um", "uh", "like", "actually", "basically", "literally", "well"}

def _extract_keywords(transcript, limit=8):
    if not transcript:
        return [], 0
    tokens = re.findall(r"[a-z]+", transcript.lower())
    fillers = sum(1 for w in tokens if w in FILLER_WORDS)
    meaningful = [w for w in tokens if len(w) >= 3 and w not in REPORT_STOPWORDS]
    return Counter(meaningful).most_common(limit), fillers

def _fmt_mmss(seconds):
    s = max(0, int(round(seconds)))
    return f"{s // 60}:{s % 60:02d}"

def _draw_page1_executive(session_data, history):
    from matplotlib.figure import Figure
    from matplotlib.patches import FancyBboxPatch

    timeline = session_data["timeline"]
    metrics = session_data["metrics"]
    face_counts = session_data["face_counts"]
    voice_counts = session_data["voice_counts"]
    trait_scores = session_data["trait_scores"]
    transcript = session_data["transcript"]

    fig = Figure(figsize=(8.27, 11.69), dpi=150)

    fig.text(0.07, 0.955, "Multimodal Interview Performance Report",
             fontsize=17, fontweight="bold", color="#1a202c")
    fig.text(0.93, 0.958, "Page 1 / 2", fontsize=9, color="#718096", ha="right")
    fig.text(0.07, 0.932, f"Date: {session_data['timestamp']}   |   Duration: {session_data['duration']:.0f} s",
             fontsize=9.2, color="#4a5568")
    q_text = session_data["question"]
    if len(q_text) > 92:
        q_text = q_text[:89] + "..."
    fig.text(0.07, 0.910, f"Question: {q_text}", fontsize=9.8, fontweight="bold", color="#2d3748")

    card = FancyBboxPatch(
        (0.07, 0.830), 0.86, 0.062,
        boxstyle="round,pad=0.008,rounding_size=0.01",
        transform=fig.transFigure, facecolor="#f7fafc", edgecolor="#cbd5e0", linewidth=0.9,
    )
    fig.patches.append(card)

    score_color = "#2f855a" if metrics["score"] >= 75 else ("#b7791f" if metrics["score"] >= 50 else "#c53030")
    fig.text(0.088, 0.858, f"Readiness Score: {metrics['score']} / 100",
             fontsize=13, fontweight="bold", color=score_color, va="center")
    fig.text(
        0.912, 0.870,
        f"Composure: {metrics['composure']:.0f}%   |   Presence: {metrics['presence']:.0f}%   |   Sync: {metrics['alignment']:.0f}%",
        fontsize=9, color="#2d3748", ha="right", va="center",
    )
    fig.text(
        0.912, 0.846,
        f"Top Face: {metrics['top_face']}   |   Top Voice: {metrics['top_voice']}   |   Top Trait: {metrics['top_trait']}",
        fontsize=8.8, color="#4a5568", ha="right", va="center",
    )

    ax1 = fig.add_axes([0.10, 0.56, 0.83, 0.21])
    if timeline:
        ts = [r["t"] for r in timeline]
        pos = [r["positivity"] for r in timeline]
        stress = [r["stress"] for r in timeline]
        ax1.plot(ts, pos, color="#2f855a", linewidth=2.2, label="Composure / Positivity (%)")
        ax1.fill_between(ts, pos, alpha=0.14, color="#2f855a")
        ax1.plot(ts, stress, color="#dd6b20", linewidth=2.0, linestyle="--", label="Stress / Tension (%)")
        max_t = max(5.0, ts[-1])
        ax1.set_xlim(left=0, right=max_t)

        peak_pos_idx = int(np.argmax(pos))
        peak_str_idx = int(np.argmax(stress))
        pt_x, pt_y = ts[peak_pos_idx], pos[peak_pos_idx]
        st_x, st_y = ts[peak_str_idx], stress[peak_str_idx]

        ax1.scatter([pt_x], [pt_y], color="#22543d", s=42, zorder=5)
        ax1.annotate(
            f"Peak Confidence @ {_fmt_mmss(pt_x)} ({pt_y:.0f}%)",
            xy=(pt_x, pt_y),
            xytext=(8 if pt_x <= max_t * 0.55 else -135, -18 if pt_y > 82 else 11),
            textcoords="offset points",
            fontsize=7.8, fontweight="bold", color="#22543d",
            bbox=dict(boxstyle="round,pad=0.22", fc="#f0fff4", ec="#2f855a", lw=0.8),
            arrowprops=dict(arrowstyle="->", color="#2f855a", lw=0.9),
        )

        if st_y >= 15:
            ax1.scatter([st_x], [st_y], color="#9c4221", s=42, zorder=5)
            near_pos = abs(st_x - pt_x) < max(2.0, max_t * 0.22) and abs(st_y - pt_y) < 20
            dy = -22 if near_pos else (-18 if st_y > 82 else 11)
            ax1.annotate(
                f"Peak Tension @ {_fmt_mmss(st_x)} ({st_y:.0f}%)",
                xy=(st_x, st_y),
                xytext=(8 if st_x <= max_t * 0.55 else -125, dy),
                textcoords="offset points",
                fontsize=7.8, fontweight="bold", color="#9c4221",
                bbox=dict(boxstyle="round,pad=0.22", fc="#fffaf0", ec="#dd6b20", lw=0.8),
                arrowprops=dict(arrowstyle="->", color="#dd6b20", lw=0.9),
            )

        ax1.legend(loc="lower right", bbox_to_anchor=(1.0, 1.02), ncol=2,
                   fontsize=8.0, frameon=False)
    else:
        ax1.text(0.5, 0.5, "No timeline data recorded", ha="center", va="center", color="#718096")
    ax1.set_ylim(0, 114)
    ax1.set_title("Composure vs. Stress Timeline", fontsize=10.8, fontweight="bold",
                  loc="left", pad=12)
    ax1.set_xlabel("Time (seconds)", fontsize=8.8)
    ax1.set_ylabel("Score (%)", fontsize=8.8)
    ax1.grid(True, linestyle=":", alpha=0.5)

    ax2 = fig.add_axes([0.10, 0.29, 0.36, 0.19])
    emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
    f_tot = sum(face_counts.values()) or 1
    v_tot = sum(voice_counts.values()) or 1
    f_pct = [100.0 * face_counts.get(e, 0) / f_tot for e in emotions]
    v_pct = [100.0 * voice_counts.get(e, 0) / v_tot for e in emotions]
    x = np.arange(len(emotions))
    ax2.bar(x - 0.18, f_pct, width=0.35, label="Face", color="#3182ce")
    ax2.bar(x + 0.18, v_pct, width=0.35, label="Voice", color="#805ad5")
    ax2.set_xticks(x)
    ax2.set_xticklabels(emotions, rotation=32, ha="right", fontsize=8)
    ax2.set_ylim(0, 108)
    ax2.set_ylabel("% of detections", fontsize=8.5)
    ax2.set_title("Face vs. Voice Emotions", fontsize=10.5, fontweight="bold", pad=8)
    ax2.legend(loc="upper right", fontsize=7.8, frameon=True)
    ax2.grid(True, axis="y", linestyle=":", alpha=0.5)

    ax3 = fig.add_axes([0.62, 0.29, 0.31, 0.19])
    if trait_scores:
        names = list(trait_scores.keys())
        vals = [trait_scores[n] for n in names]
        y_pos = np.arange(len(names))
        ax3.barh(y_pos, vals, color="#319795")
        for idx, val in enumerate(vals):
            ax3.text(min(96, val + 2), idx, f"{val:.0f}%", va="center", fontsize=8, color="#234e52")
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels(names, fontsize=8.3)
        ax3.set_xlim(0, 108)
        ax3.set_xlabel("Score (%)", fontsize=8.5)
        ax3.set_title("Big-Five Personality (Text)", fontsize=10.5, fontweight="bold", pad=8)
        ax3.grid(True, axis="x", linestyle=":", alpha=0.5)
    else:
        ax3.text(0.5, 0.5, "Add or transcribe text\nto include Big-Five traits",
                 ha="center", va="center", fontsize=8.8, color="#718096")
        ax3.set_xticks([])
        ax3.set_yticks([])
        ax3.set_title("Big-Five Personality (Text)", fontsize=10.5, fontweight="bold", pad=8)

    fig.text(0.07, 0.215, "Coaching Feedback", fontsize=11, fontweight="bold", color="#1a202c")
    y_cursor = 0.190
    for tip in metrics["tips"]:
        fig.text(0.08, y_cursor, f"• {tip}", fontsize=9.2, color="#2d3748")
        y_cursor -= 0.022

    if history:
        scores_all = [h["score"] for h in history]
        fig.text(
            0.07, 0.118,
            f"Session History: {len(history)} attempt(s)   |   Best Score: {max(scores_all)}/100   |   Average: {np.mean(scores_all):.0f}/100",
            fontsize=9, color="#4a5568",
        )

    if transcript:
        snippet = transcript.replace("\n", " ").strip()
        if len(snippet) > 250:
            snippet = snippet[:247] + "..."
        fig.text(0.07, 0.088, "Answer Transcript:", fontsize=9.5, fontweight="bold", color="#1a202c")
        fig.text(0.07, 0.072, snippet, fontsize=8.5, color="#4a5568", wrap=True, va="top")

    return fig

def _draw_page2_deep_dive(session_data, history):
    from matplotlib.figure import Figure

    timeline = session_data["timeline"]
    trait_scores = session_data["trait_scores"]
    transcript = session_data["transcript"]
    audio_samples = session_data.get("audio_samples")
    sample_rate = session_data.get("sample_rate", 16000)

    fig = Figure(figsize=(8.27, 11.69), dpi=150)

    fig.text(0.07, 0.955, "Deep-Dive Multimodal Analytics",
             fontsize=17, fontweight="bold", color="#1a202c")
    fig.text(0.93, 0.958, "Page 2 / 2", fontsize=9, color="#718096", ha="right")
    fig.text(0.07, 0.932, f"Date: {session_data['timestamp']}   |   Voice, Face-Voice Fusion, Personality & Progress",
             fontsize=9.2, color="#4a5568")

    ax_rad = fig.add_axes([0.08, 0.66, 0.34, 0.20], polar=True)
    traits_order = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]
    angles = np.linspace(0, 2 * np.pi, len(traits_order), endpoint=False).tolist()
    angles_closed = angles + [angles[0]]
    ax_rad.set_theta_offset(np.pi / 2)
    ax_rad.set_theta_direction(-1)
    ax_rad.set_xticks(angles)
    ax_rad.set_xticklabels(traits_order, fontsize=7.5)
    ax_rad.set_ylim(0, 100)
    ax_rad.set_yticks([25, 50, 75, 100])
    ax_rad.set_yticklabels(["25", "50", "75", "100"], fontsize=6.8, color="#718096")
    ax_rad.set_title("Big-Five Personality Radar", fontsize=10.5, fontweight="bold", pad=13)
    if trait_scores:
        vals = [float(trait_scores.get(t, 0.0)) for t in traits_order]
        vals_closed = vals + [vals[0]]
        ax_rad.plot(angles_closed, vals_closed, color="#2c7a7b", linewidth=2.0, marker="o", markersize=4)
        ax_rad.fill(angles_closed, vals_closed, color="#319795", alpha=0.25)
    else:
        ax_rad.text(0, 0, "Add/transcribe text\nfor radar chart", ha="center", va="center",
                    fontsize=8.2, color="#718096")

    ax_hm = fig.add_axes([0.58, 0.66, 0.35, 0.21])
    emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
    emo_idx = {e: i for i, e in enumerate(emotions)}
    mat = np.zeros((len(emotions), len(emotions)), dtype=int)
    for row in timeline:
        fe, ve = row.get("face_emotion"), row.get("voice_emotion")
        if fe in emo_idx and ve in emo_idx:
            mat[emo_idx[fe], emo_idx[ve]] += 1

    ax_hm.set_title("Face × Voice Co-Occurrence Matrix", fontsize=10.5, fontweight="bold", pad=8)
    if mat.sum() > 0:
        ax_hm.imshow(mat, cmap="YlGnBu", aspect="auto")
        ax_hm.set_xticks(range(len(emotions)))
        ax_hm.set_yticks(range(len(emotions)))
        ax_hm.set_xticklabels(emotions, rotation=32, ha="right", fontsize=7.5)
        ax_hm.set_yticklabels(emotions, fontsize=7.5)
        ax_hm.set_xlabel("Voice Emotion", fontsize=8.3, labelpad=3)
        ax_hm.set_ylabel("Face Emotion", fontsize=8.3)
        vmax = mat.max()
        for i in range(len(emotions)):
            for j in range(len(emotions)):
                if mat[i, j] > 0:
                    txt_color = "white" if mat[i, j] >= vmax * 0.6 else "#1a202c"
                    ax_hm.text(j, i, str(mat[i, j]), ha="center", va="center",
                               fontsize=8, fontweight="bold", color=txt_color)
    else:
        ax_hm.text(0.5, 0.5, "Needs both Face & Voice\ndetections in session",
                   ha="center", va="center", fontsize=8.8, color="#718096")
        ax_hm.set_xticks([])
        ax_hm.set_yticks([])

    ax_wav = fig.add_axes([0.10, 0.36, 0.83, 0.18])
    if audio_samples is not None and len(audio_samples) > sample_rate // 4:
        sig = np.asarray(audio_samples, dtype=np.float32).ravel()
        total_sec = len(sig) / float(sample_rate)

        bins = min(1400, len(sig))
        step = max(1, len(sig) // bins)
        trimmed = sig[: (len(sig) // step) * step].reshape(-1, step)
        t_env = np.linspace(0, total_sec, trimmed.shape[0])
        ymin = trimmed.min(axis=1)
        ymax = trimmed.max(axis=1)

        win = max(1, int(sample_rate * 0.25))
        n_win = len(sig) // win
        active_pct = 0.0
        pauses = 0
        if n_win > 0:
            frames = sig[: n_win * win].reshape(n_win, win)
            rms_w = np.sqrt(np.mean(frames ** 2, axis=1))
            thresh = max(0.003, float(np.percentile(rms_w, 90)) * 0.18)
            speaking = rms_w >= thresh
            active_pct = 100.0 * float(np.mean(speaking))

            in_speech = False
            for idx_w, is_sp in enumerate(speaking):
                t0 = idx_w * 0.25
                t1 = (idx_w + 1) * 0.25
                if is_sp:
                    ax_wav.axvspan(t0, t1, color="#c6f6d5", alpha=0.55, lw=0)
                    in_speech = True
                elif in_speech:
                    pauses += 1
                    in_speech = False

        ax_wav.fill_between(t_env, ymin, ymax, color="#3182ce", alpha=0.85, linewidth=0.4)
        ax_wav.axhline(0, color="#a0aec0", linewidth=0.6)
        peak = max(0.05, float(np.max(np.abs(sig))) * 1.15)
        ax_wav.set_ylim(-peak, peak)
        ax_wav.set_xlim(0, total_sec)
        ax_wav.set_title(
            f"Voice Waveform & Speaking Activity   (Speaking: {active_pct:.0f}%  |  Pauses: {pauses})",
            fontsize=10.5, fontweight="bold", pad=8,
        )
        ax_wav.set_xlabel("Time (seconds)", fontsize=8.8)
        ax_wav.set_ylabel("Amplitude", fontsize=8.8)
        ax_wav.grid(True, linestyle=":", alpha=0.45)
    else:
        ax_wav.text(0.5, 0.5, "No microphone waveform recorded in this session",
                    ha="center", va="center", fontsize=9.2, color="#718096")
        ax_wav.set_xticks([])
        ax_wav.set_yticks([])
        ax_wav.set_title("Voice Waveform & Speaking Activity", fontsize=10.5, fontweight="bold", pad=8)

    ax_prog = fig.add_axes([0.10, 0.08, 0.36, 0.19])
    ax_prog.set_title("Progress Across Attempts", fontsize=10.5, fontweight="bold", pad=8)
    if history:
        recent = history[-10:]
        xs = list(range(1, len(recent) + 1))
        ys = [h["score"] for h in recent]
        ax_prog.plot(xs, ys, color="#3182ce", linewidth=2.0, marker="o", markersize=6)
        best_i = int(np.argmax(ys))
        ax_prog.scatter([xs[best_i]], [ys[best_i]], color="#d69e2e", s=95, zorder=5,
                        label=f"Best: {ys[best_i]}/100")
        for x_pt, y_pt in zip(xs, ys):
            ax_prog.text(x_pt, min(103, y_pt + 4), str(y_pt), ha="center", fontsize=8, color="#2d3748")
        if len(ys) >= 2:
            avg = float(np.mean(ys))
            ax_prog.axhline(avg, color="#718096", linestyle="--", linewidth=1.0,
                            label=f"Avg: {avg:.0f}")
        ax_prog.set_xticks(xs)
        ax_prog.set_ylim(0, 114)
        ax_prog.set_xlabel("Recent Attempts", fontsize=8.5)
        ax_prog.set_ylabel("Readiness Score", fontsize=8.5)
        ax_prog.legend(loc="lower right", fontsize=8)
        ax_prog.grid(True, linestyle=":", alpha=0.5)
    else:
        ax_prog.text(0.5, 0.5, "Complete more sessions\nto track score progress",
                     ha="center", va="center", fontsize=8.8, color="#718096")
        ax_prog.set_xticks([])
        ax_prog.set_yticks([])

    ax_kw = fig.add_axes([0.61, 0.08, 0.32, 0.19])
    top_kw, filler_cnt = _extract_keywords(transcript, limit=8)
    title_kw = "Top Spoken Keywords" + (f" (Fillers: {filler_cnt})" if transcript else "")
    ax_kw.set_title(title_kw, fontsize=10.5, fontweight="bold", pad=8)
    if top_kw:
        words = [w for w, _ in reversed(top_kw)]
        counts = [c for _, c in reversed(top_kw)]
        y_pos = np.arange(len(words))
        ax_kw.barh(y_pos, counts, color="#805ad5")
        max_c = max(counts)
        for idx_k, cnt in enumerate(counts):
            ax_kw.text(cnt + max(0.1, max_c * 0.03), idx_k, str(cnt),
                       va="center", fontsize=8, color="#44337a")
        ax_kw.set_yticks(y_pos)
        ax_kw.set_yticklabels(words, fontsize=8.3)
        ax_kw.set_xlim(0, max_c * 1.22 + 0.5)
        ax_kw.set_xlabel("Frequency", fontsize=8.5)
        ax_kw.grid(True, axis="x", linestyle=":", alpha=0.5)
    else:
        ax_kw.text(0.5, 0.5, "Add or transcribe text\nto see top keywords",
                   ha="center", va="center", fontsize=8.8, color="#718096")
        ax_kw.set_xticks([])
        ax_kw.set_yticks([])

    return fig

def export_interview_report(pdf_path, session_data, history):
    from matplotlib.backends.backend_pdf import PdfPages

    timeline = session_data["timeline"]

    csv_path = os.path.splitext(pdf_path)[0] + "_timeline.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "t", "face_emotion", "face_conf",
                "voice_emotion", "voice_conf",
                "positivity", "stress", "aligned",
            ],
        )
        writer.writeheader()
        for row in timeline:
            writer.writerow(row)

    fig1 = _draw_page1_executive(session_data, history)
    fig2 = _draw_page2_deep_dive(session_data, history)
    with PdfPages(pdf_path) as pdf:
        pdf.savefig(fig1)
        pdf.savefig(fig2)

    return pdf_path, csv_path

class TimelineCanvas(tk.Canvas):

    def __init__(self, parent, height=120):
        super().__init__(parent, height=height, bg="#181c22", highlightthickness=1,
                         highlightbackground="#3a3f4b")
        self._data = []
        self.bind("<Configure>", lambda _e: self.redraw())

    def set_data(self, timeline):
        self._data = timeline
        self.redraw()

    def redraw(self):
        self.delete("all")
        w = max(self.winfo_width(), 400)
        h = max(self.winfo_height(), 100)
        pad_l, pad_r, pad_t, pad_b = 38, 16, 16, 22
        plot_w = max(10, w - pad_l - pad_r)
        plot_h = max(10, h - pad_t - pad_b)

        for pct in (0, 50, 100):
            y = pad_t + plot_h * (1 - pct / 100.0)
            self.create_line(pad_l, y, w - pad_r, y, fill="#2e3440", dash=(2, 3))
            self.create_text(pad_l - 6, y, text=f"{pct}%", anchor="e",
                             fill="#9aa5b1", font=("Helvetica", 9))

        self.create_line(pad_l + 6, 8, pad_l + 24, 8, fill="#48bb78", width=2)
        self.create_text(pad_l + 28, 8, text="Composure / Positive", anchor="w",
                         fill="#cbd5e0", font=("Helvetica", 9))
        self.create_line(pad_l + 170, 8, pad_l + 188, 8, fill="#ed8936", width=2)
        self.create_text(pad_l + 192, 8, text="Stress / Tension", anchor="w",
                         fill="#cbd5e0", font=("Helvetica", 9))

        if len(self._data) < 2:
            self.create_text(
                pad_l + plot_w / 2, pad_t + plot_h / 2,
                text="Live emotion timeline will appear here once the interview starts...",
                fill="#718096", font=("Helvetica", 10),
            )
            return

        window = self._data[-60:]
        t0, t1 = window[0]["t"], window[-1]["t"]
        span = max(1.0, t1 - t0)

        pos_pts = []
        stress_pts = []
        for row in window:
            x = pad_l + ((row["t"] - t0) / span) * plot_w
            y_pos = pad_t + plot_h * (1 - max(0.0, min(100.0, row["positivity"])) / 100.0)
            y_str = pad_t + plot_h * (1 - max(0.0, min(100.0, row["stress"])) / 100.0)
            pos_pts.extend([x, y_pos])
            stress_pts.extend([x, y_str])

        self.create_line(*pos_pts, fill="#48bb78", width=2.2, smooth=True)
        self.create_line(*stress_pts, fill="#ed8936", width=2.0, smooth=True)

        self.create_text(pad_l, h - 8, text=f"{t0:.0f}s", anchor="w",
                         fill="#9aa5b1", font=("Helvetica", 9))
        self.create_text(w - pad_r, h - 8, text=f"{t1:.0f}s", anchor="e",
                         fill="#9aa5b1", font=("Helvetica", 9))

class InterviewTab(ttk.Frame):

    def __init__(self, parent, app_api, stop_other_tabs=None):
        super().__init__(parent, padding=10)
        self.api = app_api
        self.stop_other_tabs = stop_other_tabs

        self.history_path = os.path.join(self.api["TMP_DIR"], "interview_history.json")
        self.history = self._load_history()

        self.running = False
        self._starting = False
        self.cap = None
        self.stream = None
        self.video_model = None
        self.audio_model = None
        self.face_finder = self.api["make_face_detector"]()

        self.buf_lock = threading.Lock()
        self.audio_buf = np.zeros(self.api["AUDIO_CHUNK_SIZE"], dtype=np.float32)
        self.samples_received = 0
        self.mic_level = 0.0
        self.recorded_chunks = []
        self.audio_predicting = False

        self.start_time = 0.0
        self.duration_limit = 45
        self.frame_no = 0
        self.faces = []

        self.timeline = []
        self.face_counts = {e: 0 for e in self.api["FACE_EMOTIONS"]}
        self.voice_counts = {e: 0 for e in self.api["VOICE_EMOTIONS"]}
        self.face_present_checks = 0
        self.face_total_checks = 0
        self.last_face_emotion = ""
        self.last_face_conf = 0.0
        self.last_face_probs = None
        self.last_voice_emotion = ""
        self.last_voice_conf = 0.0
        self.last_voice_probs = None
        self.trait_scores = {}
        self.last_session = None

        self._text_after_id = None
        self._text_busy = False
        self._text_pending = False
        self._last_text = ""

        self._build_ui()
        self._refresh_history_label()

    def _build_ui(self):
        ttk.Label(self, text="AI Multimodal Interview Coach (Live)",
                  font=("Helvetica", 16, "bold")).pack(pady=(0, 6))

        top_row = ttk.Frame(self)
        top_row.pack(fill="x", pady=(0, 6))
        ttk.Label(top_row, text="Question:").pack(side="left", padx=(0, 5))
        self.question_var = tk.StringVar(value=QUESTIONS[0])
        self.question_box = ttk.Combobox(top_row, textvariable=self.question_var,
                                         values=QUESTIONS, width=56)
        self.question_box.pack(side="left", fill="x", expand=True, padx=(0, 10))

        ttk.Label(top_row, text="Timer:").pack(side="left", padx=(0, 4))
        self.timer_var = tk.StringVar(value="45 s")
        self.timer_box = ttk.Combobox(top_row, textvariable=self.timer_var,
                                      values=["30 s", "45 s", "60 s", "No limit"],
                                      state="readonly", width=9)
        self.timer_box.pack(side="left")

        ctrl_row = ttk.Frame(self)
        ctrl_row.pack(fill="x", pady=(0, 4))
        self.start_btn = ttk.Button(ctrl_row, text="Start Interview", command=self.start)
        self.start_btn.pack(side="left", padx=(0, 6))
        self.stop_btn = ttk.Button(ctrl_row, text="Stop Interview", command=self.stop)
        self.stop_btn.pack(side="left", padx=(0, 6))
        self.export_btn = ttk.Button(ctrl_row, text="Save Report (PDF + CSV)",
                                     command=self.save_report)
        self.export_btn.pack(side="left", padx=(0, 12))
        self.export_btn.state(["disabled"])

        self.clock_lbl = ttk.Label(ctrl_row, text="Time: 00:00", font=("Helvetica", 12, "bold"))
        self.clock_lbl.pack(side="left", padx=(0, 12))
        self.history_lbl = ttk.Label(ctrl_row, text="", font=("Helvetica", 11))
        self.history_lbl.pack(side="right")

        self.status = ttk.Label(self, text="Pick a question and click 'Start Interview'.",
                                font=("Helvetica", 12))
        self.status.pack(anchor="w", pady=(0, 6))

        mid = ttk.Frame(self)
        mid.pack(fill="x", pady=(0, 6))

        left = ttk.Frame(mid)
        left.pack(side="left", padx=(0, 14), anchor="n")
        self.blank = ImageTk.PhotoImage(Image.new("RGB", CAM_SIZE, "black"))
        self.video_lbl = tk.Label(left, image=self.blank, bg="black")
        self.video_lbl.pack()

        mic_row = ttk.Frame(left)
        mic_row.pack(fill="x", pady=(5, 0))
        ttk.Label(mic_row, text="Mic:").pack(side="left", padx=(0, 6))
        self.level_bar = ttk.Progressbar(mic_row, length=245, maximum=100)
        self.level_bar.pack(side="left", fill="x", expand=True)

        right = ttk.Frame(mid)
        right.pack(side="left", fill="both", expand=True, anchor="n")

        live_box = ttk.LabelFrame(right, text="Live Multimodal Readout", padding=8)
        live_box.pack(fill="x", pady=(0, 6))
        self.face_lbl = ttk.Label(live_box, text="Face Emotion:   —", font=("Helvetica", 12))
        self.face_lbl.pack(anchor="w")
        self.voice_lbl = ttk.Label(live_box, text="Voice Emotion:  —", font=("Helvetica", 12))
        self.voice_lbl.pack(anchor="w", pady=2)
        self.sync_lbl = ttk.Label(live_box, text="Face–Voice Sync: —", font=("Helvetica", 12))
        self.sync_lbl.pack(anchor="w")

        score_box = ttk.LabelFrame(right, text="Interview Scorecard & Coaching", padding=8)
        score_box.pack(fill="both", expand=True)
        self.score_lbl = ttk.Label(score_box, text="Readiness Score: — / 100",
                                   font=("Helvetica", 15, "bold"))
        self.score_lbl.pack(anchor="w")
        self.subscores_lbl = ttk.Label(
            score_box, text="Composure: —   |   Presence: —   |   Sync: —",
            font=("Helvetica", 11),
        )
        self.subscores_lbl.pack(anchor="w", pady=(2, 4))
        self.tips_lbl = ttk.Label(
            score_box,
            text="Start an interview to get live scoring and coaching tips.",
            wraplength=470, justify="left", font=("Helvetica", 11),
        )
        self.tips_lbl.pack(anchor="w")

        ttk.Label(self, text="Live Emotion Timeline:", font=("Helvetica", 11, "bold")).pack(anchor="w")
        self.timeline_canvas = TimelineCanvas(self, height=118)
        self.timeline_canvas.pack(fill="x", pady=(2, 6))

        tx_head = ttk.Frame(self)
        tx_head.pack(fill="x")
        ttk.Label(tx_head, text="Answer Transcript / Notes (updates Big-Five personality live):",
                  font=("Helvetica", 11, "bold")).pack(side="left")
        self.trait_lbl = ttk.Label(tx_head, text="Big-Five: —", font=("Helvetica", 11))
        self.trait_lbl.pack(side="right")

        self.transcript_box = scrolledtext.ScrolledText(self, height=3, wrap="word",
                                                        font=("Helvetica", 12))
        self.transcript_box.pack(fill="both", expand=True, pady=(3, 0))
        self.transcript_box.bind("<KeyRelease>", self._on_transcript_edit)
        self.transcript_box.bind("<<Paste>>", lambda _e: self.after_idle(self._on_transcript_edit))

    def _load_history(self):
        try:
            with open(self.history_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, list) else []
        except Exception:
            return []

    def _save_history(self):
        try:
            with open(self.history_path, "w", encoding="utf-8") as f:
                json.dump(self.history[-30:], f, indent=2)
        except Exception:
            pass

    def _refresh_history_label(self):
        if not self.history:
            self.history_lbl["text"] = "Attempts: 0"
            return
        scores = [h["score"] for h in self.history]
        text = f"Attempts: {len(scores)}  |  Best: {max(scores)}/100"
        if len(scores) >= 2:
            diff = scores[-1] - scores[-2]
            sign = "+" if diff >= 0 else ""
            text += f"  |  Last: {scores[-1]} ({sign}{diff})"
        self.history_lbl["text"] = text

    def start(self):
        if self.running or self._starting:
            return
        if self.face_finder is None:
            messagebox.showerror("Interview Coach",
                                 "No face detector available. Install opencv-python<5 or dlib.")
            return
        if self.stop_other_tabs:
            self.stop_other_tabs()

        choice = self.timer_var.get()
        self.duration_limit = None if choice == "No limit" else int(choice.split()[0])

        self._starting = True
        self.start_btn.state(["disabled"])
        self.export_btn.state(["disabled"])
        self.status["text"] = "Loading Video & Audio models... (first time takes a few seconds)"

        def load_both():
            v_model = self.api["get_model"]("video")
            a_model = self.api["get_model"]("audio")
            return v_model, a_model

        self.api["run_in_background"](self, load_both, self._begin_session, self._failed)

    def _begin_session(self, models):
        if not self._starting:
            return
        self._starting = False
        self.video_model, self.audio_model = models

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self._failed(RuntimeError("Could not open webcam. Check macOS Camera permission."))
            return

        with self.buf_lock:
            self.audio_buf.fill(0)
            self.samples_received = 0
            self.mic_level = 0.0
            self.recorded_chunks = []

        self.timeline = []
        self.face_counts = {e: 0 for e in self.api["FACE_EMOTIONS"]}
        self.voice_counts = {e: 0 for e in self.api["VOICE_EMOTIONS"]}
        self.face_present_checks = 0
        self.face_total_checks = 0
        self.last_face_emotion = ""
        self.last_face_conf = 0.0
        self.last_face_probs = None
        self.last_voice_emotion = ""
        self.last_voice_conf = 0.0
        self.last_voice_probs = None
        self.trait_scores = {}
        self.last_session = None
        self.frame_no = 0
        self.faces = []
        self.audio_predicting = False

        self.face_lbl["text"] = "Face Emotion:   Scanning..."
        self.voice_lbl["text"] = "Voice Emotion:  Buffering (~3 s)..."
        self.sync_lbl["text"] = "Face–Voice Sync: Waiting..."
        self.score_lbl["text"] = "Readiness Score: — / 100"
        self.subscores_lbl["text"] = "Composure: —   |   Presence: —   |   Sync: —"
        self.tips_lbl["text"] = "Answer the question naturally while looking at the camera."
        self.trait_lbl["text"] = "Big-Five: —"
        self.transcript_box.delete("1.0", "end")
        self._last_text = ""
        self.timeline_canvas.set_data([])

        mic_ok = True
        try:
            import sounddevice as sd
            self.stream = sd.InputStream(
                samplerate=self.api["SAMPLE_RATE"],
                channels=1,
                dtype="float32",
                blocksize=1600,
                callback=self._audio_callback,
            )
            self.stream.start()
        except Exception as err:
            mic_ok = False
            self.stream = None
            self.voice_lbl["text"] = f"Voice Emotion:  Mic unavailable ({type(err).__name__})"

        self.running = True
        self.start_time = time.time()
        self.status["text"] = ("Interview live — speak your answer!" if mic_ok
                               else "Interview live (video only — check Microphone permission).")
        self._next_frame()
        self._tick_timeline()
        if mic_ok:
            self._live_voice_step()

    def stop(self):
        was_running = self.running
        self.running = False
        self._starting = False

        if self.cap is not None:
            self.cap.release()
            self.cap = None
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None

        self.video_lbl.configure(image=self.blank)
        self.video_lbl.image = self.blank
        self.level_bar["value"] = 0
        self.start_btn.state(["!disabled"])

        if not was_running:
            return

        duration = max(1.0, time.time() - self.start_time)
        self._update_scorecard()

        with self.buf_lock:
            chunks = list(self.recorded_chunks)
        audio_all = np.concatenate(chunks).astype(np.float32) if chunks else None

        metrics = _compute_metrics(
            self.timeline, self.face_present_checks, self.face_total_checks,
            self.face_counts, self.voice_counts, self.trait_scores,
        )
        self.last_session = {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "question": self.question_var.get().strip() or "Practice Interview Question",
            "duration": duration,
            "timeline": list(self.timeline),
            "face_counts": dict(self.face_counts),
            "voice_counts": dict(self.voice_counts),
            "trait_scores": dict(self.trait_scores),
            "transcript": self.transcript_box.get("1.0", "end").strip(),
            "audio_samples": audio_all,
            "sample_rate": self.api["SAMPLE_RATE"],
            "metrics": metrics,
        }

        if len(self.timeline) >= 2:
            self.history.append({
                "timestamp": self.last_session["timestamp"],
                "question": self.last_session["question"],
                "score": metrics["score"],
                "top_face": metrics["top_face"],
                "top_voice": metrics["top_voice"],
            })
            self._save_history()
            self._refresh_history_label()
            self.export_btn.state(["!disabled"])

        if audio_all is not None:
            self.status["text"] = "Interview finished! Checking speech-to-text..."
            wav_path = os.path.join(self.api["TMP_DIR"], "interview_answer.wav")
            try:
                from scipy.io import wavfile
                pcm = np.clip(audio_all * 32767.0, -32768, 32767).astype(np.int16)
                wavfile.write(wav_path, self.api["SAMPLE_RATE"], pcm)
                self.api["run_in_background"](
                    self,
                    lambda p=wav_path: _try_transcribe_wav(p),
                    self._on_transcript_ready,
                    lambda _e: self._on_transcript_ready((None, "error")),
                )
            except Exception:
                self.status["text"] = "Interview complete! Click 'Save Report' or add notes below."
        else:
            self.status["text"] = "Interview complete! Click 'Save Report' to export PDF + CSV."

    def _on_transcript_ready(self, result):
        text, status = result
        if text:
            self.transcript_box.delete("1.0", "end")
            self.transcript_box.insert("1.0", text)
            self._last_text = ""
            self.status["text"] = "Interview complete! Answer transcribed — analysing personality..."
            self._run_text_live()
        elif status == "no_pkg":
            self.status["text"] = (
                "Interview complete! Click 'Save Report', or type your answer below for Big-Five "
                "(tip: pip install SpeechRecognition for auto-transcript)."
            )
        else:
            self.status["text"] = "Interview complete! Click 'Save Report' or type notes below."

    def _failed(self, err):
        self.running = False
        self._starting = False
        self.start_btn.state(["!disabled"])
        self.status["text"] = "Error."
        messagebox.showerror("Interview Coach", self.api["friendly_error"](err))

    def _next_frame(self):
        if not self.running or self.cap is None:
            return
        ok, frame = self.cap.read()
        if ok:
            frame = cv2.resize(cv2.flip(frame, 1), CAM_SIZE)
            self.frame_no += 1
            if self.frame_no % 5 == 0:
                self.faces = self._detect_face_emotions(frame)

            for (x, y, w, h, emotion, conf) in self.faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"{emotion} {conf:.0f}%", (x, max(20, y - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

            img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            self.video_lbl.configure(image=img)
            self.video_lbl.image = img
        self.after(25, self._next_frame)

    def _detect_face_emotions(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        boxes = self.face_finder(gray)
        self.face_total_checks += 1
        results = []
        emotions = self.api["FACE_EMOTIONS"]

        first_probs = None
        for (x, y, w, h) in boxes:
            crop = gray[y:y + h, x:x + w]
            if crop.size == 0:
                continue
            face = cv2.resize(crop, (48, 48)).astype(np.float32) / 255.0
            probs = np.asarray(self.video_model.predict_on_batch(face.reshape(1, 48, 48, 1)))[0]
            if first_probs is None:
                first_probs = probs
            i = int(np.argmax(probs))
            emo, conf = emotions[i], float(probs[i]) * 100.0
            results.append((x, y, w, h, emo, conf))

        if results:
            self.face_present_checks += 1
            _, _, _, _, emo, conf = results[0]
            self.face_counts[emo] += 1
            self.last_face_emotion = emo
            self.last_face_conf = conf
            self.last_face_probs = {emotions[k]: float(first_probs[k]) * 100.0 for k in range(len(emotions))}
            self.face_lbl["text"] = f"Face Emotion:   {emo} ({conf:.0f}%)"
        else:
            self.face_lbl["text"] = "Face Emotion:   No face detected"
        return results

    def _audio_callback(self, indata, frames, time_info, status):
        samples = indata[:, 0].copy()
        n = len(samples)
        rms = float(np.sqrt(np.mean(samples ** 2)))
        chunk_size = self.api["AUDIO_CHUNK_SIZE"]
        max_chunks = int(MAX_AUDIO_RECORD_SECONDS * self.api["SAMPLE_RATE"] / max(1, n))
        with self.buf_lock:
            if n >= chunk_size:
                self.audio_buf[:] = samples[-chunk_size:]
            else:
                self.audio_buf[:-n] = self.audio_buf[n:]
                self.audio_buf[-n:] = samples
            self.samples_received += n
            self.mic_level = min(100.0, rms * 1500.0)
            if len(self.recorded_chunks) < max_chunks:
                self.recorded_chunks.append(samples)

    def _live_voice_step(self):
        if not self.running:
            return
        if not self.audio_predicting:
            with self.buf_lock:
                ready = self.samples_received >= self.api["AUDIO_CHUNK_SIZE"]
                chunk = self.audio_buf.copy() if ready else None
            if chunk is not None:
                rms = float(np.sqrt(np.mean(chunk ** 2)))
                if rms >= 0.001:
                    self.audio_predicting = True
                    self.api["run_in_background"](
                        self,
                        lambda c=chunk: self.api["predict_voice_chunk"](c, self.audio_model),
                        self._on_voice_scores,
                        lambda _e: setattr(self, "audio_predicting", False),
                    )
        self.after(900, self._live_voice_step)

    def _on_voice_scores(self, scores):
        self.audio_predicting = False
        if not self.running:
            return
        top = max(scores, key=scores.get)
        self.voice_counts[top] += 1
        self.last_voice_emotion = top
        self.last_voice_conf = scores[top]
        self.last_voice_probs = scores
        self.voice_lbl["text"] = f"Voice Emotion:  {top} ({scores[top]:.0f}%)"

    def _tick_timeline(self):
        if not self.running:
            return

        with self.buf_lock:
            self.level_bar["value"] = self.mic_level

        elapsed = time.time() - self.start_time
        if self.duration_limit is not None:
            rem = max(0, int(round(self.duration_limit - elapsed)))
            self.clock_lbl["text"] = f"Left: {rem // 60:02d}:{rem % 60:02d}"
            if elapsed >= self.duration_limit:
                self.stop()
                return
        else:
            sec = int(elapsed)
            self.clock_lbl["text"] = f"Time: {sec // 60:02d}:{sec % 60:02d}"

        if not self.timeline or (elapsed - self.timeline[-1]["t"]) >= 0.95:
            pos_vals, stress_vals = [], []
            for probs in (self.last_face_probs, self.last_voice_probs):
                if probs:
                    pos_vals.append(probs.get("Happy", 0) + probs.get("Neutral", 0) + 0.5 * probs.get("Surprise", 0))
                    stress_vals.append(probs.get("Fear", 0) + probs.get("Angry", 0) + probs.get("Sad", 0) + probs.get("Disgust", 0))

            if pos_vals:
                f_grp = _sentiment_group(self.last_face_emotion)
                v_grp = _sentiment_group(self.last_voice_emotion)
                aligned = (f_grp == v_grp) if (f_grp and v_grp) else None

                if aligned is True:
                    self.sync_lbl["text"] = "Face–Voice Sync: Aligned (consistent tone)"
                elif aligned is False:
                    self.sync_lbl["text"] = (
                        f"Face–Voice Sync: Mismatch ({self.last_face_emotion} vs {self.last_voice_emotion})"
                    )

                self.timeline.append({
                    "t": round(elapsed, 1),
                    "face_emotion": self.last_face_emotion,
                    "face_conf": round(self.last_face_conf, 1),
                    "voice_emotion": self.last_voice_emotion,
                    "voice_conf": round(self.last_voice_conf, 1),
                    "positivity": round(float(np.mean(pos_vals)), 1),
                    "stress": round(float(np.mean(stress_vals)), 1),
                    "aligned": aligned,
                })
                self.timeline_canvas.set_data(self.timeline)
                self._update_scorecard()

        self.after(200, self._tick_timeline)

    def _update_scorecard(self):
        metrics = _compute_metrics(
            self.timeline, self.face_present_checks, self.face_total_checks,
            self.face_counts, self.voice_counts, self.trait_scores,
        )
        self.score_lbl["text"] = f"Readiness Score: {metrics['score']} / 100"
        self.subscores_lbl["text"] = (
            f"Composure: {metrics['composure']:.0f}%   |   "
            f"Presence: {metrics['presence']:.0f}%   |   "
            f"Sync: {metrics['alignment']:.0f}%"
        )
        self.tips_lbl["text"] = "\n".join(f"• {tip}" for tip in metrics["tips"])
        if self.last_session is not None:
            self.last_session["metrics"] = metrics
            self.last_session["trait_scores"] = dict(self.trait_scores)
            self.last_session["transcript"] = self.transcript_box.get("1.0", "end").strip()

    def _on_transcript_edit(self, _event=None):
        text = self.transcript_box.get("1.0", "end").strip()
        if text == self._last_text:
            return
        if self._text_after_id is not None:
            self.after_cancel(self._text_after_id)
            self._text_after_id = None
        if len(text.split()) < 3:
            self.trait_scores = {}
            self.trait_lbl["text"] = "Big-Five: —"
            if self.timeline:
                self._update_scorecard()
            return
        self._text_after_id = self.after(500, self._run_text_live)

    def _run_text_live(self):
        self._text_after_id = None
        text = self.transcript_box.get("1.0", "end").strip()
        if len(text.split()) < 3:
            return
        if self._text_busy:
            self._text_pending = True
            return
        self._text_busy = True
        self._text_pending = False
        self._last_text = text
        self.trait_lbl["text"] = "Big-Five: analysing..."
        self.api["run_in_background"](
            self,
            lambda t=text: self.api["predict_personality"](t),
            self._on_text_scores,
            self._on_text_error,
        )

    def _on_text_scores(self, output):
        scores, _n_words, _top = output
        self._text_busy = False
        self.trait_scores = scores
        top = max(scores, key=scores.get)
        ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:3]
        summary = " | ".join(f"{k[:5]}. {v:.0f}%" for k, v in ordered)
        self.trait_lbl["text"] = f"Big-Five: {top} ({scores[top]:.0f}%) — {summary}"
        if self.timeline or self.last_session:
            self._update_scorecard()
        if not self.running and self.last_session:
            self.status["text"] = "Interview & personality analysis ready — click 'Save Report'!"
        if self._text_pending:
            self._text_pending = False
            self._run_text_live()

    def _on_text_error(self, _err):
        self._text_busy = False
        self._text_pending = False
        self.trait_lbl["text"] = "Big-Five: error"

    def save_report(self):
        if not self.last_session and not self.timeline:
            messagebox.showinfo("Save Report", "Complete an interview session first.")
            return
        session = self.last_session or {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "question": self.question_var.get().strip() or "Practice Interview Question",
            "duration": max(1.0, self.timeline[-1]["t"]) if self.timeline else 0.0,
            "timeline": list(self.timeline),
            "face_counts": dict(self.face_counts),
            "voice_counts": dict(self.voice_counts),
            "trait_scores": dict(self.trait_scores),
            "transcript": self.transcript_box.get("1.0", "end").strip(),
            "metrics": _compute_metrics(
                self.timeline, self.face_present_checks, self.face_total_checks,
                self.face_counts, self.voice_counts, self.trait_scores,
            ),
        }
        session["transcript"] = self.transcript_box.get("1.0", "end").strip()
        session["trait_scores"] = dict(self.trait_scores)

        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        default_dir = os.path.join(self.api["BASE_DIR"], "reports")
        os.makedirs(default_dir, exist_ok=True)
        path = filedialog.asksaveasfilename(
            initialdir=default_dir,
            initialfile=f"interview_report_{stamp}.pdf",
            defaultextension=".pdf",
            filetypes=[("PDF Report", "*.pdf")],
        )
        if not path:
            return
        try:
            pdf_path, csv_path = export_interview_report(path, session, self.history)
            messagebox.showinfo(
                "Report Saved",
                f"Saved successfully!\n\nPDF Report:\n{pdf_path}\n\nTimeline CSV:\n{csv_path}",
            )
        except Exception as err:
            messagebox.showerror("Save Report", self.api["friendly_error"](err))
