#!/usr/bin/env python3

import importlib.util
import os
import pickle
import re
import string
import subprocess
import sys
import threading

def ensure_tensorflow_python(script_path=None):
    if importlib.util.find_spec("tensorflow") is not None:
        return
    if os.environ.get("_MER_PYTHON_CHECKED"):
        return

    current = os.path.realpath(sys.executable)
    target_script = os.path.abspath(script_path or __file__)
    candidates = [
        "/Library/Frameworks/Python.framework/Versions/3.12/bin/python3",
        "/Library/Frameworks/Python.framework/Versions/3.11/bin/python3",
        "/Library/Frameworks/Python.framework/Versions/3.10/bin/python3",
    ]
    for py in candidates:
        if not os.path.isfile(py) or os.path.realpath(py) == current:
            continue
        check = subprocess.run(
            [py, "-c", "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('tensorflow') else 1)"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if check.returncode == 0:
            print(f"TensorFlow not found in {sys.executable}; switching to {py}...", flush=True)
            os.environ["_MER_PYTHON_CHECKED"] = "1"
            os.execv(py, [py, target_script, *sys.argv[1:]])

if __name__ == "__main__":
    ensure_tensorflow_python(__file__)

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext

import numpy as np
import cv2
from PIL import Image, ImageTk

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

sys.modules.setdefault("pyarrow", None)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "Models")
TMP_DIR = os.path.join(BASE_DIR, "tmp")
os.makedirs(TMP_DIR, exist_ok=True)

FACE_EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
VOICE_EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
TRAITS = ["Extraversion", "Neuroticism", "Agreeableness", "Conscientiousness", "Openness"]

SAMPLE_RATE = 16000
AUDIO_CHUNK_SIZE = 49100
AUDIO_UPDATE_MS = 800
TEXT_MAX_LEN = 300
TEXT_DEBOUNCE_MS = 450
VIDEO_SIZE = (480, 340)

_models = {}
_models_lock = threading.Lock()

def get_model(name):
    loaders = {"video": load_video_model, "audio": load_audio_model, "text": load_text_model}
    with _models_lock:
        if name not in _models:
            _models[name] = loaders[name]()
        return _models[name]

def load_video_model():
    import json
    import h5py
    import tensorflow as tf
    L = tf.keras.layers

    path = os.path.join(MODEL_DIR, "video.h5")
    with h5py.File(path, "r") as f:
        config = f.attrs["model_config"]
    config = json.loads(config.decode() if isinstance(config, bytes) else config)["config"]

    keep = {"name", "filters", "kernel_size", "strides", "padding", "activation", "use_bias",
            "axis", "epsilon", "momentum", "center", "scale", "pool_size", "units",
            "depth_multiplier", "dilation_rate"}
    tensors = {}
    for layer in config["layers"]:
        kind, cfg, name = layer["class_name"], layer["config"], layer["name"]
        if kind == "InputLayer":
            tensors[name] = L.Input(shape=tuple(cfg["batch_input_shape"][1:]), name=name)
            continue
        kwargs = {k: v for k, v in cfg.items() if k in keep}
        if isinstance(kwargs.get("axis"), list):
            kwargs["axis"] = kwargs["axis"][0]
        inputs = [tensors[node[0]] for node in layer["inbound_nodes"][0]]
        tensors[name] = getattr(L, kind)(**kwargs)(inputs if len(inputs) > 1 else inputs[0])

    model = tf.keras.Model(tensors[config["input_layers"][0][0]],
                           tensors[config["output_layers"][0][0]])
    model.load_weights(path, by_name=True)
    return model

def load_audio_model():
    import tensorflow as tf
    L = tf.keras.layers

    inputs = L.Input(shape=(5, 128, 128, 1))
    x = inputs
    for filters, pool in [(64, 2), (64, 4), (128, 4), (128, 4)]:
        x = L.TimeDistributed(L.Conv2D(filters, (3, 3), padding="same"))(x)
        x = L.TimeDistributed(L.BatchNormalization())(x)
        x = L.TimeDistributed(L.Activation("elu"))(x)
        x = L.TimeDistributed(L.MaxPooling2D((pool, pool), strides=(pool, pool), padding="same"))(x)
        x = L.TimeDistributed(L.Dropout(0.2))(x)
    x = L.TimeDistributed(L.Flatten())(x)
    x = L.LSTM(256, dropout=0.2)(x)
    outputs = L.Dense(7, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    model.load_weights(os.path.join(MODEL_DIR, "audio.hdf5"))
    return model

def load_text_model():
    import tensorflow as tf
    L = tf.keras.layers

    def old_hard_sigmoid(x):
        return tf.clip_by_value(0.2 * x + 0.5, 0.0, 1.0)

    inputs = L.Input(shape=(TEXT_MAX_LEN,))
    x = L.Embedding(22347, 300)(inputs)
    for filters in (128, 256, 384):
        x = L.Conv1D(filters, 8, padding="same", activation="relu")(x)
        x = L.MaxPooling1D(2)(x)
        x = L.SpatialDropout1D(0.3)(x)
        x = L.BatchNormalization()(x)
    x = L.LSTM(180, return_sequences=True, recurrent_activation=old_hard_sigmoid)(x)
    x = L.LSTM(180, return_sequences=True, recurrent_activation=old_hard_sigmoid)(x)
    x = L.LSTM(180, recurrent_activation=old_hard_sigmoid)(x)
    x = L.Dense(128, activation="softmax")(x)
    outputs = L.Dense(5, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    model.load_weights(os.path.join(MODEL_DIR, "Personality_traits_NN.h5"))
    return model, TextPreprocessor()

def mel_spectrogram(y):
    import librosa
    spec = np.abs(librosa.stft(y, n_fft=512, window="hamming", win_length=256, hop_length=128)) ** 2
    mel = librosa.feature.melspectrogram(S=spec, sr=SAMPLE_RATE, n_mels=128, fmax=4000)
    return librosa.power_to_db(mel, ref=np.max)

def predict_voice_chunk(samples, model=None):
    from scipy.stats import zscore

    if model is None:
        model = get_model("audio")
    norm = np.nan_to_num(zscore(samples)).astype(np.float32)
    mel = mel_spectrogram(norm)
    X = np.array([[mel[:, t * 64: t * 64 + 128] for t in range(5)]])[..., np.newaxis]
    probs = np.asarray(model.predict_on_batch(X))[0]
    return {VOICE_EMOTIONS[i]: float(p) * 100 for i, p in enumerate(probs)}

def predict_voice_emotion(wav_path):
    import librosa
    from scipy.stats import zscore

    model = get_model("audio")

    y, _ = librosa.load(wav_path, sr=SAMPLE_RATE, offset=0.5)
    chunk_step = SAMPLE_RATE
    if len(y) < AUDIO_CHUNK_SIZE:
        raise ValueError("The recording is too short. Please use at least 4 seconds of audio.")
    n_chunks = 1 + (len(y) - AUDIO_CHUNK_SIZE) // chunk_step
    chunks = [y[i * chunk_step: i * chunk_step + AUDIO_CHUNK_SIZE] for i in range(n_chunks)]

    mels = [mel_spectrogram(np.nan_to_num(zscore(c)).astype(np.float32)) for c in chunks]

    X = np.array([[m[:, t * 64: t * 64 + 128] for t in range(5)] for m in mels])
    X = X[..., np.newaxis]

    probs = np.asarray(model.predict_on_batch(X)).mean(axis=0)
    return {VOICE_EMOTIONS[i]: float(p) * 100 for i, p in enumerate(probs)}

NLTK_PACKAGES = {
    "corpora/stopwords": "stopwords",
    "corpora/wordnet": "wordnet",
    "corpora/omw-1.4": "omw-1.4",
    "tokenizers/punkt": "punkt",
    "tokenizers/punkt_tab": "punkt_tab",
    "taggers/averaged_perceptron_tagger": "averaged_perceptron_tagger",
    "taggers/averaged_perceptron_tagger_eng": "averaged_perceptron_tagger_eng",
}

def ensure_nltk_data():
    import nltk
    for resource, package in NLTK_PACKAGES.items():
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download(package, quiet=True)

class _TokenizerData:
    pass

class _TokenizerUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("keras"):
            return _TokenizerData
        return super().find_class(module, name)

BASIC_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "as", "at", "be", "because", "been", "before", "being", "below",
    "between", "both", "but", "by", "can", "did", "do", "does", "doing", "don",
    "down", "during", "each", "few", "for", "from", "further", "had", "has", "have",
    "having", "he", "her", "here", "hers", "herself", "him", "himself", "his", "how",
    "i", "if", "in", "into", "is", "it", "its", "itself", "just", "me", "more",
    "most", "my", "myself", "no", "nor", "not", "now", "of", "off", "on", "once",
    "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own", "s",
    "same", "she", "should", "so", "some", "such", "t", "than", "that", "the",
    "their", "theirs", "them", "themselves", "then", "there", "these", "they",
    "this", "those", "through", "to", "too", "under", "until", "up", "very", "was",
    "we", "were", "what", "when", "where", "which", "while", "who", "whom", "why",
    "will", "with", "you", "your", "yours", "yourself", "yourselves",
}

class TextPreprocessor:

    def __init__(self):
        self.punct = set(string.punctuation)
        self.use_nltk = False
        try:
            ensure_nltk_data()
            from nltk.corpus import stopwords
            from nltk.stem import WordNetLemmatizer

            self.stopwords = set(stopwords.words("english"))
            self.lemmatizer = WordNetLemmatizer()
            self.use_nltk = True
        except Exception:
            self.stopwords = BASIC_STOPWORDS
            self.lemmatizer = None

        with open(os.path.join(MODEL_DIR, "padding.pickle"), "rb") as f:
            self.tokenizer = _TokenizerUnpickler(f).load()

    def clean(self, text):
        text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
        for pattern, repl in [(r"what's", "what is "), (r"\'s", " "), (r"\'ve", " have "),
                              (r"can't", "cannot "), (r"n't", " not "), (r"i'm", "i am "),
                              (r"\'re", " are "), (r"\'d", " would "), (r"\'ll", " will "),
                              (r"(\d+)(k)", r"\g<1>000")]:
            text = re.sub(pattern, repl, text)

        if self.use_nltk:
            from nltk import sent_tokenize, wordpunct_tokenize, pos_tag
            from nltk.corpus import wordnet as wn

            pos_map = {"N": wn.NOUN, "V": wn.VERB, "R": wn.ADV, "J": wn.ADJ}
            words = []
            for sentence in sent_tokenize(text):
                for token, tag in pos_tag(wordpunct_tokenize(sentence)):
                    token = token.lower().strip().strip("_").strip("*")
                    if token in self.stopwords or all(ch in self.punct for ch in token):
                        continue
                    words.append(self.lemmatizer.lemmatize(token, pos_map.get(tag[0], wn.NOUN)))
            return words

        tokens = re.findall(r"[a-z0-9]+", text.lower())
        return [w for w in tokens if w not in self.stopwords]

    def to_ids(self, words):
        t = self.tokenizer
        text = " ".join(words)
        if t.lower:
            text = text.lower()
        text = text.translate(str.maketrans({c: t.split for c in t.filters}))
        ids = []
        for w in text.split(t.split):
            i = t.word_index.get(w) if w else None
            if i is not None and (not t.num_words or i < t.num_words):
                ids.append(i)
        ids = ids[-TEXT_MAX_LEN:]
        return np.array([[0] * (TEXT_MAX_LEN - len(ids)) + ids], dtype=np.int32)

def predict_personality(text):
    model, prep = get_model("text")
    words = prep.clean(text)
    probs = np.asarray(model.predict_on_batch(prep.to_ids(words)))[0]

    counts = {}
    for w in words:
        counts[w] = counts.get(w, 0) + 1
    top_words = sorted(counts, key=counts.get, reverse=True)[:10]

    return {TRAITS[i]: float(p) * 100 for i, p in enumerate(probs)}, len(text.split()), top_words

def friendly_error(err):
    if isinstance(err, ModuleNotFoundError):
        pip_names = {"cv2": "opencv-python", "PIL": "pillow", "sklearn": "scikit-learn"}
        pkg = pip_names.get(err.name, err.name)
        return f"Missing Python package '{err.name}'.\n\nInstall it with:\n    pip install {pkg}"
    return f"{type(err).__name__}: {err}"

def run_in_background(widget, work, on_done, on_error):
    result = {}

    def target():
        try:
            result["value"] = work()
        except Exception as e:
            result["error"] = e

    thread = threading.Thread(target=target, daemon=True)
    thread.start()

    def check():
        if thread.is_alive():
            widget.after(100, check)
        elif "error" in result:
            on_error(result["error"])
        else:
            on_done(result["value"])

    widget.after(100, check)

class ResultBars(ttk.Frame):

    def __init__(self, parent, labels):
        super().__init__(parent)
        self.rows = {}
        for r, name in enumerate(labels):
            ttk.Label(self, text=name, width=17).grid(row=r, column=0, sticky="w", pady=3)
            bar = ttk.Progressbar(self, length=300, maximum=100)
            bar.grid(row=r, column=1, padx=8)
            value = ttk.Label(self, text="0 %", width=7)
            value.grid(row=r, column=2, sticky="e")
            self.rows[name] = (bar, value)

    def show(self, values):
        for name, (bar, label) in self.rows.items():
            v = values.get(name, 0)
            bar["value"] = v
            label["text"] = f"{v:.0f} %"

def make_face_detector():
    if hasattr(cv2, "CascadeClassifier"):
        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        if not cascade.empty():
            def detect_haar(gray):
                return cascade.detectMultiScale(
                    gray, scaleFactor=1.2, minNeighbors=4, minSize=(36, 36))
            return detect_haar

    try:
        import dlib
        detector = dlib.get_frontal_face_detector()

        def detect_dlib(gray):
            h_img, w_img = gray.shape[:2]
            boxes = []
            for r in detector(gray, 0):
                x1, y1 = max(0, r.left()), max(0, r.top())
                x2, y2 = min(w_img, r.right()), min(h_img, r.bottom())
                if x2 > x1 and y2 > y1:
                    boxes.append((x1, y1, x2 - x1, y2 - y1))
            return boxes

        return detect_dlib
    except Exception:
        return None

class VideoTab(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent, padding=10)
        self.cap = None
        self.model = None
        self.running = False
        self.frame_no = 0
        self.faces = []
        self.counts = {e: 0 for e in FACE_EMOTIONS}
        self.face_finder = make_face_detector()

        ttk.Label(self, text="Facial emotion recognition (webcam)",
                  font=("Helvetica", 16, "bold")).pack(pady=(0, 8))

        self.blank = ImageTk.PhotoImage(Image.new("RGB", VIDEO_SIZE, "black"))
        self.video_label = tk.Label(self, image=self.blank, bg="black")
        self.video_label.pack()

        buttons = ttk.Frame(self)
        buttons.pack(pady=8)
        self.start_btn = ttk.Button(buttons, text="Start camera", command=self.start)
        self.start_btn.pack(side="left", padx=5)
        ttk.Button(buttons, text="Stop camera", command=self.stop).pack(side="left", padx=5)

        self.status = ttk.Label(self, text="Press 'Start camera'.", font=("Helvetica", 13))
        self.status.pack()

        ttk.Label(self, text="Emotions seen during this session:").pack(pady=(8, 2))
        self.bars = ResultBars(self, FACE_EMOTIONS)
        self.bars.pack()

    def start(self):
        if self.running:
            return
        if self.face_finder is None:
            self._failed(RuntimeError("No face detector available. Install opencv-python<5 or dlib."))
            return
        self.start_btn.state(["disabled"])
        self.status["text"] = "Loading model... (first time can take a few seconds)"
        run_in_background(self, lambda: get_model("video"), self._open_camera, self._failed)

    def _open_camera(self, model):
        self.model = model
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self._failed(RuntimeError("Could not open the webcam. "
                                      "Check camera permission for your Terminal / IDE."))
            return
        self.running = True
        self.frame_no = 0
        self.faces = []
        self.counts = {e: 0 for e in FACE_EMOTIONS}
        self.bars.show({})
        self.status["text"] = "Looking for faces..."
        self._next_frame()

    def _failed(self, err):
        self.start_btn.state(["!disabled"])
        self.status["text"] = "Error."
        messagebox.showerror("Video", friendly_error(err))

    def stop(self):
        self.running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.start_btn.state(["!disabled"])
        self.video_label.configure(image=self.blank)
        self.video_label.image = self.blank
        total = sum(self.counts.values())
        if total:
            top = max(self.counts, key=self.counts.get)
            self.status["text"] = f"Stopped. Your main emotion was: {top}"
        else:
            self.status["text"] = "Stopped."

    def _next_frame(self):
        if not self.running:
            return
        ok, frame = self.cap.read()
        if ok:
            frame = cv2.resize(cv2.flip(frame, 1), VIDEO_SIZE)
            self.frame_no += 1
            if self.frame_no % 5 == 0:
                self.faces = self.find_emotions(frame)

            for (x, y, w, h, emotion, conf) in self.faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"{emotion} {conf:.0f}%", (x, max(25, y - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            image = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            self.video_label.configure(image=image)
            self.video_label.image = image
        self.after(15, self._next_frame)

    def find_emotions(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        boxes = self.face_finder(gray)
        results = []
        for (x, y, w, h) in boxes:
            crop = gray[y:y + h, x:x + w]
            if crop.size == 0:
                continue
            face = cv2.resize(crop, (48, 48)).astype(np.float32) / 255.0
            probs = np.asarray(self.model.predict_on_batch(face.reshape(1, 48, 48, 1)))[0]
            i = int(np.argmax(probs))
            results.append((x, y, w, h, FACE_EMOTIONS[i], probs[i] * 100))
            self.counts[FACE_EMOTIONS[i]] += 1

        if results:
            self.status["text"] = f"Current emotion: {results[0][4]}"
            total = sum(self.counts.values())
            self.bars.show({e: 100 * c / total for e, c in self.counts.items()})
        else:
            self.status["text"] = "No face found"
        return results

class AudioTab(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent, padding=10)
        self.stream = None
        self.model = None
        self.running = False
        self._starting = False
        self.predicting = False
        self.buf_lock = threading.Lock()
        self.audio_buf = np.zeros(AUDIO_CHUNK_SIZE, dtype=np.float32)
        self.samples_received = 0
        self.mic_level = 0.0
        self.counts = {e: 0 for e in VOICE_EMOTIONS}

        ttk.Label(self, text="Speech emotion recognition (live mic)",
                  font=("Helvetica", 16, "bold")).pack(pady=(0, 8))
        ttk.Label(self, text="Press 'Start mic' and speak naturally — emotions update live.",
                  wraplength=600).pack(pady=(0, 10))

        buttons = ttk.Frame(self)
        buttons.pack(pady=5)
        self.start_btn = ttk.Button(buttons, text="Start mic", command=self.start)
        self.start_btn.pack(side="left", padx=5)
        self.stop_btn = ttk.Button(buttons, text="Stop mic", command=self.stop)
        self.stop_btn.pack(side="left", padx=5)
        self.open_btn = ttk.Button(buttons, text="Open .wav file", command=self.open_file)
        self.open_btn.pack(side="left", padx=5)

        level_row = ttk.Frame(self)
        level_row.pack(pady=(10, 4))
        ttk.Label(level_row, text="Mic level:").pack(side="left", padx=(0, 8))
        self.level_bar = ttk.Progressbar(level_row, length=260, maximum=100)
        self.level_bar.pack(side="left")

        self.status = ttk.Label(self, text="Press 'Start mic' and start speaking.",
                                font=("Helvetica", 13))
        self.status.pack(pady=8)

        self.bars = ResultBars(self, VOICE_EMOTIONS)
        self.bars.pack(pady=5)

        self.result = ttk.Label(self, text="", font=("Helvetica", 15, "bold"))
        self.result.pack(pady=10)

    def start(self):
        if self.running or self._starting:
            return
        self._starting = True
        self.start_btn.state(["disabled"])
        self.open_btn.state(["disabled"])
        self.result["text"] = ""
        self.status["text"] = "Loading audio model... (first time can take a few seconds)"
        run_in_background(self, lambda: get_model("audio"), self._open_mic, self._failed)

    def _open_mic(self, model):
        if not self._starting:
            return
        self._starting = False
        try:
            import sounddevice as sd
            self.model = model
            with self.buf_lock:
                self.audio_buf.fill(0)
                self.samples_received = 0
                self.mic_level = 0.0
            self.counts = {e: 0 for e in VOICE_EMOTIONS}
            self.bars.show({})
            self.stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                blocksize=1600,
                callback=self._audio_callback,
            )
            self.stream.start()
        except Exception as err:
            self._failed(err)
            return

        self.running = True
        self.predicting = False
        self.status["text"] = "Listening... speak now!"
        self._update_level()
        self._live_step()

    def _audio_callback(self, indata, frames, time_info, status):
        samples = indata[:, 0]
        n = len(samples)
        rms = float(np.sqrt(np.mean(samples ** 2)))
        with self.buf_lock:
            if n >= AUDIO_CHUNK_SIZE:
                self.audio_buf[:] = samples[-AUDIO_CHUNK_SIZE:]
            else:
                self.audio_buf[:-n] = self.audio_buf[n:]
                self.audio_buf[-n:] = samples
            self.samples_received += n
            self.mic_level = min(100.0, rms * 1500.0)

    def _update_level(self):
        if not self.running:
            self.level_bar["value"] = 0
            return
        with self.buf_lock:
            level = self.mic_level
            received = self.samples_received
        self.level_bar["value"] = level
        if received < AUDIO_CHUNK_SIZE:
            secs = received / SAMPLE_RATE
            total_secs = AUDIO_CHUNK_SIZE / SAMPLE_RATE
            self.status["text"] = f"Listening... filling audio buffer ({secs:.1f} / {total_secs:.1f} s)"
        self.after(100, self._update_level)

    def _live_step(self):
        if not self.running:
            return
        if not self.predicting:
            with self.buf_lock:
                ready = self.samples_received >= AUDIO_CHUNK_SIZE
                chunk = self.audio_buf.copy() if ready else None
            if chunk is not None:
                rms = float(np.sqrt(np.mean(chunk ** 2)))
                if rms < 0.001:
                    self.status["text"] = "Listening... (speak into the mic)"
                else:
                    self.predicting = True
                    run_in_background(
                        self,
                        lambda c=chunk: predict_voice_chunk(c, self.model),
                        self._show_live,
                        self._live_error,
                    )
        self.after(AUDIO_UPDATE_MS, self._live_step)

    def _show_live(self, scores):
        self.predicting = False
        if not self.running:
            return
        self.bars.show(scores)
        top = max(scores, key=scores.get)
        self.counts[top] += 1
        self.status["text"] = f"Live — Current emotion: {top} ({scores[top]:.0f}%)"
        self.result["text"] = f"Current emotion: {top} ({scores[top]:.0f}%)"

    def _live_error(self, err):
        self.predicting = False
        if self.running:
            self.stop()
            messagebox.showerror("Audio", friendly_error(err))

    def stop(self):
        was_running = self.running
        self.running = False
        self._starting = False
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None
        self.level_bar["value"] = 0
        self.start_btn.state(["!disabled"])
        self.open_btn.state(["!disabled"])
        if was_running:
            total = sum(self.counts.values())
            if total:
                top = max(self.counts, key=self.counts.get)
                self.status["text"] = f"Stopped. Your main emotion was: {top}"
                self.result["text"] = f"Main emotion: {top}"
            else:
                self.status["text"] = "Stopped."

    def open_file(self):
        if self.running:
            self.stop()
        path = filedialog.askopenfilename(filetypes=[("WAV audio", "*.wav")])
        if path:
            self.start_btn.state(["disabled"])
            self.open_btn.state(["disabled"])
            self.result["text"] = ""
            self.status["text"] = "Analysing audio file..."
            run_in_background(self, lambda: predict_voice_emotion(path), self._show_file, self._failed)

    def _show_file(self, scores):
        self.start_btn.state(["!disabled"])
        self.open_btn.state(["!disabled"])
        self.bars.show(scores)
        top = max(scores, key=scores.get)
        self.status["text"] = "Done."
        self.result["text"] = f"Main emotion: {top} ({scores[top]:.0f}%)"

    def _failed(self, err):
        self.running = False
        self._starting = False
        self.start_btn.state(["!disabled"])
        self.open_btn.state(["!disabled"])
        self.status["text"] = "Error."
        messagebox.showerror("Audio", friendly_error(err))

class TextTab(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent, padding=10)
        self._after_id = None
        self._busy = False
        self._pending = False
        self._last_text = ""

        ttk.Label(self, text="Personality traits from text (live)",
                  font=("Helvetica", 16, "bold")).pack(pady=(0, 8))
        ttk.Label(self, text="Start typing below — results update live as you write:").pack()

        self.textbox = scrolledtext.ScrolledText(self, width=75, height=10, wrap="word",
                                                 font=("Helvetica", 13))
        self.textbox.pack(pady=5)
        self.textbox.bind("<KeyRelease>", self._on_text_change)
        self.textbox.bind("<<Paste>>", lambda _e: self.after_idle(self._on_text_change))
        self.textbox.bind("<<Cut>>", lambda _e: self.after_idle(self._on_text_change))

        buttons = ttk.Frame(self)
        buttons.pack(pady=5)
        self.analyse_btn = ttk.Button(buttons, text="Analyse now", command=self.analyse)
        self.analyse_btn.pack(side="left", padx=5)
        ttk.Button(buttons, text="Load .txt file", command=self.load_file).pack(side="left", padx=5)
        ttk.Button(buttons, text="Clear", command=self.clear).pack(side="left", padx=5)

        self.status = ttk.Label(self, text="Start typing (at least 3 words) for live results.",
                                font=("Helvetica", 13))
        self.status.pack(pady=5)

        self.bars = ResultBars(self, TRAITS)
        self.bars.pack(pady=5)

        self.result = ttk.Label(self, text="", font=("Helvetica", 15, "bold"))
        self.result.pack(pady=5)
        self.words = ttk.Label(self, text="", wraplength=600)
        self.words.pack()

    def load_file(self):
        path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
        if path:
            with open(path, encoding="utf-8", errors="ignore") as f:
                self.textbox.delete("1.0", "end")
                self.textbox.insert("1.0", f.read())
            self._last_text = ""
            self._run_live()

    def clear(self):
        if self._after_id is not None:
            self.after_cancel(self._after_id)
            self._after_id = None
        self._pending = False
        self._last_text = ""
        self.textbox.delete("1.0", "end")
        self.bars.show({})
        self.status["text"] = "Start typing (at least 3 words) for live results."
        self.result["text"] = ""
        self.words["text"] = ""

    def _on_text_change(self, _event=None):
        text = self.textbox.get("1.0", "end").strip()
        if text == self._last_text:
            return
        if self._after_id is not None:
            self.after_cancel(self._after_id)
            self._after_id = None
        if not text:
            self._last_text = ""
            self.bars.show({})
            self.result["text"] = ""
            self.words["text"] = ""
            self.status["text"] = "Start typing (at least 3 words) for live results."
            return
        n_words = len(text.split())
        if n_words < 3:
            self.status["text"] = f"Words: {n_words} — keep typing for live analysis..."
            return
        self._after_id = self.after(TEXT_DEBOUNCE_MS, self._run_live)

    def analyse(self):
        text = self.textbox.get("1.0", "end").strip()
        if len(text.split()) < 3:
            messagebox.showinfo("Text", "Please write at least 3 words.")
            return
        self._last_text = ""
        self._run_live()

    def _run_live(self):
        self._after_id = None
        text = self.textbox.get("1.0", "end").strip()
        if len(text.split()) < 3:
            return
        if self._busy:
            self._pending = True
            return
        self._busy = True
        self._pending = False
        self._last_text = text
        if "text" not in _models:
            self.status["text"] = "Loading text model... (first time can take a few seconds)"
        else:
            self.status["text"] = f"Updating live... ({len(text.split())} words)"
        run_in_background(self, lambda t=text: predict_personality(t), self._show, self._failed)

    def _show(self, output):
        scores, n_words, top_words = output
        self._busy = False
        if not self.textbox.get("1.0", "end").strip():
            return
        self.bars.show(scores)
        top = max(scores, key=scores.get)
        self.status["text"] = f"Live — {n_words} words"
        self.result["text"] = f"Main trait: {top} ({scores[top]:.0f}%)"
        self.words["text"] = ("Most used words: " + ", ".join(top_words)) if top_words else ""
        if self._pending:
            self._pending = False
            self._run_live()

    def _failed(self, err):
        self._busy = False
        self._pending = False
        self.status["text"] = "Error."
        messagebox.showerror("Text", friendly_error(err))

def _activate_macos_app():
    if sys.platform != "darwin":
        return
    try:
        import ctypes
        import ctypes.util

        objc = ctypes.cdll.LoadLibrary(ctypes.util.find_library("objc"))
        ctypes.cdll.LoadLibrary("/System/Library/Frameworks/AppKit.framework/AppKit")
        objc.objc_getClass.restype = ctypes.c_void_p
        objc.objc_getClass.argtypes = [ctypes.c_char_p]
        objc.sel_registerName.restype = ctypes.c_void_p
        objc.sel_registerName.argtypes = [ctypes.c_char_p]

        msg0 = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)(
            ("objc_msgSend", objc))
        msg_opt = ctypes.CFUNCTYPE(
            ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong)(
            ("objc_msgSend", objc))

        cls = objc.objc_getClass(b"NSRunningApplication")
        current = msg0(cls, objc.sel_registerName(b"currentApplication"))
        msg_opt(current, objc.sel_registerName(b"activateWithOptions:"), 3)
    except Exception:
        pass

def main():
    if BASE_DIR not in sys.path:
        sys.path.insert(0, BASE_DIR)
    from interview_coach import InterviewTab

    root = tk.Tk()
    root.title("Multimodal Emotion Recognition")
    root.geometry("900x740")
    root.minsize(820, 680)

    ttk.Label(root, text="Multimodal Emotion Recognition",
              font=("Helvetica", 20, "bold")).pack(pady=10)

    tabs = ttk.Notebook(root)
    video_tab = VideoTab(tabs)
    audio_tab = AudioTab(tabs)
    text_tab = TextTab(tabs)

    coach_api = {
        "BASE_DIR": BASE_DIR,
        "TMP_DIR": TMP_DIR,
        "FACE_EMOTIONS": FACE_EMOTIONS,
        "VOICE_EMOTIONS": VOICE_EMOTIONS,
        "SAMPLE_RATE": SAMPLE_RATE,
        "AUDIO_CHUNK_SIZE": AUDIO_CHUNK_SIZE,
        "get_model": get_model,
        "make_face_detector": make_face_detector,
        "predict_voice_chunk": predict_voice_chunk,
        "predict_personality": predict_personality,
        "run_in_background": run_in_background,
        "friendly_error": friendly_error,
    }
    interview_tab = InterviewTab(
        tabs,
        coach_api,
        stop_other_tabs=lambda: (video_tab.stop(), audio_tab.stop()),
    )

    tabs.add(video_tab, text="  Video  ")
    tabs.add(audio_tab, text="  Audio  ")
    tabs.add(text_tab, text="  Text  ")
    tabs.add(interview_tab, text="  Interview Coach  ")
    tabs.pack(fill="both", expand=True, padx=10, pady=(0, 10))

    def on_close():
        video_tab.stop()
        audio_tab.stop()
        interview_tab.stop()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)

    def bring_to_front():
        if not root.winfo_exists():
            return
        try:
            root.deiconify()
            root.lift()
            root.attributes("-topmost", True)
            root.update()
            _activate_macos_app()
            if root.winfo_exists():
                root.focus_force()
                root.after(400, lambda: root.winfo_exists() and root.attributes("-topmost", False))
        except tk.TclError:
            pass

    root.after(100, bring_to_front)

    print("App window is open (close the window or press Ctrl+C to quit).", flush=True)
    root.mainloop()

if __name__ == "__main__":
    main()
