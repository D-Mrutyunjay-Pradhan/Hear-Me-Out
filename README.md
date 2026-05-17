# 🤟 HEAR ME OUT
### Real-Time Sign Language to Text Conversion

> A hybrid deep learning system that converts hand gestures into readable text in real time — bridging communication between the deaf community and the general public.

![Python](https://img.shields.io/badge/Python-3.9.13-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10.1-orange?logo=tensorflow)
![Flask](https://img.shields.io/badge/Flask-2.2.5-lightgrey?logo=flask)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.5-green)
![OpenCV](https://img.shields.io/badge/OpenCV-4.7.0-red?logo=opencv)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📌 About the Project

**Hear Me Out** is a web-based assistive technology application developed as a B.Tech major project at **Gandhi Institute for Education & Technology, Bhubaneswar** (AY 2025–26).

The system captures live hand gestures through a webcam, processes them using a **Hybrid CNN + MediaPipe model**, and converts the recognized sign into on-screen text — enabling smooth communication between sign language users and those unfamiliar with it.

According to the WHO, over **1.5 billion people** live with some degree of hearing loss. This project aims to provide an affordable, hardware-free solution to reduce that communication gap.

---

## ✨ Features

- 🎥 **Real-time webcam gesture recognition** via browser
- 🤖 **Hybrid AI model** — MobileNetV2 (CNN) + MediaPipe hand landmarks
- 🔤 Recognizes **35 classes**: digits 1–9, letters A–Z, SPACE, DELETE
- 📝 **Sentence builder** — assembles letters into words as you sign
- 💡 **Visual feedback** — detected gesture shown with green highlight on acceptance
- 📷 **Camera toggle** — pause/resume camera with a switch
- 🗑️ **Clear sentence** — reset the output with one click
- ⚡ Runs fully on CPU — no GPU required for inference

---

## 🧠 How It Works

```
Webcam Frame
     ↓
Preprocessing (resize 160×160, normalize)
     ↓
MediaPipe Hand Detection → 21 Landmark Points (63 values)
     ↓
MobileNetV2 (CNN) Feature Extraction
     ↓
Concatenate CNN features + Landmark features
     ↓
Dense Layers → Softmax (35 classes)
     ↓
Gesture Label → Sentence Builder → Text Display
```

The model uses a **dual-input architecture**:
- **Image branch**: MobileNetV2 (pretrained on ImageNet) → Flatten → Dense(128)
- **Landmark branch**: Dense(64) on 63 normalized coordinates
- **Combined**: Concatenate → Dense(128) → Softmax(35)

---

## 🗂️ Project Structure

```
hear-me-out/
│
├── app.py                  # Flask web app — main entry point
├── Extra/
│   ├── config.py           # Paths, class labels, training params
│   ├── utils.py            # Image preprocessing & landmark normalization
│   ├── data_generator.py   # Keras Sequence data generator
│   ├── data_augmentation.py# Image augmentation pipeline
│   ├── callbacks.py        # EarlyStopping & ModelCheckpoint
│   └── train_test_split.py # Train/test split utility
│
├── train_module.py         # Build & train the hybrid CNN model
├── data_collection.py      # Collect gesture images + landmarks via webcam
├── evaluate_graph.py       # Evaluate model & plot confusion matrix
│
├── templates/
│   └── index.html          # Frontend UI
│
├── models/
│   └── all_sign_model.h5   # Trained model (download separately)
│
├── data/
│   ├── images/             # Gesture images (class-wise folders)
│   └── landmarks/          # Corresponding .npy landmark files
│
└── requirements.txt
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- Webcam
- ~2 GB disk space for dataset

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/hear-me-out.git
cd hear-me-out
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the Trained Model

Download `all_sign_model.h5` from the [Releases](https://github.com/your-username/hear-me-out/releases) page and place it in:

```
models/all_sign_model.h5
```

> **Note:** The model file is large (~100MB+) and is not included in this repository. Use Git LFS or the Releases section.

### 4. Run the App

```bash
python app.py
```

Open your browser and go to: **http://127.0.0.1:5000**

---

## 🏋️ Training From Scratch

### Step 1 — Collect Data

```bash
python data_collection.py
```

- Enter the label when prompted (e.g., `A`)
- Press **`S`** to save a frame
- Press **`Q`** to quit and move to the next label
- Collect ~1000 images per class

### Step 2 — Train the Model

```bash
python train_module.py
```

Training parameters (edit in `Extra/config.py`):

| Parameter | Default |
|-----------|---------|
| Image size | 160×160 |
| Batch size | 16 |
| Epochs | 10 |
| Validation split | 30% |

### Step 3 — Evaluate

```bash
python evaluate_graph.py
```

This will print accuracy and display a confusion matrix heatmap.

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| CNN Only Accuracy | 78% |
| MediaPipe Only Accuracy | 84% |
| **Hybrid Model Accuracy** | **92%** |
| Real-time FPS | ~25–30 FPS |
| Number of Classes | 35 |
| Total Training Images | ~37,000 |

---

## 🖥️ Dataset

The custom dataset was collected via webcam using `data_collection.py`.

| Property | Value |
|----------|-------|
| Total classes | 35 (1–9, A–Z, SPACE, DELETE) |
| Images per class | ~1,000 |
| Total images | ~37,000 |
| Image format | `.jpg` (160×160) |
| Landmark format | `.npy` (63 float values) |

Each class is stored in its own folder under `data/images/` and `data/landmarks/`.

> The full dataset (~37,000 images + .npy files) is not included due to size. See the [Releases](https://github.com/your-username/hear-me-out/releases) page or contact the authors.

---

## 🛠️ Tech Stack

| Technology | Purpose |
|------------|---------|
| Python 3.9 | Core language |
| TensorFlow / Keras | Model training & inference |
| MobileNetV2 | CNN backbone (transfer learning) |
| MediaPipe | Real-time hand landmark detection |
| OpenCV | Webcam capture & image processing |
| Flask | Web server & video streaming |
| NumPy | Numerical operations |
| scikit-learn | Data splitting & evaluation |

---

## 📋 Requirements

```
tensorflow==2.10.1
keras
opencv-python==4.7.0.72
mediapipe==0.10.5
flask==2.2.5
numpy==1.23.5
scikit-learn
matplotlib
seaborn
```

Install all with:
```bash
pip install -r requirements.txt
```

---

## 🎓 Project Info

| Field | Details |
|-------|---------|
| Project Title | Hear Me Out |
| Institution | Gandhi Institute for Education & Technology, Bhubaneswar |
| Department | Computer Science & Engineering |
| University | Biju Pattnaik University of Technology (BPUT), Odisha |
| Academic Year | 2025–26 |
| Guide | Prof. Ashutosh Katual |

### 👥 Team

| Name | Roll No. |
|------|----------|
| D Mrutyunjay Pradhan | 2321326048 |
| Pritam Mishra | 2321326053 |
| Biswajit Khandual | 2321326047 |
| Subham Kumar Sahoo | 2321326056 |
| Arpita Dhal | 2321326046 |

---

## ⚠️ Known Limitations

- Works best with a **single hand** in frame
- Recognizes **static gestures only** (no motion-based signs)
- Performance may drop in **poor lighting** or cluttered backgrounds
- Currently supports a **fixed vocabulary** of 35 classes (no full sentences)

---

## 🔮 Future Scope

- Multi-hand gesture support
- Dynamic / motion-based sign recognition (LSTM / Transformer)
- Sentence-level NLP correction
- Text-to-speech output
- Mobile app (Android/iOS)
- Larger, more diverse dataset

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- [Google MediaPipe](https://github.com/google-ai-edge/mediapipe) for hand tracking
- [TensorFlow / Keras](https://www.tensorflow.org/) for the deep learning framework
- [MobileNetV2](https://arxiv.org/abs/1801.04381) pretrained on ImageNet

---

<p align="center">
  Made with ❤️ at GIET Bhubaneswar &nbsp;|&nbsp; AY 2025–26
</p>
