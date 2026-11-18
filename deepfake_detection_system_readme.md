# 🛡️ Deepfake Detection System — Hybrid Spatial-Temporal Forensics

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-CPU_Optimized-EE4C2C?style=for-the-badge&logo=pytorch)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?style=for-the-badge&logo=fastapi)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer_Vision-5C3EE8?style=for-the-badge&logo=opencv)
![Three.js](https://img.shields.io/badge/Three.js-3D_WebGL_UI-black?style=for-the-badge&logo=three.js)

An end-to-end **AI-Powered Media Authenticity Verification System** developed as a Final Year Project at the **Department of Computer Science, University of Sindh**. The system detects synthetic GAN-generated portraits and neural face-swapped videos in real time using a lightweight, CPU-deployable **MobileNetV3-Small** architecture paired with **OpenCV** facial localization and an interactive **3D WebGL** cyber-forensics web interface.

---

## 👥 Project Team & Academic Supervision

| Role | Name | Affiliation |
| :--- | :--- | :--- |
| **Team Leader** | **Mehtab Ali** | Department of Computer Science, University of Sindh |
| **Co-Team Leader** | **Najeeb Ahmed** | Department of Computer Science, University of Sindh |
| **Team Member** | **Maryam Ali** | Department of Computer Science, University of Sindh |
| **Project Supervisor** | **Prof. Dr. Fida Hussain Chandio** | Department of Computer Science, University of Sindh |

---

## 🎯 Key Features & Highlights

* **Dual-Mode Media Forensics:** Supports both static image formats (`JPG`, `JPEG`, `PNG`) and video sequences (`MP4`, `AVI`, `MOV`).
* **Automated Facial ROI Localization:** Uses OpenCV's Haar-Cascade frontal face classifier to isolate the primary human face (`224×224` RGB tensor), eliminating background noise and preventing false triggers on non-face media or stylized artwork.
* **Temporal Keyframe Sampling:** Extracts up to **36 uniformly spaced keyframes** across uploaded videos and computes cross-frame probability variance to detect temporal face-swap flickering.
* **Lightweight CPU Inference:** Trained on cloud GPUs and compressed into a **~6 MB PyTorch state dictionary** (`best_deepfake_model.pth`), delivering sub-2-second inference on standard CPU laptops without requiring dedicated GPU hardware.
* **Interactive 3D Cyber-Forensics UI:** Built with Tailwind CSS and Three.js WebGL featuring a rotating 3D biometric neural core, particle field, laser scan overlays, and detailed forensic telemetry bars (*Facial Consistency*, *Temporal Coherence*, *Artifact Detection*, and *Blending Anomalies*).
* **Zero Data Retention:** Uploaded files are processed inside isolated volatile temporary buffers and automatically purged immediately after inference completes.

---

## 📊 Model Performance & Benchmark Evaluation

Evaluated on a strictly leak-free, held-out hybrid test set of **1,740 unseen images and video frames** (balanced 50% Real / 50% Fake):

| Metric | Score / Value |
| :--- | :--- |
| **Hybrid Test Accuracy** | **92.30%** |
| **ROC-AUC Score** | **0.9764** |
| **Fake Class (Precision / Recall / F1)** | `0.91` / `0.93` / `0.92` |
| **Real Class (Precision / Recall / F1)** | `0.93` / `0.91` / `0.92` |
| **Average CPU Inference Time** | `< 2.0 seconds` |

---

## 🧠 System Architecture & 5-Stage Pipeline

```text
[Uploaded Image/Video] 
        │
        ▼
1. Media Ingestion (FastAPI Async Multipart Handler)
        │
        ▼
2. Keyframe Extraction (OpenCV Linear Timeline Sampling — up to 36 frames)
        │
        ▼
3. Facial Localization (Haar-Cascade Bounding Box + 224×224 RGB Normalization)
        │
        ▼
4. Spatial-Temporal CNN Analysis (Fine-Tuned MobileNetV3-Small Backbone)
        │
        ▼
5. Forensic Verdict & Telemetry (Softmax Confidence + Detailed Sub-Metrics)
```

### Training Corpus (Balanced 50/50 Split)
* **140k Real and Fake Faces (Image Domain):** Combines real human photographs (FFHQ) with StyleGAN synthetic faces to learn high-frequency spatial upsampling artifacts, skin pore smoothing, and iris asymmetry.
* **Deep Fake Detection — DFD (Video Domain):** Combines original actor recordings with neural face-swapped sequences. Grouped strictly by `video_id` prior to splitting (`8,216` train / `1,768` validation / `1,740` test) to guarantee zero subject or frame leakage.

---

## 📁 Repository Structure

```text
deepfake-detector/
├── artifacts/
│   └── best_deepfake_model.pth   # Trained MobileNetV3-Small PyTorch weights (~6 MB)
├── static/
│   └── index.html                # Single-page 3D WebGL + Tailwind CSS frontend
├── inference.py                  # OpenCV face extraction & PyTorch inference engine
├── main.py                       # FastAPI REST backend & static file server
├── requirements.txt              # Python dependencies (CPU-optimized)
├── Dockerfile                    # Container configuration for cloud deployment
├── .gitignore                    # Excludes virtual environments and cache files
└── README.md                     # Project documentation
```

---

## 🚀 Local Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/NajeebAhmed69/deepfake-detection-system.git
cd deepfake-detection-system
```

### 2. Create and Activate a Virtual Environment
**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch the Application
```bash
uvicorn main:app --reload
```
Open your browser and navigate to **`http://127.0.0.1:8000`**.

---

## 🐳 Cloud Deployment (Docker / Hugging Face Spaces)

Because this application executes real-time PyTorch and OpenCV inference on the backend, it includes a production-ready `Dockerfile` configured for port `7860`:

```bash
docker build -t deepfake-detector .
docker run -p 7860:7860 deepfake-detector
```

---

## ⚠️ Operational Scope & Known Limitations

* **Photorealistic Human Faces Only:** Optimized for real camera photography and photorealistic deepfakes. Stylized cartoons, anime, and non-face images are automatically intercepted by the face-detection guard.
* **Lighting & Compression Sensitivity:** Extreme low-light grain, heavy social-media beauty filters, or steep side profiles may affect confidence scores.
* **Visual-Only Forensics:** Evaluates spatial and temporal visual artifacts without audio deepfake spectrogram analysis.

---

## 📜 License & Academic Acknowledgment

This project was developed for academic and research purposes in fulfillment of the **Bachelor of Science in Computer Science (Final Year Project 2026)** at the **University of Sindh, Jamshoro**, under the supervision of **Prof. Dr. Fida Hussain Chandio**.