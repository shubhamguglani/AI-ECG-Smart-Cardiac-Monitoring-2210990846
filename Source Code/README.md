# Artificial Intelligence for Real-Time ECG Analysis and Smart Cardiac Monitoring

## 📋 Team Details

| Field              | Details                                      |
|--------------------|----------------------------------------------|
| **Student Name**   | Shubham Guglani                              |
| **Roll Number**    | 2210990846                                   |
| **Project Title**  | Artificial Intelligence for Real-Time ECG Analysis and Smart Cardiac Monitoring |
| **Type**           | Research Paper                               |
| **Department**     | Computer Science and Engineering             |
| **University**     | Chitkara University, Punjab, India           |
| **Submitted To**   | Dr. Preeti Saini, Assistant Professor, CSE  |
| **Course**         | CO-OP Project at Industry (Module-2) (22CS421) |
| **Current Status** | ✅ Research Paper Submitted — Under Review at ICMREST-2026 |

---

## 📁 Repository Structure

```
AI-ECG-Smart-Cardiac-Monitoring-2210990846/
│
├── 📂 IPR Submission Proof/
│   └── ICMREST_2026_Submission_Proof.png     ← Screenshot of paper submission confirmation
│
├── 📂 Report and PPT/
│   ├── COOP_Report_Shubham_Guglani.docx      ← Full CO-OP Project Report
│   └── ECG_Research_Presentation.pptx        ← Research Presentation Slides
│
├── 📂 Source Code/
│   ├── train.py                               ← Main training script
│   ├── requirements.txt                       ← Python dependencies
│   ├── data/
│   │   └── dataset_loader.py                  ← MIT-BIH & synthetic data loader
│   ├── models/
│   │   ├── cnn_bilstm.py                      ← Hybrid CNN-BiLSTM model
│   │   └── compression.py                     ← INT8 Quantization & Weight Pruning
│   ├── utils/
│   │   └── preprocessing.py                   ← ECG signal preprocessing pipeline
│   └── api/
│       └── inference_api.py                   ← FastAPI real-time inference server
│
└── README.md                                  ← This file
```

---

## 🔬 Project Abstract

Traditional ECG monitoring relies on manual or rule-based interpretation, which frequently misses intermittent and early-stage arrhythmias. This project implements a **Hybrid CNN-BiLSTM deep learning framework** for real-time ECG arrhythmia classification, deployed through a cloud-native microservices architecture.

**Key Results:**
- 🎯 **98.22%** accuracy — Hybrid CNN-BiLSTM on MIT-BIH Arrhythmia Database
- ⚡ **22ms** median inference latency via FastAPI + Triton
- 📦 **75% memory reduction** via INT8 quantization for wearable deployment
- 🚀 **780 req/sec** throughput under 1,000 concurrent patient streams

---

## 🧠 AI Models Implemented

| Model              | Accuracy | Time Complexity | Best For              |
|--------------------|----------|------------------|-----------------------|
| 1D CNN             | 90.93%   | O(n)             | Edge / Wearable       |
| Hybrid CNN-BiLSTM  | 98.22%   | O(n)             | Cloud / Clinical      |
| CNN-Transformer    | 99.71%   | O(n²)            | High-accuracy cloud   |
| SNN (Event-Driven) | 98.26%   | O(spikes)        | Ultra-low power edge  |

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r Source\ Code/requirements.txt
```

### 2. Train on synthetic data (demo — no dataset needed)
```bash
cd "Source Code"
python train.py --data synthetic --epochs 30
```

### 3. Train on MIT-BIH Arrhythmia Database
```bash
python train.py --data mitbih --data_path ./data/mitbih --epochs 50
```

### 4. Run the real-time inference API
```bash
uvicorn api.inference_api:app --host 0.0.0.0 --port 8000 --reload
```

### 5. Test the API
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"signal": [0.1, 0.3, 0.8, 1.2, 0.5, 0.2, 0.1], "sampling_rate": 360}'
```

### 6. Optional: Apply compression for edge deployment
```bash
python train.py --data synthetic --prune --quantize
```

---

## 🏗️ System Architecture

```
Wearables / IoT Devices
        ↓ (MQTT / WebSockets)
   FastAPI Gateway
        ↓
   Apache Kafka (Message Broker)
        ↓
   Preprocessing Pipeline
   (Filter → Normalize → Segment)
        ↓
   CNN-BiLSTM Inference
   (NVIDIA Triton Server)
        ↓
   Clinician Dashboard + EHR (FHIR)
```

---

## 📊 Dataset Information

| Dataset                  | Records | Description                          |
|--------------------------|---------|--------------------------------------|
| MIT-BIH Arrhythmia DB    | 48      | 2-channel ECG, 360 Hz, 47 subjects   |
| PTB-XL                   | 21,837  | 12-lead ECG, 18,885 patients          |
| Synthetic (Time-VQVAE)   | 50,000  | Generated for class balancing        |

---

## 🔧 Technologies Used

- **Python 3.10+**, TensorFlow 2.18, PyTorch
- **Signal Processing**: SciPy, BioSPPy, Py-ECGDetectors
- **API**: FastAPI, Uvicorn, Pydantic
- **Infrastructure**: Docker, Kubernetes, Apache Kafka
- **Model Serving**: NVIDIA Triton Inference Server

---

## 📜 IPR Status

- **Conference**: 5th International Conference on Multidisciplinary Research in Education, Science and Technology **(ICMREST-2026)**
- **Submission Status**: ✅ **Under Review**
- **Paper Type**: Research Paper
- **Submission Proof**: See `/IPR Submission Proof/` folder

---

## 📞 Contact

**Shubham Guglani**  
Roll No: 2210990846  
B.E. Computer Science and Engineering  
Chitkara University, Punjab, India  
📧 Shubham0846.be22@chitkara.edu.in
