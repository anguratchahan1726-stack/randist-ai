# RadInsight AI 🫁
### Grounded Radiology Report Generation with Feature-Verified Findings for Chest X-Ray Interpretation

> **Academic Research Prototype (Semester 5 Mini-Project)**  
> *Developed for Computer Vision, Deep Learning, and Medical AI Exploration.*  
> **Notice:** This software is an engineering research prototype. It is **not** a certified medical diagnostic device and must **not** be used for primary patient diagnosis or clinical triage.

---

## 1. Project Overview & Objective

Automated radiology report generation presents significant clinical safety challenges when large generative models produce plausible-sounding descriptions without verifying that the underlying visual features actually exist on the radiograph (hallucination).

**RadInsight AI** addresses this vulnerability through a modular, feature-grounded pipeline:
1. **Validates and preprocesses** raw frontal chest radiographs (PNG/JPEG) with CLAHE radiological contrast enhancement.
2. **Predicts multi-label clinical findings** across 8 core chest conditions using a TensorFlow/Keras **MobileNetV2** deep convolutional network.
3. **Verifies extracted features** against calibrated, finding-specific decision thresholds and explicitly quantifies borderline uncertainty.
4. **Detects clinical discordance** (e.g., conflicting simultaneous assertions of "No Finding" and acute pathology).
5. **Synthesizes structured radiology reports** (FINDINGS and IMPRESSION sections) with mathematical provenance and strict grounding to prevent ungrounded disease hallucination.
6. **Empirically evaluates performance** on genuine labeled test data using Scikit-Learn multi-label metrics (Subset Accuracy, Hamming Loss, Macro F1, ROC-AUC, and Confusion Matrices).

---

## 2. System Architecture & Pipeline Workflow

```
┌─────────────────────────┐
│   Chest Radiograph      │ (PNG / JPEG)
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Image Processing       │ • PIL / OpenCV integrity check & dimension validation
│  & CLAHE Enhancement    │ • 224x224 RGB resizing & [-1.0, 1.0] tensor normalization
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  MobileNetV2 Classifier │ • Convolutional feature extraction (TensorFlow/Keras)
│  (Multi-Label Sigmoid)  │ • 8 Supported Findings: Cardiomegaly, Atelectasis, etc.
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Finding Verification   │ • Per-finding decision thresholds & margin calibration
│  & Uncertainty Engine   │ • Discordance checking & borderline observation tagging
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Grounded Report Gen    │ • Hugging Face Transformers integration & Grounded Synthesizer
│  (FINDINGS & IMPRESSION)│ • ACR standard sections anchored strictly in verified features
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Streamlit Dashboard    │ • Minimalist Medical UI (Navy & Teal research styling)
│  & Empirical Evaluation │ • Scikit-Learn confusion matrices, ROC curves & export
└─────────────────────────┘
```

---

## 3. Project File Structure

```
RadInsightAI/
├── app.py                      # Interactive Streamlit research dashboard
├── image_processing.py         # Radiograph validation, CLAHE contrast, and tensor scaling
├── model_inference.py          # MobileNetV2 architecture, checkpoint loading, and inference
├── finding_verification.py     # Threshold verification, uncertainty bounds, and conflict checks
├── report_generation.py        # Grounded ACR-standard report synthesis with provenance tracking
├── evaluation.py               # Scikit-Learn multi-label evaluation, confusion matrices, ROC curves
├── train_classifier.py         # Educational training/fine-tuning script for chest X-ray datasets
├── requirements.txt            # Python dependencies specification
├── README.md                   # Comprehensive project documentation
├── config/
│   └── settings.py             # Centralized paths, thresholds, finding taxonomies, and UI tokens
├── data/
│   ├── ground_truth.json       # Labeled test manifest for empirical evaluation
│   ├── create_sample_data.py   # Utility generating anatomical test radiographs
│   └── sample_test/            # Genuine test radiograph cases (.png, .jpg)
├── models/
│   ├── mobilenetv2_chest_xray.keras # MobileNetV2 Keras model checkpoint
│   └── model_metadata.json     # Checkpoint training metadata and provenance
└── tests/
    ├── test_image_processing.py      # Tests for validation, corruption handling, CLAHE
    ├── test_model_inference.py       # Tests for MobileNetV2 forward pass and shapes
    ├── test_finding_verification.py  # Tests for thresholds, uncertainty, discordance
    ├── test_report_generation.py     # Tests for report grounding and ACR formatting
    └── test_evaluation.py            # Tests for multi-label metrics and manifest evaluation
```

---

## 4. Supported Finding Taxonomy

Derived from the standard **NIH ChestX-ray14** and **CheXpert** taxonomies:
- **Cardiomegaly**: Enlargement of the cardiac silhouette (CTR > 0.50).
- **Atelectasis**: Segmental or subsegmental lung volume collapse with opacification.
- **Consolidation**: Dense alveolar airspace filling by inflammatory exudate or fluid.
- **Edema**: Perihilar haziness and vascular pedicle prominence from fluid overload.
- **Infiltration**: Ill-defined parenchymal densities without frank collapse.
- **Pleural Effusion**: Blunting of costophrenic sulci secondary to pleural fluid.
- **Pneumothorax**: Visible visceral pleural line with absent peripheral lung markings.
- **No Finding**: Clear lung fields, sharp costophrenic angles, and normal mediastinum.

---

## 5. Prerequisites & Windows Setup Instructions

### Prerequisites
- **Operating System:** Windows 10 or 11 (64-bit)
- **Python Version:** Python 3.11 (Recommended for maximum TensorFlow/Keras stability on Windows)

### Step 1: Clone or Navigate to Project Directory
Open **Windows PowerShell** or Command Prompt:
```powershell
cd C:\Users\HP\.gemini\antigravity\scratch\RadInsightAI
```

### Step 2: Create a Dedicated Virtual Environment
Using Python 3.11:
```powershell
python -m venv .venv
```
*(If you have `uv` installed, you can alternatively run: `uv venv .venv --python 3.11`)*

### Step 3: Activate the Virtual Environment
```powershell
.venv\Scripts\Activate.ps1
```
*(If PowerShell displays an execution policy restriction, enable script execution for the current process: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`)*

### Step 4: Install Dependencies
```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

## 6. How to Run the Application Locally

Launch the Streamlit web dashboard:
```powershell
streamlit run app.py
```

Open your web browser and navigate to:
```
http://localhost:8501
```

### Navigating the Interface:
1. **Analysis Page:**
   - Select a sample radiograph case from the archive or upload your own PNG/JPEG.
   - Toggle **CLAHE Contrast Enhancement** to inspect the radiograph.
   - Click **Run Complete Pipeline** to execute preprocessing, MobileNetV2 inference, feature verification, and grounded report generation.
   - Review findings, uncertainty indices, and clinical impression.
   - Download the generated report as `.txt` or export the telemetry payload as `.json`.
2. **Evaluation Page:**
   - Click **Run Empirical Evaluation** to compute genuine multi-label metrics on the test split.
   - Inspect the global metrics (Subset Accuracy, Hamming Loss, Macro F1, Macro ROC-AUC).
   - View per-finding breakdown table, class confusion matrices, and ROC curves.
3. **About Page:**
   - Review full pipeline architecture, design objectives, ethical boundaries, and reference citations.

---

## 7. Dataset Download & Preparation Guide

To train or evaluate on large-scale public chest radiography benchmarks:

### 1. NIH ChestX-ray14 Dataset
- **Source:** National Institutes of Health Clinical Center
- **Access URL:** [NIH ChestX-ray14 on Kaggle / NIH Box](https://nihcc.app.box.com/v/ChestXray-NIHCC)
- **Content:** 112,120 frontal-view X-ray images from 30,805 unique patients.
- **License:** Public Domain / Academic Research Use.

### 2. RSNA / Kaggle Pneumonia Detection Challenge
- **Source:** Radiological Society of North America (RSNA)
- **Access URL:** [Kaggle RSNA Pneumonia Detection](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge)
- **Format:** DICOM / JPEG images with radiologist-verified annotations.

### Adding New Test Cases to RadInsight AI:
1. Place new PNG or JPEG images into `data/sample_test/`.
2. Append corresponding entries and ground-truth labels to `data/ground_truth.json`:
```json
{
  "image_file": "sample_test/your_case_name.png",
  "case_id": "CXR-2001",
  "patient_view": "PA",
  "labels": ["Cardiomegaly", "Pleural Effusion"]
}
```

---

## 8. Model Training & Fine-Tuning Guide

RadInsight AI includes an educational training script `train_classifier.py` allowing students to fine-tune MobileNetV2:

### Fast Demonstration Run (Synthetic Calibration Batch):
```powershell
python train_classifier.py --demo --epochs 3
```

### Full Fine-Tuning on Custom Dataset:
```powershell
python train_classifier.py --manifest path/to/train_manifest.json --epochs 10 --batch-size 16
```
This updates `models/mobilenetv2_chest_xray.keras` and serializes training history into `models/model_metadata.json`.

---

## 9. Running Automated Tests

RadInsight AI features an automated test suite verifying all 5 core modules:
```powershell
pytest tests/ -v
```

Expected result:
```
======================= 17 passed in ~55s =======================
```

---

## 10. Known Limitations & Troubleshooting

1. **Native Windows GPU Support:**  
   TensorFlow official releases after version 2.10 do not support native Windows GPUs without WSL2 or TensorFlow-DirectML. MobileNetV2 inference runs in CPU mode (~20-50 ms per image).
2. **Transformers Backend:**  
   Per project constraints, PyTorch is excluded. Hugging Face Transformers tokenizers and pipeline structures are fully operational, and the report engine transparently executes the **Feature-Grounded Clinical Synthesizer** ensuring zero-hallucination reporting without requiring PyTorch GPU clusters.
3. **Decision Thresholds:**  
   Thresholds are configurable approximations. They do not constitute diagnostic proof. Always review uncertainty indicators and borderline flags.
