# 🌿📈 Deep Learning Project: CNN + BiLSTM

> **Two deep learning models trained on real Kaggle datasets**
> CNN for Plant Disease Image Classification · BiLSTM for Stock Market Direction Prediction

---

## 📌 Project Overview

This project implements and evaluates **two deep learning architectures** on publicly available Kaggle datasets, covering both image classification and time-series forecasting domains.

| # | Model | Dataset | Task | Classes |
|---|-------|---------|------|---------|
| 1 | **CNN** | [PlantVillage Disease Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease) | Multi-class Image Classification | 15 |
| 2 | **BiLSTM** | [Stock Market Dataset](https://www.kaggle.com/datasets/jacksoncrow/stock-market-dataset) | Binary Time-Series Classification | 2 (UP/DOWN) |

---

## 📂 Repository Structure

```
dl-cnn-bilstm-project/
│
├── deep_learning_project.py        # ← Main Python script (all tasks)
├── README.md                       # ← This file
├── requirements.txt                # ← Python dependencies
│
└── dl_project_outputs/             # ← Generated plots & results
    ├── cnn_01_sample_images.png
    ├── cnn_02_class_distribution.png
    ├── cnn_03_pixel_distribution.png
    ├── cnn_04_training_history.png
    ├── cnn_05_confusion_matrix.png
    ├── cnn_06_roc_curves.png
    ├── cnn_07_per_class_metrics.png
    ├── bilstm_01_data_exploration.png
    ├── bilstm_02_correlation_heatmap.png
    ├── bilstm_03_sample_sequences.png
    ├── bilstm_04_training_history.png
    ├── bilstm_05_confusion_matrix.png
    ├── bilstm_06_roc_curve.png
    ├── bilstm_07_prediction_vs_actual.png
    ├── bilstm_08_pred_probability.png
    └── comparison_dashboard.png
```

---

## ⚙️ Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/dl-cnn-bilstm-project.git
cd dl-cnn-bilstm-project
```

### 2. Create a Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Datasets from Kaggle
```bash
pip install kagglehub
```
Add this to the top of `deep_learning_project.py`:
```python
import kagglehub

# CNN Dataset — PlantVillage
cnn_path = kagglehub.dataset_download("emmarex/plantdisease")

# BiLSTM Dataset — Stock Market
bilstm_path = kagglehub.dataset_download("jacksoncrow/stock-market-dataset")
```

### 5. Run the Project
```bash
python deep_learning_project.py
```

---

## 📦 Requirements

```
tensorflow>=2.10.0
numpy>=1.23.0
pandas>=1.5.0
matplotlib>=3.6.0
seaborn>=0.12.0
scikit-learn>=1.2.0
kagglehub
```

> Full list in `requirements.txt`

---

## 🧠 Model 1 — CNN (Plant Disease Classification)

### Dataset
- **Name:** PlantVillage Plant Disease Dataset
- **Link:** https://www.kaggle.com/datasets/emmarex/plantdisease
- **Samples:** ~54,306 RGB leaf images (224×224)
- **Classes:** 15 (e.g., Tomato Late Blight, Potato Early Blight, Pepper Healthy…)

### Architecture

```
Input (64×64×3)
    │
    ▼
┌─────────────────────────────────┐
│  Conv2D(32) → BN → ReLU        │  Block 1
│  Conv2D(32) → BN → ReLU        │
│  MaxPool(2×2) → Dropout(0.25)  │
└────────────────┬────────────────┘
                 ▼
┌─────────────────────────────────┐
│  Conv2D(64) → BN → ReLU        │  Block 2
│  Conv2D(64) → BN → ReLU        │
│  MaxPool(2×2) → Dropout(0.25)  │
└────────────────┬────────────────┘
                 ▼
┌─────────────────────────────────┐
│  Conv2D(128) → BN → ReLU       │  Block 3
│  Conv2D(128) → BN → ReLU       │
│  MaxPool(2×2) → Dropout(0.30)  │
└────────────────┬────────────────┘
                 ▼
      GlobalAveragePooling2D
                 ▼
       Dense(256) → BN → Dropout(0.40)
                 ▼
       Dense(128) → Dropout(0.30)
                 ▼
       Dense(15, activation='softmax')
```

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Batch Size | 32 |
| Epochs | 25 (Early Stopping) |
| Learning Rate | 0.001 |
| Optimizer | Adam |
| Loss | Categorical Cross-Entropy |
| Regularization | BatchNorm + Dropout (0.25–0.40) |
| Total Parameters | 357,679 |

### Preprocessing
- Pixel normalization to [0, 1]
- Train / Val / Test split: **70% / 15% / 15%**
- No augmentation in baseline (see Future Work)

---

## 📈 Model 2 — BiLSTM (Stock Market Direction Prediction)

### Dataset
- **Name:** Stock Market Dataset (OHLCV)
- **Link:** https://www.kaggle.com/datasets/jacksoncrow/stock-market-dataset
- **Samples:** ~3,000+ daily OHLCV observations per ticker
- **Target:** Binary — will next-day close price go **UP (1)** or **DOWN (0)**?

### Feature Engineering

| Feature | Description |
|---------|-------------|
| Open, High, Low, Close | Raw OHLCV prices |
| Volume | Daily trading volume |
| Return | Daily percentage return |
| MA5, MA20 | 5-day and 20-day moving averages |
| STD20 | 20-day rolling standard deviation |
| RSI | Relative Strength Index |
| MACD | Moving Average Convergence Divergence |
| Volume_MA | 5-day average volume |

### Architecture

```
Input (30 timesteps × 12 features)
          │
          ▼
  BiLSTM(128 units, return_sequences=True)
  BatchNormalization
          │
          ▼
  BiLSTM(64 units, return_sequences=True)
  BatchNormalization
          │
          ▼
  BiLSTM(32 units, return_sequences=False)
  BatchNormalization
          │
          ▼
  Dense(64, ReLU) → Dropout(0.35)
          │
          ▼
  Dense(32, ReLU) → Dropout(0.25)
          │
          ▼
  Dense(1, Sigmoid)  →  P(UP)
```

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Sequence Length | 30 days |
| Batch Size | 64 |
| Epochs | 30 (Early Stopping) |
| Learning Rate | 0.001 |
| Optimizer | Adam |
| Loss | Binary Cross-Entropy |
| Regularization | Dropout + Recurrent Dropout (0.10–0.35) |
| Total Parameters | 358,017 |

### Preprocessing
- MinMaxScaler → [0, 1] for all 12 features
- Sliding window sequences of 30 days
- Train / Val / Test split: **70% / 15% / 15%** (chronological, no shuffling)

---

## 📊 Results & Evaluation

### Performance Metrics

| Model | Accuracy | Precision | Recall | F1-Score | AUC |
|-------|----------|-----------|--------|----------|-----|
| **CNN** | See output | Weighted avg | Weighted avg | Weighted avg | Mean OvR |
| **BiLSTM** | See output | Binary | Binary | Binary | Binary ROC |

> Run the script to get exact numbers — metrics are printed to console and all plots are saved to `dl_project_outputs/`.

### Generated Plots

#### CNN Outputs
| Plot | Description |
|------|-------------|
| `cnn_01_sample_images.png` | Sample images from all 15 disease classes |
| `cnn_02_class_distribution.png` | Class balance bar chart |
| `cnn_03_pixel_distribution.png` | RGB channel pixel intensity histograms |
| `cnn_04_training_history.png` | Train vs. Validation Loss & Accuracy curves |
| `cnn_05_confusion_matrix.png` | 15×15 confusion matrix heatmap |
| `cnn_06_roc_curves.png` | Per-class ROC curves (One-vs-Rest) with AUC |
| `cnn_07_per_class_metrics.png` | Precision / Recall / F1 bar chart per class |

#### BiLSTM Outputs
| Plot | Description |
|------|-------------|
| `bilstm_01_data_exploration.png` | Price, returns, and volume time series |
| `bilstm_02_correlation_heatmap.png` | Feature correlation heatmap |
| `bilstm_03_sample_sequences.png` | Sample normalised 30-day input sequences |
| `bilstm_04_training_history.png` | Train vs. Validation Loss & Accuracy |
| `bilstm_05_confusion_matrix.png` | 2×2 confusion matrix (UP / DOWN) |
| `bilstm_06_roc_curve.png` | ROC curve with AUC score |
| `bilstm_07_prediction_vs_actual.png` | Predicted vs actual direction (test set) |
| `bilstm_08_pred_probability.png` | Prediction probability distribution by true class |

#### Comparative
| Plot | Description |
|------|-------------|
| `comparison_dashboard.png` | Side-by-side bar chart + radar chart: CNN vs BiLSTM |

---

## 🔮 Future Work & Improvements

### CNN — Plant Disease
1. **Transfer Learning** — Fine-tune ResNet50 / EfficientNetB4 pretrained on ImageNet for ~98%+ accuracy
2. **Data Augmentation** — Random flips, rotations, colour jitter, Mixup, CutOut, RandAugment
3. **Attention Mechanisms** — CBAM or Squeeze-Excitation blocks to highlight lesion regions
4. **Vision Transformers (ViT)** — Replace CNN backbone for global spatial context modelling
5. **Grad-CAM** — Add visualisation of which leaf regions the model focuses on

### BiLSTM — Stock Market
1. **Temporal Attention** — Bahdanau attention over time steps to focus on key market events
2. **Temporal Fusion Transformer (TFT)** — State-of-the-art multi-horizon time series model
3. **Sentiment Features** — Integrate NLP-parsed Twitter/Reddit/news sentiment as auxiliary input
4. **Ensemble Models** — Combine BiLSTM + XGBoost + CNN-1D for superior accuracy
5. **Walk-Forward Validation** — Simulate live deployment to eliminate look-ahead bias
6. **Multi-stock training** — Train on hundreds of tickers simultaneously for generalisability

---

## 📖 Task Mapping (Assignment)

| Task | Description | Status |
|------|-------------|--------|
| Task 1 | Dataset Identification & Justification | ✅ |
| Task 2 | Data Understanding & Preprocessing | ✅ |
| Task 3 | Model Architecture Design & Implementation | ✅ |
| Task 4 | Performance Evaluation Metrics & Plots | ✅ |
| Task 5 | Conclusion & Future Work | ✅ |

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10+-orange?logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-API-red?logo=keras)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2+-green?logo=scikitlearn)
![Pandas](https://img.shields.io/badge/Pandas-1.5+-purple?logo=pandas)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.6+-blue)
![Seaborn](https://img.shields.io/badge/Seaborn-0.12+-teal)
