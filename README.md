# 💳 Credit Card Fraud Detection
### Machine Learning-Powered Anomaly Detection System

> **Data Science Project — Sem VI | 2026**
> Team: U23CS138 · U23CS110 · U23CS126 · U23CS146 · U23CS122

---

## 📌 Overview

Credit card fraud costs the global economy over **$32 billion annually**. This project builds a complete, scalable ML pipeline to detect fraudulent transactions in real time — using a highly imbalanced dataset where fraud accounts for only **0.22%** of all transactions (a 444:1 class ratio).

We trained and evaluated **4 models** — Logistic Regression, Random Forest, XGBoost, and Isolation Forest — and achieved a **93.2% fraud recall** with Random Forest after SMOTE balancing and threshold optimization.

---

## 📂 Repository Structure

```
credit-card-fraud-detection/
│
├── finaldraft.ipynb          # Main Jupyter notebook (full pipeline)
├── README.md                 # This file
│
├── data/
│   └── creditcard.csv        # Dataset (download separately — see below)
│
└── outputs/
    ├── class_distribution.png
    ├── amount_analysis.png
    ├── time_analysis.png
    ├── correlation_matrix.png
    ├── class_correlations.png
    ├── scaling_comparison.png
    ├── smote_comparison.png
    ├── confusion_matrices.png
    ├── roc_curves.png
    ├── precision_recall_curves.png
    ├── model_comparison.png
    ├── feature_importance.png
    ├── threshold_analysis.png
    └── business_impact.png
```

---

## 📊 Dataset

**Source:** [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

| Attribute | Value |
|-----------|-------|
| Total Records | 98,743 (after cleaning) |
| Features | 31 (V1–V28 via PCA + Time, Amount, Class) |
| Fraudulent Transactions | 222 |
| Fraud Rate | 0.22% |
| Imbalance Ratio | 1 : 444 |
| Train / Test Split | 80 / 20 stratified |

> ⚠️ The dataset is **not included** in this repo due to size. Download `creditcard.csv` from the Kaggle link above and place it in the `data/` folder before running the notebook.

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```

### 2. Install dependencies
```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost imbalanced-learn joblib
```

Or with pip requirements:
```bash
pip install -r requirements.txt
```

### 3. Add the dataset
Download `creditcard.csv` from Kaggle and place it in:
```
data/creditcard.csv
```

### 4. Run the notebook
```bash
jupyter notebook finaldraft.ipynb
```

---

## 🔬 Methodology

### Pipeline

```
Raw Data → Clean & EDA → Scale → SMOTE → Train Models → Evaluate → Best Model
```

### Steps

| Step | What We Did |
|------|-------------|
| **Data Cleaning** | Dropped 9 rows with NaN values; removed 377 duplicate rows |
| **EDA** | Analyzed class distribution, amount patterns, time patterns, feature correlations |
| **Feature Scaling** | Applied `RobustScaler` to `Amount` and `Time` — handles outliers better than `StandardScaler` |
| **Train-Test Split** | 80/20 stratified split → 78,994 train / 19,749 test |
| **SMOTE** | Raised fraud samples from 178 → 78,816 (balanced 1:1 ratio) |
| **Model Training** | Logistic Regression, Random Forest (tuned), XGBoost (tuned), Isolation Forest |
| **Threshold Tuning** | Swept thresholds 0.1–0.9; optimal found at **0.75** for Random Forest |

---

## 📈 Results

| Model | Precision | Recall | F1-Score | ROC-AUC |
|-------|-----------|--------|----------|---------|
| Logistic Regression | 0.0757 | 0.9545 | 0.1402 | 0.9711 |
| **Random Forest ★** | **0.5190** | **0.9318** | **0.6667** | **0.9774** |
| XGBoost | 0.3596 | 0.9318 | 0.5190 | 0.9708 |
| Isolation Forest | 0.1186 | 0.6364 | 0.2000 | N/A |

### ✅ Best Model — Random Forest with SMOTE

- **93.2% fraud recall** — 41 out of 44 fraud cases caught in the test set
- **F1-Score 0.8478** after setting optimal threshold to 0.75
- **86% estimated reduction** in fraud losses
- **Top predictive feature:** V14 (consistent across both RF and XGBoost)

---

## 🔑 Key Techniques

- **SMOTE** (Synthetic Minority Oversampling Technique) for class imbalance
- **RobustScaler** for outlier-resistant feature normalization
- **RandomizedSearchCV** for hyperparameter tuning on a 50K sample (computationally efficient)
- **Threshold Optimization** to maximize F1-Score beyond default 0.5 cutoff
- **Feature Importance** analysis across Random Forest and XGBoost
- **Parallel Processing** via `joblib` for scalable batch inference

---

## 📉 Limitations

- V1–V28 are PCA-transformed and anonymised — limits domain-specific feature engineering
- SMOTE generates synthetic samples that may not reflect real fraud patterns perfectly
- Full `GridSearchCV` was computationally prohibitive; fast `RandomizedSearchCV` used instead
- Recall vs precision tradeoff is highly sensitive to the classification threshold

---

## 🚀 Future Work

- **Ensemble Model** — Stack RF + XGBoost in a voting ensemble for improved precision
- **Deep Learning** — LSTM or Autoencoder for sequential transaction pattern detection
- **Concept Drift** — Online learning / periodic retraining to adapt to evolving fraud strategies
- **Real-time Pipeline** — Deploy via Kafka + FastAPI for sub-millisecond streaming inference

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-green)
![imbalanced-learn](https://img.shields.io/badge/imbalanced--learn-0.10+-red)
![Pandas](https://img.shields.io/badge/Pandas-1.5+-lightblue)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.6+-purple)

---

## 📄 License

This project is for academic purposes. Dataset is © ULB Machine Learning Group — see Kaggle for terms.
