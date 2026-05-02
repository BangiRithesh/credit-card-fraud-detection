# Credit Card Fraud Detection

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A comprehensive machine learning solution for detecting fraudulent credit card transactions using advanced preprocessing techniques (SMOTE, RobustScaler) and multiple classifiers (Logistic Regression, Random Forest, XGBoost, Isolation Forest). This project demonstrates a complete pipeline from data exploration to business impact analysis.

## 📋 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Results](#results)
- [Business Impact](#business-impact)
- [Key Findings](#key-findings)
- [Future Work](#future-work)
- [References](#references)
- [License](#license)

## 📌 Overview
Credit card fraud causes billions of dollars in losses annually. This project implements and compares multiple machine learning models to detect fraudulent transactions with high recall and precision, minimizing false positives while catching the majority of frauds.

**Best Model**: Random Forest with SMOTE → **95.5% fraud detection rate**, **90% reduction in financial losses**.

## 📊 Dataset
The dataset used is the **Credit Card Fraud Detection** dataset from Kaggle ([link](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)). It contains 284,807 transactions over two days, with only 492 frauds (0.172%).

- **Features**: 28 PCA-transformed features (V1–V28) + `Time` and `Amount`.
- **Target**: `Class` (0 = genuine, 1 = fraud).
- **Class imbalance**: Highly skewed (fraud rate ~0.17%).

## ✨ Features
- **Exploratory Data Analysis (EDA)**: Class distribution, amount analysis, temporal patterns, correlation matrices.
- **Data Preprocessing**:
  - Handling missing values & duplicates.
  - RobustScaler for `Amount` and `Time` (handles outliers).
  - SMOTE (Synthetic Minority Over-sampling Technique) to balance classes.
- **Model Training & Tuning**:
  - Logistic Regression
  - Random Forest (hyperparameter tuning via RandomizedSearchCV)
  - XGBoost
  - Isolation Forest (unsupervised anomaly detection)
- **Evaluation**:
  - Precision, Recall, F1-Score, ROC-AUC, Average Precision.
  - Confusion matrices, ROC curves, Precision-Recall curves.
  - Feature importance analysis.
  - Threshold optimization.
- **Business Impact Analysis**: Cost savings estimation based on fraud amounts and investigation costs.

## 🛠 Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Clone the repository
```bash
git clone https://github.com/yourusername/credit-card-fraud-detection.git
cd credit-card-fraud-detection
