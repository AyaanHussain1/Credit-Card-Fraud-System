# Credit Card Fraud Detection System

A machine learning web app that detects fraudulent credit card transactions in real time, built on a highly imbalanced real-world transaction dataset.

## Overview

Credit card fraud detection is a classic imbalanced-classification problem — in the raw dataset used here, only **419 out of 229,789 transactions (0.18%)** are fraudulent. Training a model directly on this would bias it heavily toward predicting "legitimate" every time. This project addresses that with **undersampling**, builds a balanced Logistic Regression classifier, and deploys it as an interactive Streamlit app that lets a user select a transaction, adjust the amount, and get an instant fraud/safe prediction with confidence percentages.

## How It Works

1. **Data** — Started with a real credit card transactions dataset (229,789 rows, 30 anonymized features `V1`–`V28` plus `Time` and `Amount`).
2. **Handling Class Imbalance** — Since fraud cases are extremely rare, applied **random undersampling**: kept all 419 fraud transactions and randomly sampled 401 legitimate transactions to build a balanced dataset (~820 total records).
3. **Model Training** — Trained a **Logistic Regression** classifier on the balanced dataset with an 80/20 train-test split (stratified).
4. **Result** — Achieved **93% accuracy** on both the training and test sets.
5. **Deployment** — Built a Streamlit app that loads the trained model and a sample transaction set, letting users pick a transaction, adjust the amount, and get a real-time prediction with fraud/safe probability breakdown.

## Tech Stack

- **Language:** Python
- **Data Handling:** Pandas, NumPy
- **Modeling:** Scikit-learn (Logistic Regression)
- **Visualization (EDA):** Matplotlib, Seaborn
- **App/Deployment:** Streamlit

## Project Structure

```
credit-card-fraud-detection/
├── main.py              # Data prep, undersampling, model training
├── app.py               # Streamlit web app for live predictions
├── model.pkl            # Trained Logistic Regression model
├── creditcard.xls        # Full raw dataset (229,789 transactions)
├── fraud.csv             # Isolated fraud-only transactions
├── new_data.xls           # Balanced sample dataset used by the app (802 transactions)
├── requirements.txt      # Python dependencies
└── .streamlit/            # Streamlit theme configuration
```

## Installation & Usage

```bash
# Clone the repository
git clone https://github.com/<your-username>/credit-card-fraud-detection.git
cd credit-card-fraud-detection

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## Results

| Metric              | Score |
|---------------------|-------|
| Training Accuracy   | 93%   |
| Test Accuracy       | 93%   |

## Disclaimer

This tool is built for educational and portfolio purposes only. It is trained on a limited, undersampled dataset and should not be used for real financial fraud decisions.
