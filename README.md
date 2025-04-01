# loan-amount-prediction-ML
Predicting loan amounts using traditional ML and deep learning models based on financial attributes.


# 📊 Loan Amount Prediction Using Financial Features

This repository contains a machine learning project focused on predicting loan amounts based on both numerical and categorical financial data. Various traditional machine learning models and deep learning architectures were implemented and compared.

---

## 🔍 Project Overview

The objective is to build accurate and efficient models for estimating **loan amounts** by analyzing features such as:

- Age
- Income
- Credit Score
- Debt-to-Income Ratio
- Previous Defaults (categorical)
- Risk Rating (categorical, encoded)

The project compares the performance of several ML and neural network models, focusing on error metrics and training time.

---

## 📈 Models Used

### 🔹 Traditional Machine Learning Models
- Linear Regression
- Lasso Regression
- Ridge Regression
- Random Forest
- Gradient Boosting
- XGBoost

### 🔹 Deep Learning Models
- Artificial Neural Networks (ANN)
- Long Short-Term Memory (LSTM)
- Gated Recurrent Unit (GRU)

---

## 📁 Files in This Repository

Loan Amount Prediction/ │ ├── small project.py # Python script with all models ├── Small project presentation.pdf # Final presentation slides


> A final report and dataset folder may be added later.

---

## 🧪 Evaluation Metrics

- **Mean Squared Error (MSE)**
- **Mean Absolute Error (MAE)**
- **Training Time (seconds)**
- **Validation Loss (for DL models)**

---

## 💡 Key Insights

- Ridge Regression offered the best training speed with high accuracy.
- GRU achieved the lowest error in deep learning runs, although it required more training time.
- ANN models provided a good balance between speed and accuracy.
- Tree-based models like Random Forest and Gradient Boosting were robust but slower.

---

## 🛠️ Requirements

Install the necessary packages using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost tensorflow keras
