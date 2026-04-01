# Vahanbima
Machine Learning project to predict Customer Lifetime Value (CLTV) for an insurance company using advanced feature engineering, regression models, and model interpretability techniques (SHAP).



# 🚗 VahanBima CLTV Prediction

## 📌 Problem Statement
Predict Customer Lifetime Value (CLTV) for insurance customers to help the company identify high-value customers and personalize services.

---

## 📊 Dataset
- Train: ~90K records
- Test: ~60K records
- Features:
  - Demographics (gender, income, qualification)
  - Policy details
  - Claim history
  - Customer tenure (vintage)

---

## 🎯 Objective
Build a high-performance and interpretable ML model to predict CLTV.

---

## 🧠 Approach

### 1. Data Preprocessing
- Handling categorical variables (One-Hot Encoding)
- Feature alignment between train & test
- Log transformation for skewed target

### 2. Feature Engineering
- Claim per year
- Claim flag (0 / 1)
- Claim bins
- Interaction features

### 3. Models Used
- Linear Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- XGBoost

### 4. Evaluation Metric
- R² Score

---

## 📈 Results

| Model              | R² Score |
|-------------------|---------|
| Linear Regression | ~0.15   |
| Decision Tree     | ~0.11   |
| Random Forest     | ~0.15   |
| Gradient Boosting | ~0.16   |
| XGBoost           | ~0.14   |

---

## 🚀 Key Insights
- CLTV is highly skewed → log transformation improves performance
- Claim behavior strongly impacts CLTV
- Number of policies is a major driver of CLTV

---

## 🔍 Model Interpretation
- Feature importance analysis
- SHAP values for explainability

---

## 🛠️ Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost
- Matplotlib, Seaborn

---

## 📁 Project Structure

├── data/train.csv,test.csv

├── notebooks/eda.ipynb
├── src/main.ipynb
├── submission/submission.csv
├── README.md








---

## 📌 Future Improvements
- Hyperparameter tuning (Optuna)
- Advanced ensemble methods
- Deep learning models

---

## 👨‍💻 Author
Somesh Verma
