# 🚀 VahanBima CLTV Prediction Pipeline

## 📌 Overview
This project builds an end-to-end Machine Learning pipeline to predict **Customer Lifetime Value (CLTV)** for an insurance company. The solution focuses on feature engineering, model comparison, and interpretability to identify high-value customers.

---

## 🎯 Problem Statement
Predict CLTV for insurance customers to help the business:
- Identify high-value customers  
- Improve retention strategies  
- Personalize services  

---

## 📊 Dataset
- **Train:** ~90,000 records  
- **Test:** ~60,000 records  

### Features:
- Demographics (gender, income, qualification)  
- Policy details  
- Claim history  
- Customer tenure (vintage)  

---

## ⚙️ Approach

### 1. Data Preprocessing
- Handled categorical variables using One-Hot Encoding  
- Ensured feature consistency between train & test  
- Applied **log transformation** to handle skewed target variable  

### 2. Feature Engineering
- Claim per year  
- Claim flag (0/1)  
- Claim-based interaction features  
- Policy-based features  

### 3. Models Used
- Linear Regression  
- Decision Tree  
- Random Forest  
- Gradient Boosting  
- XGBoost  

### 4. Evaluation Metric
- **R² Score**

---

## 📈 Results

| Model              | R² Score |
|-------------------|--------|
| Linear Regression | ~0.32 |
| Decision Tree     | ~0.29 |
| Random Forest     | ~0.32 |
| Gradient Boosting | **~0.33 (Best)** |
| XGBoost           | ~0.32 |

---

## 🚀 Key Insights
- CLTV is highly skewed → log transformation improved performance  
- Claim-related features strongly influence CLTV  
- Number of policies is a key driver of customer value  
- Ensemble models outperformed basic regression models  

---

## 🔍 Model Interpretability
- Feature importance analysis  
- SHAP values used to understand model predictions  

---

## 🛠️ Tech Stack
- Python  
- Pandas, NumPy  
- Scikit-learn  
- XGBoost  
- Matplotlib, Seaborn  

---

## 📁 Project Structure
VahanBima-CLTV-Prediction/
│
├── data/
│ ├── train.csv
│ ├── test.csv
│
├── notebook/ 
│ ├── eda.ipynb
│ 
├── src/
│ ├── preprocessing.py
│ ├── feature_engineering.py
│ ├── model.py
│ ├── train.py
│
├── main.py
├── requirements.txt
├── README.md



---

## ▶️ How to Run

```bash
git clone <your-repo-link>
cd Vahanbima

python -m venv venv
venv\Scripts\activate   # Windows

pip install -r requirements.txt
python main.py
