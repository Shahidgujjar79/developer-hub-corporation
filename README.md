# 🧬 DevelopersHub — Data Science Portal

A fully interactive Streamlit web application covering all 5 Data Science & Analytics internship tasks.

---

## 📁 Project Structure

```
ds_app/
├── app.py                  ← Main Streamlit entry point
├── requirements.txt        ← Python dependencies
├── README.md               ← This file
└── tasks/
    ├── __init__.py
    ├── task1_iris.py       ← Iris EDA & Classification
    ├── task2_credit.py     ← Credit Risk / Loan Approval
    ├── task3_churn.py      ← Customer Churn Prediction
    ├── task4_insurance.py  ← Insurance Cost Estimation
    └── task5_loan.py       ← Personal Loan Acceptance
```

---

## 🚀 Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the app
```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501**

---

## 📦 Datasets

Each task works **immediately with built-in synthetic data**. Upload real Kaggle datasets via the sidebar for full results.

| Task | Dataset | Source | File to Upload |
|------|---------|--------|----------------|
| Task 1 | Iris Dataset | seaborn built-in | *auto-loaded* |
| Task 2 | Loan Prediction | [Kaggle](https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset) | `train.csv` |
| Task 3 | Churn Modelling | [Kaggle](https://www.kaggle.com/datasets/shubhendra7/customer-churn-dataset) | `Churn_Modelling.csv` |
| Task 4 | Medical Cost Personal | [Kaggle](https://www.kaggle.com/datasets/mirichoi0218/insurance) | `insurance.csv` |
| Task 5 | Bank Personal Loan | [Kaggle](https://www.kaggle.com/datasets/itsmesunil/bank-loan-modelling) | `Bank_Personal_Loan_Modelling.csv` |

---

## 🤖 Models Used

| Task | Model |
|------|-------|
| Task 1 | Random Forest Classifier |
| Task 2 | Logistic Regression + Decision Tree |
| Task 3 | Random Forest Classifier |
| Task 4 | Gradient Boosting Regressor |
| Task 5 | Random Forest + Logistic Regression |

---

## ✅ Features

- 📊 **EDA charts** — scatter plots, histograms, box plots, heatmaps, bar charts
- 🤖 **Trained ML models** — accuracy/R² metrics shown live
- 🔮 **Live prediction** — fill in a form, get instant predictions
- 📁 **Upload real data** — swap synthetic data with real Kaggle CSVs via sidebar
- 🎨 **Dark UI** — clean, dark-themed interface

---

*DevelopersHub Corporation · Data Science & Analytics Internship · Due: 15th May 2026*
