import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings("ignore")

def generate_credit_data(n=614):
    np.random.seed(0)
    gender = np.random.choice(["Male", "Female"], n, p=[0.65, 0.35])
    married = np.random.choice(["Yes", "No"], n, p=[0.65, 0.35])
    dependents = np.random.choice(["0", "1", "2", "3+"], n, p=[0.57, 0.17, 0.16, 0.10])
    education = np.random.choice(["Graduate", "Not Graduate"], n, p=[0.78, 0.22])
    self_employed = np.random.choice(["Yes", "No"], n, p=[0.14, 0.86])
    applicant_income = np.random.exponential(5000, n).astype(int) + 1000
    coapplicant_income = np.random.choice(
        np.concatenate([np.zeros(300), np.random.exponential(2000, 314)]), n, replace=False
    )
    loan_amount = (applicant_income * np.random.uniform(0.8, 4.0, n) / 1000).astype(int)
    loan_term = np.random.choice([120, 180, 240, 300, 360, 480], n, p=[0.03, 0.04, 0.04, 0.07, 0.79, 0.03])
    credit_history = np.random.choice([1, 0], n, p=[0.84, 0.16])
    property_area = np.random.choice(["Urban", "Rural", "Semiurban"], n)

    # Approval logic
    logit = (
        -1.5
        + 0.0001 * applicant_income
        + 2.5 * credit_history
        + 0.5 * (education == "Graduate").astype(int)
        + 0.3 * (property_area == "Semiurban").astype(int)
        - 0.003 * loan_amount
        + np.random.normal(0, 0.3, n)
    )
    prob = 1 / (1 + np.exp(-logit))
    status = np.where(np.random.rand(n) < prob, "Y", "N")

    return pd.DataFrame({
        "Gender": gender, "Married": married, "Dependents": dependents,
        "Education": education, "Self_Employed": self_employed,
        "ApplicantIncome": applicant_income, "CoapplicantIncome": np.round(coapplicant_income, 0),
        "LoanAmount": loan_amount, "Loan_Amount_Term": loan_term,
        "Credit_History": credit_history, "Property_Area": property_area,
        "Loan_Status": status
    })

def run():
    st.markdown("## 💰 Task 2 — Credit Risk & Loan Default Prediction")
    st.markdown("Predict whether a loan application will be approved or rejected.")

    st.sidebar.markdown("### 📂 Dataset")
    uploaded = st.sidebar.file_uploader("Upload train.csv (Kaggle Loan Prediction)", type="csv", key="t2")
    if uploaded:
        df = pd.read_csv(uploaded)
        if "Loan_ID" in df.columns:
            df = df.drop(columns=["Loan_ID"])
        df = df.dropna()
        st.sidebar.success("✅ Real dataset loaded!")
    else:
        df = generate_credit_data()
        st.sidebar.info("Using synthetic demo data.\n\n[Download from Kaggle](https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset)")

    st.markdown(f"**Dataset:** {len(df):,} records · Approval rate: **{(df['Loan_Status']=='Y').mean()*100:.1f}%**")

    st.markdown("### 📊 Exploratory Data Analysis")
    tab1, tab2 = st.tabs(["Approval by Category", "Income Distribution"])

    with tab1:
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        for ax, col in zip(axes, ["Education", "Property_Area", "Credit_History"]):
            ct = df.groupby([col, "Loan_Status"]).size().unstack(fill_value=0)
            ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
            ct_pct.plot(kind="bar", ax=ax, color=["#e74c3c", "#2ecc71"], edgecolor="white")
            ax.set_title(f"Approval by {col}")
            ax.set_xlabel("")
            ax.set_ylabel("% of Applications")
            ax.tick_params(axis="x", rotation=30)
            ax.legend(["Rejected", "Approved"], fontsize=8)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    with tab2:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        colors = {"Y": "#2ecc71", "N": "#e74c3c"}
        for status, grp in df.groupby("Loan_Status"):
            axes[0].hist(grp["ApplicantIncome"], bins=30, alpha=0.65, color=colors[status],
                         label="Approved" if status == "Y" else "Rejected")
        axes[0].set_title("Applicant Income by Loan Status")
        axes[0].set_xlabel("Income ($)")
        axes[0].legend()

        for status, grp in df.groupby("Loan_Status"):
            axes[1].hist(grp["LoanAmount"].dropna(), bins=25, alpha=0.65, color=colors[status],
                         label="Approved" if status == "Y" else "Rejected")
        axes[1].set_title("Loan Amount by Status")
        axes[1].set_xlabel("Loan Amount ($k)")
        axes[1].legend()
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Model
    st.markdown("### 🤖 Model Training")
    df_enc = df.copy()
    cat_cols = ["Gender", "Married", "Dependents", "Education", "Self_Employed", "Property_Area"]
    le = LabelEncoder()
    for col in cat_cols:
        df_enc[col] = le.fit_transform(df_enc[col].astype(str))
    df_enc["Loan_Status"] = (df_enc["Loan_Status"] == "Y").astype(int)

    feature_cols = [c for c in df_enc.columns if c != "Loan_Status"]
    X = df_enc[feature_cols].fillna(df_enc[feature_cols].median())
    y = df_enc["Loan_Status"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    lr = LogisticRegression(max_iter=500)
    dt = DecisionTreeClassifier(max_depth=6, random_state=42)
    lr.fit(X_sc, y_train)
    dt.fit(X_train, y_train)

    lr_acc = accuracy_score(y_test, lr.predict(X_test_sc))
    dt_acc = accuracy_score(y_test, dt.predict(X_test))

    c1, c2 = st.columns(2)
    c1.metric("Logistic Regression Accuracy", f"{lr_acc*100:.2f}%")
    c2.metric("Decision Tree Accuracy", f"{dt_acc*100:.2f}%")

    # Prediction
    st.markdown("### 🔮 Predict Loan Approval")
    with st.form("credit_form"):
        c1, c2, c3 = st.columns(3)
        income = c1.number_input("Applicant Income ($)", 1000, 100000, 5000, step=500)
        co_income = c2.number_input("Coapplicant Income ($)", 0, 50000, 0, step=500)
        loan_amt = c3.number_input("Loan Amount ($k)", 10, 700, 150)
        c4, c5, c6 = st.columns(3)
        gender = c4.selectbox("Gender", ["Male", "Female"])
        married = c5.selectbox("Married", ["Yes", "No"])
        education = c6.selectbox("Education", ["Graduate", "Not Graduate"])
        c7, c8, c9 = st.columns(3)
        self_emp = c7.selectbox("Self Employed", ["No", "Yes"])
        dependents = c8.selectbox("Dependents", ["0", "1", "2", "3+"])
        property_area = c9.selectbox("Property Area", ["Urban", "Rural", "Semiurban"])
        credit_hist = st.radio("Credit History (1 = Good)", [1, 0], horizontal=True)
        submit = st.form_submit_button("🏦 Check Loan Eligibility", use_container_width=True)

    if submit:
        dep_map = {"0": 0, "1": 1, "2": 2, "3+": 3}
        input_dict = {
            "Gender": 1 if gender == "Male" else 0,
            "Married": 1 if married == "Yes" else 0,
            "Dependents": dep_map[dependents],
            "Education": 0 if education == "Graduate" else 1,
            "Self_Employed": 1 if self_emp == "Yes" else 0,
            "ApplicantIncome": income,
            "CoapplicantIncome": co_income,
            "LoanAmount": loan_amt,
            "Loan_Amount_Term": 360,
            "Credit_History": credit_hist,
            "Property_Area": {"Urban": 2, "Rural": 0, "Semiurban": 1}[property_area]
        }
        input_df = pd.DataFrame([input_dict])[feature_cols]
        input_sc = scaler.transform(input_df)
        pred = lr.predict(input_sc)[0]
        prob = lr.predict_proba(input_sc)[0][1]

        if pred == 1:
            st.success(f"✅ **APPROVED** — Loan likely to be approved! ({prob*100:.1f}% confidence)")
        else:
            st.error(f"❌ **REJECTED** — Application likely to be rejected. ({(1-prob)*100:.1f}% rejection confidence)")
        if credit_hist == 0:
            st.warning("⚠️ Poor credit history is the single biggest rejection factor.")
