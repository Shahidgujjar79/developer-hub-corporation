import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

def generate_churn_data(n=10000):
    np.random.seed(7)
    row_number = np.arange(1, n + 1)
    customer_id = np.random.randint(15000000, 16000000, n)
    surname = ["Smith"] * n
    credit_score = np.random.randint(350, 851, n)
    geography = np.random.choice(["France", "Spain", "Germany"], n, p=[0.50, 0.25, 0.25])
    gender = np.random.choice(["Male", "Female"], n, p=[0.55, 0.45])
    age = np.random.randint(18, 93, n)
    tenure = np.random.randint(0, 11, n)
    balance = np.where(np.random.rand(n) < 0.35, 0,
                       np.round(np.random.uniform(10000, 250000, n), 2))
    num_of_products = np.random.choice([1, 2, 3, 4], n, p=[0.50, 0.46, 0.025, 0.015])
    has_cr_card = np.random.choice([0, 1], n, p=[0.29, 0.71])
    is_active = np.random.choice([0, 1], n, p=[0.49, 0.51])
    estimated_salary = np.round(np.random.uniform(11.58, 199992.48, n), 2)

    logit = (
        -3.5
        + 0.005 * (age - 40)
        + 0.3 * (geography == "Germany").astype(int)
        - 0.4 * is_active
        + 0.5 * (num_of_products >= 3).astype(int)
        - 0.002 * (credit_score - 600)
        + np.random.normal(0, 0.3, n)
    )
    prob = 1 / (1 + np.exp(-logit))
    exited = (np.random.rand(n) < prob).astype(int)

    return pd.DataFrame({
        "RowNumber": row_number, "CustomerId": customer_id, "Surname": surname,
        "CreditScore": credit_score, "Geography": geography, "Gender": gender,
        "Age": age, "Tenure": tenure, "Balance": balance,
        "NumOfProducts": num_of_products, "HasCrCard": has_cr_card,
        "IsActiveMember": is_active, "EstimatedSalary": estimated_salary,
        "Exited": exited
    })

def run():
    st.markdown("## 🏦 Task 3 — Bank Customer Churn Prediction")
    st.markdown("Identify customers likely to leave the bank before they do.")

    st.sidebar.markdown("### 📂 Dataset")
    uploaded = st.sidebar.file_uploader("Upload Churn_Modelling.csv (Kaggle)", type="csv", key="t3")
    if uploaded:
        df = pd.read_csv(uploaded)
        st.sidebar.success("✅ Real dataset loaded!")
    else:
        df = generate_churn_data()
        st.sidebar.info("Using synthetic demo data.\n\n[Download from Kaggle](https://www.kaggle.com/datasets/shubhendra7/customer-churn-dataset)")

    drop_cols = ["RowNumber", "CustomerId", "Surname"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    churn_rate = df["Exited"].mean() * 100
    st.markdown(f"**Dataset:** {len(df):,} customers · Churn rate: **{churn_rate:.1f}%**")

    st.markdown("### 📊 Exploratory Data Analysis")
    tab1, tab2, tab3 = st.tabs(["Churn by Geography & Gender", "Age & Balance", "Feature Importance"])

    with tab1:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        geo_churn = df.groupby("Geography")["Exited"].mean() * 100
        axes[0].bar(geo_churn.index, geo_churn.values, color=["#3498db", "#e74c3c", "#2ecc71"])
        axes[0].set_title("Churn Rate by Geography (%)")
        axes[0].set_ylabel("Churn Rate (%)")
        for i, v in enumerate(geo_churn.values):
            axes[0].text(i, v + 0.5, f"{v:.1f}%", ha="center", fontsize=10)

        gender_churn = df.groupby("Gender")["Exited"].mean() * 100
        axes[1].bar(gender_churn.index, gender_churn.values, color=["#9b59b6", "#e67e22"])
        axes[1].set_title("Churn Rate by Gender (%)")
        axes[1].set_ylabel("Churn Rate (%)")
        for i, v in enumerate(gender_churn.values):
            axes[1].text(i, v + 0.5, f"{v:.1f}%", ha="center", fontsize=10)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    with tab2:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        colors = {0: "#3498db", 1: "#e74c3c"}
        for exited, grp in df.groupby("Exited"):
            axes[0].hist(grp["Age"], bins=25, alpha=0.65, color=colors[exited],
                         label="Churned" if exited else "Retained")
        axes[0].set_title("Age Distribution: Churned vs Retained")
        axes[0].set_xlabel("Age")
        axes[0].legend()

        non_zero = df[df["Balance"] > 0]
        for exited, grp in non_zero.groupby("Exited"):
            axes[1].hist(grp["Balance"], bins=25, alpha=0.65, color=colors[exited],
                         label="Churned" if exited else "Retained")
        axes[1].set_title("Balance Distribution (non-zero)")
        axes[1].set_xlabel("Balance ($)")
        axes[1].legend()
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Train model
    df_enc = df.copy()
    le = LabelEncoder()
    for col in ["Geography", "Gender"]:
        df_enc[col] = le.fit_transform(df_enc[col])

    feature_cols = [c for c in df_enc.columns if c != "Exited"]
    X = df_enc[feature_cols]
    y = df_enc["Exited"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    with tab3:
        acc = accuracy_score(y_test, model.predict(X_test))
        st.metric("Model Accuracy (Random Forest)", f"{acc*100:.2f}%")
        fig, ax = plt.subplots(figsize=(7, 4))
        feat_imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values()
        feat_imp.plot(kind="barh", ax=ax, color="#e74c3c")
        ax.set_title("Feature Importance — What Drives Churn?")
        ax.set_xlabel("Importance Score")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("### 🔮 Predict Customer Churn Risk")
    with st.form("churn_form"):
        c1, c2, c3 = st.columns(3)
        credit_score = c1.number_input("Credit Score", 350, 850, 650)
        age = c2.number_input("Age", 18, 92, 40)
        tenure = c3.number_input("Years with Bank", 0, 10, 3)
        c4, c5, c6 = st.columns(3)
        balance = c4.number_input("Account Balance ($)", 0, 250000, 75000, step=1000)
        salary = c5.number_input("Estimated Salary ($)", 0, 200000, 80000, step=1000)
        num_products = c6.selectbox("Number of Products", [1, 2, 3, 4])
        c7, c8, c9 = st.columns(3)
        geography = c7.selectbox("Country", ["France", "Spain", "Germany"])
        gender = c8.selectbox("Gender", ["Male", "Female"])
        is_active = c9.selectbox("Active Member", ["Yes", "No"])
        has_card = st.radio("Has Credit Card", ["Yes", "No"], horizontal=True)
        submit = st.form_submit_button("🎯 Predict Churn Risk", use_container_width=True)

    if submit:
        geo_map = {"France": 0, "Germany": 1, "Spain": 2}
        input_dict = {
            "CreditScore": credit_score,
            "Geography": geo_map[geography],
            "Gender": 0 if gender == "Female" else 1,
            "Age": age, "Tenure": tenure, "Balance": balance,
            "NumOfProducts": num_products,
            "HasCrCard": 1 if has_card == "Yes" else 0,
            "IsActiveMember": 1 if is_active == "Yes" else 0,
            "EstimatedSalary": salary
        }
        input_df = pd.DataFrame([input_dict])[feature_cols]
        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1]

        if pred == 1:
            st.error(f"⚠️ **HIGH CHURN RISK** — This customer is likely to leave! ({prob*100:.1f}% probability)")
            st.info("💡 Consider offering retention incentives: loyalty rewards, personalized offers, or relationship manager outreach.")
        else:
            st.success(f"✅ **LOW CHURN RISK** — Customer is likely to stay. ({(1-prob)*100:.1f}% retention probability)")
        st.progress(float(prob), text=f"Churn Probability: {prob*100:.1f}%")

        if geography == "Germany":
            st.warning("🇩🇪 German customers churn at nearly double the rate of French customers.")
        if num_products >= 3:
            st.warning("📦 Customers with 3+ products have unusually high churn rates.")
