import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

def generate_loan_data(n=5000):
    np.random.seed(42)
    age = np.random.randint(23, 67, n)
    experience = np.clip(age - np.random.randint(20, 25, n), 0, 45)
    income = np.random.randint(8, 224, n)   # in thousands
    family = np.random.randint(1, 5, n)
    ccavg = np.round(np.random.exponential(1.9, n), 1)
    education = np.random.choice([1, 2, 3], n, p=[0.40, 0.28, 0.32])
    mortgage = np.random.choice(
        np.concatenate([np.zeros(3000), np.random.randint(100, 635, 2000)]),
        n, replace=False
    )
    securities = np.random.choice([0, 1], n, p=[0.90, 0.10])
    cd_account = np.random.choice([0, 1], n, p=[0.94, 0.06])
    online = np.random.choice([0, 1], n, p=[0.40, 0.60])
    credit_card = np.random.choice([0, 1], n, p=[0.71, 0.29])

    # Probability driven by income, education, cc usage
    logit = (
        -6.0
        + 0.04 * income
        + 0.5 * education
        + 0.3 * ccavg
        + 3.5 * cd_account
        - 0.03 * family
    )
    prob = 1 / (1 + np.exp(-logit))
    personal_loan = (np.random.rand(n) < prob).astype(int)

    return pd.DataFrame({
        "Age": age, "Experience": experience, "Income": income,
        "Family": family, "CCAvg": ccavg, "Education": education,
        "Mortgage": mortgage, "Securities Account": securities,
        "CD Account": cd_account, "Online": online,
        "CreditCard": credit_card, "Personal Loan": personal_loan
    })

def run():
    st.markdown("## 💳 Task 5 — Personal Loan Acceptance Prediction")
    st.markdown("Predict whether a bank customer will accept a personal loan offer.")

    # --- Load data ---
    st.sidebar.markdown("### 📂 Dataset")
    uploaded = st.sidebar.file_uploader("Upload Bank_Personal_Loan_Modelling.csv (Kaggle)", type="csv", key="t5")
    if uploaded:
        df = pd.read_csv(uploaded)
        if "ID" in df.columns:
            df = df.drop(columns=["ID", "ZIP Code"], errors="ignore")
        st.sidebar.success("✅ Real dataset loaded!")
    else:
        df = generate_loan_data()
        st.sidebar.info("Using synthetic demo data. Upload real dataset for best results.\n\n[Download from Kaggle](https://www.kaggle.com/datasets/itsmesunil/bank-loan-modelling)")

    st.markdown(f"**Dataset:** {df.shape[0]:,} records | Loan acceptance rate: **{df['Personal Loan'].mean()*100:.1f}%**")

    # --- EDA Charts ---
    st.markdown("### 📊 Exploratory Data Analysis")

    tab1, tab2, tab3 = st.tabs(["Income vs Acceptance", "Education Breakdown", "Feature Heatmap"])

    with tab1:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        colors = {0: "#3498db", 1: "#e74c3c"}
        labels = {0: "Rejected", 1: "Accepted"}
        for val, grp in df.groupby("Personal Loan"):
            axes[0].hist(grp["Income"], bins=30, alpha=0.65,
                         color=colors[val], label=labels[val])
        axes[0].set_title("Income Distribution by Loan Decision")
        axes[0].set_xlabel("Income ($ thousands)")
        axes[0].legend()

        for val, grp in df.groupby("Personal Loan"):
            axes[1].scatter(grp["Income"], grp["CCAvg"], alpha=0.3,
                            color=colors[val], label=labels[val], s=15)
        axes[1].set_title("Income vs CC Spending")
        axes[1].set_xlabel("Income ($ thousands)")
        axes[1].set_ylabel("CC Avg Spending ($k/month)")
        axes[1].legend()
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    with tab2:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        edu_map = {1: "Undergrad", 2: "Graduate", 3: "Advanced"}
        df["Edu Label"] = df["Education"].map(edu_map)
        edu_rate = df.groupby("Edu Label")["Personal Loan"].mean() * 100
        edu_rate.sort_values().plot(kind="barh", ax=axes[0], color="#9b59b6")
        axes[0].set_title("Loan Acceptance Rate by Education")
        axes[0].set_xlabel("Acceptance Rate (%)")

        family_rate = df.groupby("Family")["Personal Loan"].mean() * 100
        axes[1].bar(family_rate.index, family_rate.values, color="#1abc9c")
        axes[1].set_title("Acceptance Rate by Family Size")
        axes[1].set_xlabel("Family Members")
        axes[1].set_ylabel("Acceptance Rate (%)")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    with tab3:
        numeric_cols = ["Age", "Income", "Family", "CCAvg", "Mortgage", "Personal Loan"]
        available = [c for c in numeric_cols if c in df.columns]
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(df[available].corr(), annot=True, fmt=".2f", cmap="coolwarm",
                    linewidths=0.5, ax=ax)
        ax.set_title("Feature Correlation Heatmap")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    # --- Model Training ---
    st.markdown("### 🤖 Model Comparison")

    feature_cols = [c for c in df.columns if c != "Personal Loan"]
    X = df[feature_cols]
    y = df["Personal Loan"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(max_depth=7, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }
    results = {}
    best_model = None
    best_acc = 0
    for name, m in models.items():
        if name == "Logistic Regression":
            m.fit(X_train_sc, y_train)
            acc = accuracy_score(y_test, m.predict(X_test_sc))
        else:
            m.fit(X_train, y_train)
            acc = accuracy_score(y_test, m.predict(X_test))
        results[name] = acc
        if acc > best_acc:
            best_acc = acc
            best_model = (name, m)

    cols = st.columns(3)
    for i, (name, acc) in enumerate(results.items()):
        cols[i].metric(name, f"{acc*100:.2f}%", delta="✅ Best" if name == best_model[0] else None)

    # Feature importance from RF
    rf = models["Random Forest"]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    feat_series = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=True).tail(8)
    feat_series.plot(kind="barh", ax=ax, color="#e74c3c")
    ax.set_title("Top Feature Importances (Random Forest)")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()

    # --- Prediction ---
    st.markdown("### 🔮 Will This Customer Accept the Loan?")
    with st.form("loan_form"):
        c1, c2, c3 = st.columns(3)
        age = c1.number_input("Age", 23, 67, 35)
        income = c2.number_input("Income ($k/yr)", 8, 224, 60)
        ccavg = c3.number_input("CC Avg Spending ($k/mo)", 0.0, 10.0, 1.5, step=0.1)
        c4, c5, c6 = st.columns(3)
        family = c4.selectbox("Family Size", [1, 2, 3, 4])
        education = c5.selectbox("Education", [1, 2, 3], format_func=lambda x: {1:"Undergrad",2:"Graduate",3:"Advanced"}[x])
        mortgage = c6.number_input("Mortgage ($k)", 0, 635, 0)
        c7, c8, c9 = st.columns(3)
        securities = c7.selectbox("Securities Account", ["No", "Yes"])
        cd_account = c8.selectbox("CD Account", ["No", "Yes"])
        online = c9.selectbox("Online Banking", ["No", "Yes"])
        credit_card = st.selectbox("Has Credit Card", ["No", "Yes"])
        exp = max(0, age - 22)
        submit = st.form_submit_button("🎯 Predict Loan Decision", use_container_width=True)

    if submit:
        # Build input matching training columns
        input_dict = {
            "Age": age, "Experience": exp, "Income": income,
            "Family": family, "CCAvg": ccavg, "Education": education,
            "Mortgage": mortgage,
            "Securities Account": 1 if securities == "Yes" else 0,
            "CD Account": 1 if cd_account == "Yes" else 0,
            "Online": 1 if online == "Yes" else 0,
            "CreditCard": 1 if credit_card == "Yes" else 0
        }
        input_df = pd.DataFrame([input_dict])[feature_cols]
        prediction = rf.predict(input_df)[0]
        prob = rf.predict_proba(input_df)[0][1]

        if prediction == 1:
            st.success(f"✅ **ACCEPTED** — This customer is likely to accept the loan offer!")
        else:
            st.error(f"❌ **REJECTED** — This customer is unlikely to accept the loan offer.")
        st.progress(float(prob), text=f"Acceptance Probability: {prob*100:.1f}%")

        if income > 100:
            st.info("💡 High income is the strongest predictor of loan acceptance.")
        if cd_account == "Yes":
            st.info("💡 CD account holders are ~5x more likely to accept personal loans.")
