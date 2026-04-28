import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

def generate_insurance_data(n=1338):
    np.random.seed(42)
    ages = np.random.randint(18, 65, n)
    sexes = np.random.choice(["male", "female"], n)
    bmis = np.round(np.random.normal(30.7, 6.1, n), 1)
    bmis = np.clip(bmis, 15, 55)
    children = np.random.choice([0, 1, 2, 3, 4, 5], n, p=[0.43, 0.24, 0.18, 0.10, 0.03, 0.02])
    smokers = np.random.choice(["yes", "no"], n, p=[0.20, 0.80])
    regions = np.random.choice(["northeast", "northwest", "southeast", "southwest"], n)

    charges = (
        250 * ages
        + 150 * bmis
        + 500 * children
        + np.where(smokers == "yes", 24000, 0)
        + np.where(bmis > 30, np.where(smokers == "yes", 20000, 2000), 0)
        + np.random.normal(0, 1500, n)
    )
    charges = np.round(np.clip(charges, 1000, 65000), 2)

    return pd.DataFrame({
        "age": ages, "sex": sexes, "bmi": bmis,
        "children": children, "smoker": smokers,
        "region": regions, "charges": charges
    })

def run():
    st.markdown("## 🏥 Task 4 — Medical Insurance Cost Prediction")
    st.markdown("Predict annual insurance charges based on personal and health attributes.")

    # --- Load data ---
    st.sidebar.markdown("### 📂 Dataset")
    uploaded = st.sidebar.file_uploader("Upload insurance.csv (Kaggle)", type="csv", key="t4")
    if uploaded:
        df = pd.read_csv(uploaded)
        st.sidebar.success("✅ Real dataset loaded!")
    else:
        df = generate_insurance_data()
        st.sidebar.info("Using synthetic demo data. Upload real Kaggle dataset for best results.\n\n[Download from Kaggle](https://www.kaggle.com/datasets/mirichoi0218/insurance)")

    st.markdown(f"**Dataset:** {df.shape[0]:,} records, {df.shape[1]} features")

    # --- EDA Charts ---
    st.markdown("### 📊 Exploratory Data Analysis")

    tab1, tab2, tab3 = st.tabs(["BMI vs Charges", "Age vs Charges", "Distribution"])

    with tab1:
        fig, ax = plt.subplots(figsize=(9, 5))
        colors = {"yes": "#e74c3c", "no": "#3498db"}
        for smoker_val, grp in df.groupby("smoker"):
            ax.scatter(grp["bmi"], grp["charges"], alpha=0.5,
                       color=colors[smoker_val], label=f"Smoker: {smoker_val}", s=25)
        ax.set_xlabel("BMI")
        ax.set_ylabel("Annual Charges ($)")
        ax.set_title("BMI vs Insurance Charges (by Smoker Status)")
        ax.legend()
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    with tab2:
        fig, ax = plt.subplots(figsize=(9, 5))
        for smoker_val, grp in df.groupby("smoker"):
            ax.scatter(grp["age"], grp["charges"], alpha=0.5,
                       color=colors[smoker_val], label=f"Smoker: {smoker_val}", s=25)
        ax.set_xlabel("Age")
        ax.set_ylabel("Annual Charges ($)")
        ax.set_title("Age vs Insurance Charges")
        ax.legend()
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    with tab3:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].hist(df["charges"], bins=40, color="#9b59b6", edgecolor="white", alpha=0.85)
        axes[0].set_title("Distribution of Insurance Charges")
        axes[0].set_xlabel("Charges ($)")
        axes[0].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}k"))

        avg_by_region = df.groupby("region")["charges"].mean().sort_values()
        axes[1].barh(avg_by_region.index, avg_by_region.values, color="#1abc9c")
        axes[1].set_title("Avg Charges by Region")
        axes[1].set_xlabel("Avg Charges ($)")
        axes[1].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}k"))
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    # --- Model Training ---
    st.markdown("### 🤖 Model Training")
    le = LabelEncoder()
    df_enc = df.copy()
    for col in ["sex", "smoker", "region"]:
        df_enc[col] = le.fit_transform(df_enc[col])

    X = df_enc.drop("charges", axis=1)
    y = df_enc["charges"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = GradientBoostingRegressor(n_estimators=150, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    col1, col2, col3 = st.columns(3)
    col1.metric("MAE", f"${mean_absolute_error(y_test, y_pred):,.0f}")
    col2.metric("RMSE", f"${np.sqrt(mean_squared_error(y_test, y_pred)):,.0f}")
    col3.metric("R² Score", f"{r2_score(y_test, y_pred):.3f}")

    # Feature importance
    fig, ax = plt.subplots(figsize=(7, 3.5))
    feat_imp = pd.Series(model.feature_importances_, index=X.columns).sort_values()
    feat_imp.plot(kind="barh", ax=ax, color="#e67e22")
    ax.set_title("Feature Importance (Gradient Boosting)")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()

    # --- Prediction ---
    st.markdown("### 🔮 Predict Your Insurance Cost")
    with st.form("ins_form"):
        c1, c2, c3 = st.columns(3)
        age = c1.number_input("Age", 18, 64, 30)
        bmi = c2.number_input("BMI", 15.0, 55.0, 27.5, step=0.1)
        children = c3.selectbox("Children", [0, 1, 2, 3, 4, 5])
        c4, c5, c6 = st.columns(3)
        sex = c4.selectbox("Sex", ["male", "female"])
        smoker = c5.selectbox("Smoker", ["no", "yes"])
        region = c6.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])
        submit = st.form_submit_button("💰 Estimate Cost", use_container_width=True)

    if submit:
        # Encode inputs
        sex_enc = 1 if sex == "male" else 0
        smoker_enc = 1 if smoker == "yes" else 0
        region_map = {"northeast": 0, "northwest": 1, "southeast": 2, "southwest": 3}
        input_data = pd.DataFrame([[age, sex_enc, bmi, children, smoker_enc, region_map[region]]],
                                  columns=X.columns)
        prediction = model.predict(input_data)[0]
        monthly = prediction / 12

        st.success(f"💵 Estimated Annual Charge: **${prediction:,.2f}**")
        st.info(f"📅 That's roughly **${monthly:,.2f}/month**")

        if smoker == "yes":
            st.warning("🚬 Smoking significantly increases your insurance cost — typically 3–4x higher than non-smokers.")
        if bmi > 30:
            st.warning("⚖️ BMI > 30 (obese range) is a key cost driver. Even a 5-point reduction can meaningfully lower premiums.")
