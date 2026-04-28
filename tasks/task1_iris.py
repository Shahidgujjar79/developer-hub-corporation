import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings("ignore")

def run():
    st.markdown("## 🌸 Task 1 — Iris Species Classification")
    st.markdown("Explore the classic Iris dataset and classify flower species from measurements.")

    # Load data
    try:
        df = sns.load_dataset("iris")
    except Exception:
        np.random.seed(42)
        n = 150
        species = np.repeat(["setosa", "versicolor", "virginica"], 50)
        df = pd.DataFrame({
            "sepal_length": np.concatenate([np.random.normal(5.0, 0.35, 50), np.random.normal(5.9, 0.52, 50), np.random.normal(6.6, 0.64, 50)]),
            "sepal_width":  np.concatenate([np.random.normal(3.4, 0.38, 50), np.random.normal(2.8, 0.31, 50), np.random.normal(3.0, 0.32, 50)]),
            "petal_length": np.concatenate([np.random.normal(1.5, 0.17, 50), np.random.normal(4.3, 0.47, 50), np.random.normal(5.6, 0.55, 50)]),
            "petal_width":  np.concatenate([np.random.normal(0.25, 0.11, 50), np.random.normal(1.33, 0.20, 50), np.random.normal(2.0, 0.27, 50)]),
            "species": species
        })

    st.markdown(f"**Dataset:** {len(df)} samples · 3 species · 4 features")

    # EDA
    st.markdown("### 📊 Exploratory Data Analysis")
    tab1, tab2, tab3 = st.tabs(["Scatter Plot", "Histogram", "Box Plot"])

    palette = {"setosa": "#e74c3c", "versicolor": "#3498db", "virginica": "#2ecc71"}

    with tab1:
        fig, ax = plt.subplots(figsize=(8, 5))
        for species, grp in df.groupby("species"):
            ax.scatter(grp["petal_length"], grp["petal_width"],
                       color=palette[species], label=species.capitalize(), alpha=0.8, s=60, edgecolors="white", linewidth=0.5)
        ax.set_xlabel("Petal Length (cm)")
        ax.set_ylabel("Petal Width (cm)")
        ax.set_title("Petal Length vs Petal Width by Species")
        ax.legend(title="Species")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    with tab2:
        fig, axes = plt.subplots(2, 2, figsize=(10, 7))
        features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
        for ax, feat in zip(axes.flat, features):
            for species, grp in df.groupby("species"):
                ax.hist(grp[feat], bins=15, alpha=0.65, color=palette[species], label=species.capitalize())
            ax.set_title(feat.replace("_", " ").title())
            ax.set_xlabel("cm")
        axes[0][1].legend(title="Species")
        fig.suptitle("Feature Distributions by Species", fontsize=13, fontweight="bold")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    with tab3:
        fig, axes = plt.subplots(1, 4, figsize=(12, 5))
        for ax, feat in zip(axes, features):
            data = [df[df["species"] == s][feat].values for s in ["setosa", "versicolor", "virginica"]]
            bp = ax.boxplot(data, patch_artist=True, labels=["Set.", "Ver.", "Vir."])
            for patch, color in zip(bp["boxes"], palette.values()):
                patch.set_facecolor(color)
                patch.set_alpha(0.75)
            ax.set_title(feat.replace("_", " ").title(), fontsize=9)
        fig.suptitle("Feature Spread Across Species", fontsize=12, fontweight="bold")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Model
    st.markdown("### 🤖 Model Training — Random Forest")
    features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    X = df[features]
    y = df["species"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    st.metric("Test Accuracy", f"{acc*100:.2f}%")

    # Prediction
    st.markdown("### 🔮 Classify a New Flower")
    with st.form("iris_form"):
        c1, c2, c3, c4 = st.columns(4)
        sl = c1.number_input("Sepal Length (cm)", 4.0, 8.0, 5.1, step=0.1)
        sw = c2.number_input("Sepal Width (cm)", 2.0, 4.5, 3.5, step=0.1)
        pl = c3.number_input("Petal Length (cm)", 1.0, 7.0, 1.4, step=0.1)
        pw = c4.number_input("Petal Width (cm)", 0.1, 2.5, 0.2, step=0.1)
        submit = st.form_submit_button("🌸 Classify Species", use_container_width=True)

    if submit:
        inp = pd.DataFrame([[sl, sw, pl, pw]], columns=features)
        pred = model.predict(inp)[0]
        probs = model.predict_proba(inp)[0]
        conf = max(probs) * 100
        emoji = {"setosa": "🔴", "versicolor": "🔵", "virginica": "🟢"}
        st.success(f"{emoji.get(pred, '🌸')} Predicted Species: **{pred.capitalize()}** ({conf:.1f}% confidence)")
        for cls, p in zip(model.classes_, probs):
            st.progress(float(p), text=f"{cls.capitalize()}: {p*100:.1f}%")
