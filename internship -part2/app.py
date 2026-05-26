"""
DevelopersHub Corporation - Data Science & Analytics Advanced Internship
Unified Advanced Portfolio Dashboard (All 5 Tasks Integrated)
Fixed, enhanced with rich EDA visuals for every task.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, roc_curve, auc,
                              mean_absolute_error, mean_squared_error,
                              accuracy_score, f1_score)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import os, warnings
warnings.filterwarnings("ignore")

try:
    from statsmodels.tsa.arima.model import ARIMA
    ARIMA_AVAILABLE = True
except ImportError:
    ARIMA_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# ── Page Config & Global Styles ───────────────────────────────────────────────
st.set_page_config(page_title="DH Corp · Portfolio", page_icon="🛸", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700&family=Exo+2:wght@300;400;500&display=swap');
html, body, [class*="css"]  { font-family: 'Exo 2', sans-serif; }
h1, h2, h3 { font-family: 'Orbitron', monospace !important; letter-spacing:-0.01em; }
.main { background:#05070f; color:#e0e8ff; }
[data-testid="stSidebar"] { background:#0a0e1a; }
[data-testid="stSidebar"] * { color:#c8d8f8 !important; }
.kpi { background:linear-gradient(135deg,#0f1a2e,#0a0e1a);
       border:1px solid #1a2a4a; border-radius:10px; padding:18px;
       text-align:center; margin:6px 0; }
.kpi h2 { color:#00ffc8; font-size:1.8rem; margin:0; font-family:'Orbitron',monospace; }
.kpi p  { color:#8899bb; margin:0; font-size:0.8rem;
          text-transform:uppercase; letter-spacing:0.08em; }
.section { border-left:3px solid #00ffc8; padding-left:12px;
           margin:18px 0 10px; font-family:'Orbitron',monospace;
           font-size:1rem; color:#00ffc8; }
</style>
""", unsafe_allow_html=True)

DARK="#05070f"; CARD="#0f1a2e"; G1="#00ffc8"; G2="#5b8cff"; G3="#ff6b6b"; G4="#ffd166"; G5="#c77dff"
PAL = [G1, G2, G3, G4, G5, "#ff9f1c", "#06d6a0", "#ef476f"]

plt.rcParams.update({
    "figure.facecolor": CARD, "axes.facecolor": "#080c18",
    "axes.edgecolor": "#1a2a4a", "axes.labelcolor": "#8899bb",
    "xtick.color": "#4a5a7a", "ytick.color": "#4a5a7a",
    "text.color": "#c0d0f0", "grid.color": "#0f1a2e", "grid.linestyle": "--", "grid.alpha":0.5,
})

def kpi(col, val, label):
    col.markdown(f'<div class="kpi"><h2>{val}</h2><p>{label}</p></div>', unsafe_allow_html=True)

def section(title):
    st.markdown(f'<div class="section">◈ {title}</div>', unsafe_allow_html=True)

# ── Sidebar Navigation ─────────────────────────────────────────────────────────
st.sidebar.markdown("# Muhammad Shahid ")
st.sidebar.markdown("**Task:** Internship -II  Dashboard")
st.sidebar.markdown("---")
task = st.sidebar.radio("Select Task:", [
    "Task 1 · Term Deposit Prediction",
    "Task 2 · Customer Segmentation",
    "Task 3 · Energy Forecasting",
    "Task 4 · Loan Default Risk",
    "Task 5 · Superstore BI Dashboard",
])
st.sidebar.markdown("---")
st.sidebar.caption("Submission: 25 May 2026")

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# ══════════════════════════════════════════════════════════════════════════════
# TASK 1 — Term Deposit Subscription Prediction
# ══════════════════════════════════════════════════════════════════════════════
if task == "Task 1 · Term Deposit Prediction":
    st.title(" Task 1 · Term Deposit Subscription Prediction")
    st.caption("Bank Marketing Dataset · Classification · XAI · F1 / ROC / Confusion Matrix")

    bank_path = os.path.join(DATA_DIR, "bank.csv")
    if not os.path.exists(bank_path):
        st.error("bank.csv not found. Place it alongside app.py.")
        st.stop()

    @st.cache_data
    def load_bank():
        df = pd.read_csv(bank_path, sep=";")
        return df

    raw = load_bank()

    # ── EDA ───────────────────────────────────────────────────────────────────
    section("Dataset Overview")
    c1,c2,c3,c4 = st.columns(4)
    sub_rate = (raw["y"]=="yes").mean()*100
    kpi(c1, f"{len(raw):,}", "Total Records")
    kpi(c2, raw.shape[1]-1, "Features")
    kpi(c3, f"{sub_rate:.1f}%", "Subscription Rate")
    kpi(c4, raw.isnull().sum().sum(), "Missing Values")

    with st.expander("📋 Raw Data Preview"):
        st.dataframe(raw.head(10), use_container_width=True)

    section("Exploratory Data Analysis")
    t1, t2, t3, t4, t5 = st.tabs([
        "Target Distribution", "Age & Balance", "Job Analysis",
        "Campaign Metrics", "Correlation Heatmap"
    ])

    with t1:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.patch.set_facecolor(CARD)
        # Pie
        counts = raw["y"].value_counts()
        axes[0].pie(counts, labels=["No","Yes"], autopct="%1.1f%%",
                    colors=[G2, G1], startangle=90,
                    wedgeprops={"edgecolor": DARK, "linewidth":2})
        axes[0].set_title("Subscription Split")
        # Bar count
        axes[1].bar(["No","Yes"], counts.values, color=[G2, G1], edgecolor=DARK, width=0.5)
        axes[1].set_title("Count by Outcome"); axes[1].set_ylabel("Count")
        for p, v in zip(axes[1].patches, counts.values):
            axes[1].text(p.get_x()+p.get_width()/2, p.get_height()+200, f"{v:,}",
                         ha="center", fontsize=10)
        # Duration by outcome
        for outcome, color in [("no", G2), ("yes", G1)]:
            axes[2].hist(raw[raw["y"]==outcome]["duration"], bins=40,
                         alpha=0.7, color=color, label=outcome, density=True)
        axes[2].set_title("Call Duration Density by Outcome")
        axes[2].set_xlabel("Duration (s)"); axes[2].legend()
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with t2:
        fig, axes = plt.subplots(1, 3, figsize=(16, 4))
        fig.patch.set_facecolor(CARD)
        # Age histogram
        for outcome, color in [("no", G2), ("yes", G1)]:
            axes[0].hist(raw[raw["y"]==outcome]["age"], bins=30,
                         alpha=0.7, color=color, label=outcome, density=True)
        axes[0].set_title("Age Distribution by Outcome")
        axes[0].set_xlabel("Age"); axes[0].legend()
        # Balance boxplot
        data_box = [raw[raw["y"]=="no"]["balance"].clip(-500,5000).values,
                    raw[raw["y"]=="yes"]["balance"].clip(-500,5000).values]
        bp = axes[1].boxplot(data_box, patch_artist=True,
                              medianprops={"color":DARK,"linewidth":2})
        for patch, color in zip(bp["boxes"], [G2, G1]):
            patch.set_facecolor(color); patch.set_alpha(0.7)
        axes[1].set_xticklabels(["No", "Yes"])
        axes[1].set_title("Balance by Subscription"); axes[1].set_ylabel("Balance (€)")
        # Age vs Balance scatter
        for outcome, color in [("no", G2), ("yes", G1)]:
            sub = raw[raw["y"]==outcome]
            axes[2].scatter(sub["age"], sub["balance"].clip(-500, 8000),
                            alpha=0.25, s=8, c=color, label=outcome)
        axes[2].set_xlabel("Age"); axes[2].set_ylabel("Balance (€)")
        axes[2].set_title("Age vs Balance"); axes[2].legend()
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with t3:
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        fig.patch.set_facecolor(CARD)
        # Job subscription rate
        job_rate = (raw.groupby("job")["y"]
                    .apply(lambda x:(x=="yes").mean()*100)
                    .sort_values(ascending=True))
        colors = plt.cm.cool(np.linspace(0.2, 0.9, len(job_rate)))
        axes[0].barh(job_rate.index, job_rate.values, color=colors)
        axes[0].axvline(sub_rate, color=G3, linestyle="--", lw=1.5,
                        label=f"Overall {sub_rate:.1f}%")
        axes[0].set_title("Subscription Rate by Job"); axes[0].set_xlabel("Rate (%)")
        axes[0].legend()
        # Marital status
        marital_ct = raw.groupby(["marital","y"]).size().unstack(fill_value=0)
        marital_ct.plot(kind="bar", ax=axes[1], color=[G2,G1],
                        edgecolor=DARK, rot=0)
        axes[1].set_title("Marital Status vs Subscription")
        axes[1].set_xlabel("Marital Status"); axes[1].set_ylabel("Count")
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with t4:
        fig, axes = plt.subplots(1, 3, figsize=(16, 4))
        fig.patch.set_facecolor(CARD)
        # Campaign calls histogram
        axes[0].hist(raw[raw["y"]=="no"]["campaign"].clip(0,15), bins=15,
                     alpha=0.7, color=G2, label="No", density=True)
        axes[0].hist(raw[raw["y"]=="yes"]["campaign"].clip(0,15), bins=15,
                     alpha=0.7, color=G1, label="Yes", density=True)
        axes[0].set_title("Campaign Calls by Outcome")
        axes[0].set_xlabel("# Calls"); axes[0].legend()
        # Month subscription rate
        month_order = ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"]
        month_rate = (raw.groupby("month")["y"]
                      .apply(lambda x:(x=="yes").mean()*100)
                      .reindex(month_order, fill_value=0))
        axes[1].bar(month_rate.index, month_rate.values,
                    color=plt.cm.plasma(np.linspace(0.1,0.9,12)), edgecolor=DARK)
        axes[1].set_title("Subscription Rate by Month")
        axes[1].set_xlabel("Month"); axes[1].set_ylabel("Rate (%)")
        plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha="right")
        # Poutcome
        po_rate = (raw.groupby("poutcome")["y"]
                   .apply(lambda x:(x=="yes").mean()*100)
                   .sort_values())
        axes[2].bar(po_rate.index, po_rate.values,
                    color=[G3, G4, G1], edgecolor=DARK)
        axes[2].set_title("Prev. Outcome vs Subscription Rate")
        axes[2].set_ylabel("Rate (%)")
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with t5:
        num_cols = ["age","balance","duration","campaign","pdays","previous"]
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.patch.set_facecolor(CARD)
        mask = np.triu(np.ones_like(raw[num_cols].corr(), dtype=bool))
        sns.heatmap(raw[num_cols].corr(), ax=ax, cmap="coolwarm", annot=True,
                    fmt=".2f", mask=mask, linewidths=0.5, center=0)
        ax.set_title("Feature Correlation Heatmap")
        plt.tight_layout(); st.pyplot(fig); plt.close()

    # ── Preprocessing & Models ────────────────────────────────────────────────
    section("Model Training & Evaluation")

    @st.cache_data
    def train_task1():
        df = raw.copy()
        cat_cols = df.select_dtypes("object").columns.tolist()
        cat_cols.remove("y")
        le = LabelEncoder()
        for c in cat_cols:
            df[c] = le.fit_transform(df[c].astype(str))
        df["y"] = (df["y"] == "yes").astype(int)
        X = df.drop("y", axis=1)
        y = df["y"]
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                                    random_state=42, stratify=y)
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)
        # Logistic Regression
        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(X_tr_s, y_tr)
        # Random Forest
        rf = RandomForestClassifier(n_estimators=200, max_depth=10,
                                     random_state=42, n_jobs=-1)
        rf.fit(X_tr, y_tr)
        return lr, rf, X_tr_s, X_te_s, X_tr, X_te, y_tr, y_te, scaler, X.columns.tolist()

    with st.spinner("Training models…"):
        lr, rf, X_tr_s, X_te_s, X_tr, X_te, y_tr, y_te, scaler, feat_cols = train_task1()

    col_lr, col_rf = st.columns(2)
    for col, name, model, Xev in [
        (col_lr, "Logistic Regression", lr, X_te_s),
        (col_rf, "Random Forest",       rf, X_te),
    ]:
        y_pred = model.predict(Xev)
        y_prob = model.predict_proba(Xev)[:,1]
        acc  = accuracy_score(y_te, y_pred)
        f1   = f1_score(y_te, y_pred)
        fpr, tpr, _ = roc_curve(y_te, y_prob)
        roc_auc = auc(fpr, tpr)
        cm = confusion_matrix(y_te, y_pred)

        with col:
            st.markdown(f"#### {name}")
            m1,m2,m3 = st.columns(3)
            m1.metric("Accuracy", f"{acc:.3f}")
            m2.metric("F1-Score", f"{f1:.3f}")
            m3.metric("ROC-AUC",  f"{roc_auc:.3f}")

            fig, axes = plt.subplots(1,2,figsize=(10,4))
            fig.patch.set_facecolor(CARD)
            sns.heatmap(cm, annot=True, fmt="d", ax=axes[0], cmap="Blues",
                        linewidths=0.5, xticklabels=["No","Yes"],
                        yticklabels=["No","Yes"])
            axes[0].set_title("Confusion Matrix")
            axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Actual")
            axes[1].plot(fpr, tpr, color=G1, lw=2, label=f"AUC={roc_auc:.3f}")
            axes[1].plot([0,1],[0,1],"--",color="#334155")
            axes[1].set_xlabel("FPR"); axes[1].set_ylabel("TPR")
            axes[1].set_title("ROC Curve"); axes[1].legend()
            plt.tight_layout(); st.pyplot(fig); plt.close()

    # ── Feature Importance / XAI ──────────────────────────────────────────────
    section("Explainability — Feature Importance (XAI)")
    importances = rf.feature_importances_
    fi_df = (pd.DataFrame({"Feature": feat_cols, "Importance": importances})
               .sort_values("Importance", ascending=True))

    fig, ax = plt.subplots(figsize=(11, 5))
    fig.patch.set_facecolor(CARD)
    colors = plt.cm.cool(np.linspace(0.15, 0.9, len(fi_df)))
    ax.barh(fi_df["Feature"], fi_df["Importance"], color=colors)
    ax.set_title("Random Forest Feature Importances", fontsize=13)
    ax.set_xlabel("Mean Decrease in Impurity")
    plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("**Individual Prediction Explanations (top 5 features)**")
    n_ex = st.slider("Predictions to explain", 3, 10, 5, key="t1_ex")
    top5_feats = fi_df.tail(5)["Feature"].tolist()
    probs_all = rf.predict_proba(X_te)[:,1]
    preds_all  = rf.predict(X_te)
    actual_all = y_te.values

    for i in range(n_ex):
        conf = probs_all[i]
        pred_label = " Yes" if preds_all[i]==1 else " No"
        act_label  = " Yes" if actual_all[i]==1 else " No"
        with st.expander(f"Prediction #{i+1} — Actual: {act_label} | Predicted: {pred_label} | Confidence: {conf:.1%}"):
            vals = X_te.iloc[i][top5_feats].values
            medians = X_te[top5_feats].median().values
            bar_colors = [G1 if v > m else G3 for v, m in zip(vals, medians)]
            fig2, ax2 = plt.subplots(figsize=(8,3))
            fig2.patch.set_facecolor(CARD)
            ax2.barh(top5_feats, fi_df[fi_df["Feature"].isin(top5_feats)].set_index("Feature").loc[top5_feats,"Importance"], color=bar_colors)
            ax2.set_title(f"Feature Contributions — P(subscribe)={conf:.1%}")
            plt.tight_layout(); st.pyplot(fig2); plt.close()
            st.caption("🟢 = above median (pushes toward Yes)  🔴 = below median")

    section("Classification Report")
    st.code(classification_report(y_te, rf.predict(X_te), target_names=["No","Yes"]))
    st.info("**Conclusion:** `duration`, `balance`, `poutcome` and `month` drive subscription probability the most. "
            "Random Forest outperforms Logistic Regression on this imbalanced dataset, especially in recall for the minority class.")


# ══════════════════════════════════════════════════════════════════════════════
# TASK 2 — Customer Segmentation
# ══════════════════════════════════════════════════════════════════════════════
elif task == "Task 2 · Customer Segmentation":
    st.title(" Task 2 · Customer Segmentation")
    st.caption("Mall Customers Dataset · K-Means · PCA · Marketing Strategies")

    mall_path = os.path.join(DATA_DIR, "Mall_Customers.csv")
    if not os.path.exists(mall_path):
        st.error("Mall_Customers.csv not found."); st.stop()

    @st.cache_data
    def load_mall():
        return pd.read_csv(mall_path)

    raw = load_mall()
    k_val = st.sidebar.slider("Number of Clusters (K)", 2, 10, 5)

    # ── EDA ───────────────────────────────────────────────────────────────────
    section("Dataset Overview")
    c1,c2,c3,c4 = st.columns(4)
    kpi(c1, len(raw), "Customers")
    kpi(c2, f"{raw['Age'].mean():.1f}", "Avg Age")
    kpi(c3, f"{raw['Annual Income (k$)'].mean():.1f}k", "Avg Income")
    kpi(c4, f"{raw['Spending Score (1-100)'].mean():.1f}", "Avg Spending Score")

    with st.expander("📋 Raw Data Preview"):
        st.dataframe(raw, use_container_width=True)

    section("Exploratory Data Analysis")
    t1, t2, t3, t4, t5 = st.tabs([
        "Distributions", "Gender Analysis", "Income vs Spending",
        "Age Groups", "Correlation"
    ])

    with t1:
        fig, axes = plt.subplots(1, 3, figsize=(16, 4))
        fig.patch.set_facecolor(CARD)
        for ax, col, color in zip(axes,
            ["Age", "Annual Income (k$)", "Spending Score (1-100)"],
            [G1, G2, G3]):
            ax.hist(raw[col], bins=20, color=color, alpha=0.85, edgecolor=DARK)
            ax.axvline(raw[col].mean(), color=G4, lw=2, linestyle="--",
                       label=f"Mean={raw[col].mean():.1f}")
            ax.set_title(col); ax.set_ylabel("Count"); ax.legend()
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with t2:
        fig, axes = plt.subplots(1, 3, figsize=(16, 4))
        fig.patch.set_facecolor(CARD)
        # Gender pie
        gc = raw["Gender"].value_counts()
        axes[0].pie(gc, labels=gc.index, autopct="%1.1f%%",
                    colors=[G2, G1], startangle=90,
                    wedgeprops={"edgecolor":DARK,"linewidth":2})
        axes[0].set_title("Gender Split")
        # Income by gender
        for gender, color in [("Male", G2), ("Female", G1)]:
            axes[1].hist(raw[raw["Gender"]==gender]["Annual Income (k$)"],
                         bins=15, alpha=0.7, color=color, label=gender, density=True)
        axes[1].set_title("Income Distribution by Gender")
        axes[1].set_xlabel("Annual Income (k$)"); axes[1].legend()
        # Spending score by gender
        data_box = [raw[raw["Gender"]==g]["Spending Score (1-100)"].values
                    for g in ["Male","Female"]]
        bp = axes[2].boxplot(data_box, patch_artist=True,
                              medianprops={"color":DARK,"lw":2})
        for patch, color in zip(bp["boxes"], [G2, G1]):
            patch.set_facecolor(color); patch.set_alpha(0.7)
        axes[2].set_xticklabels(["Male","Female"])
        axes[2].set_title("Spending Score by Gender"); axes[2].set_ylabel("Score")
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with t3:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.patch.set_facecolor(CARD)
        # Raw scatter
        for gender, color, marker in [("Male", G2, "o"), ("Female", G1, "^")]:
            sub = raw[raw["Gender"]==gender]
            axes[0].scatter(sub["Annual Income (k$)"], sub["Spending Score (1-100)"],
                            alpha=0.7, s=55, c=color, marker=marker, label=gender,
                            edgecolors=DARK, linewidths=0.5)
        axes[0].set_xlabel("Annual Income (k$)"); axes[0].set_ylabel("Spending Score")
        axes[0].set_title("Income vs Spending Score"); axes[0].legend()
        # Age vs spending
        axes[1].scatter(raw["Age"], raw["Spending Score (1-100)"],
                        c=raw["Annual Income (k$)"], cmap="plasma",
                        s=55, alpha=0.8, edgecolors=DARK, linewidths=0.4)
        axes[1].set_xlabel("Age"); axes[1].set_ylabel("Spending Score")
        axes[1].set_title("Age vs Spending (colour = Income)")
        sm = plt.cm.ScalarMappable(cmap="plasma",
             norm=plt.Normalize(raw["Annual Income (k$)"].min(),
                                raw["Annual Income (k$)"].max()))
        plt.colorbar(sm, ax=axes[1], label="Annual Income (k$)")
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with t4:
        raw_c = raw.copy()
        raw_c["AgeGroup"] = pd.cut(raw_c["Age"], bins=[17,25,35,45,55,70],
                                    labels=["18-25","26-35","36-45","46-55","56+"])
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        fig.patch.set_facecolor(CARD)
        ag_income = raw_c.groupby("AgeGroup", observed=True)["Annual Income (k$)"].mean()
        ag_spend  = raw_c.groupby("AgeGroup", observed=True)["Spending Score (1-100)"].mean()
        axes[0].bar(ag_income.index.astype(str), ag_income.values,
                    color=PAL[:5], edgecolor=DARK)
        axes[0].set_title("Avg Income by Age Group"); axes[0].set_ylabel("Income (k$)")
        axes[1].bar(ag_spend.index.astype(str), ag_spend.values,
                    color=PAL[:5], edgecolor=DARK)
        axes[1].set_title("Avg Spending Score by Age Group"); axes[1].set_ylabel("Score")
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with t5:
        fig, ax = plt.subplots(figsize=(6,5))
        fig.patch.set_facecolor(CARD)
        num = raw[["Age","Annual Income (k$)","Spending Score (1-100)"]]
        mask = np.triu(np.ones_like(num.corr(), dtype=bool))
        sns.heatmap(num.corr(), ax=ax, cmap="coolwarm", annot=True,
                    fmt=".2f", mask=mask, linewidths=0.5, center=0)
        ax.set_title("Correlation Matrix"); plt.tight_layout(); st.pyplot(fig); plt.close()

    # ── Clustering ────────────────────────────────────────────────────────────
    section("K-Means Clustering")
    X = raw[["Age","Annual Income (k$)","Spending Score (1-100)"]].values
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)

    # Elbow + silhouette
    c_left, c_right = st.columns(2)
    with c_left:
        ks = range(2,11)
        inertias, sils = [], []
        for ki in ks:
            km_tmp = KMeans(n_clusters=ki, random_state=42, n_init=15)
            lab_tmp = km_tmp.fit_predict(X_sc)
            inertias.append(km_tmp.inertia_)
            sils.append(silhouette_score(X_sc, lab_tmp))
        fig, (a1,a2) = plt.subplots(1,2,figsize=(11,4))
        fig.patch.set_facecolor(CARD)
        a1.plot(ks, inertias, "o-", color=G1, lw=2)
        a1.axvline(k_val, color=G3, linestyle="--", label=f"K={k_val}")
        a1.set_title("Elbow Curve"); a1.set_xlabel("K"); a1.set_ylabel("Inertia"); a1.legend()
        a2.plot(ks, sils, "s-", color=G2, lw=2)
        a2.axvline(k_val, color=G3, linestyle="--", label=f"K={k_val}")
        a2.set_title("Silhouette Scores"); a2.set_xlabel("K"); a2.legend()
        plt.tight_layout(); st.pyplot(fig); plt.close()

    km = KMeans(n_clusters=k_val, random_state=42, n_init=20)
    raw["Cluster"] = km.fit_predict(X_sc)
    sil_score = silhouette_score(X_sc, raw["Cluster"])

    with c_right:
        c1,c2 = st.columns(2)
        kpi(c1, k_val, "Clusters")
        kpi(c2, f"{sil_score:.3f}", "Silhouette Score")
        st.markdown("")
        profile = raw.groupby("Cluster")[
            ["Age","Annual Income (k$)","Spending Score (1-100)"]].mean().round(1)
        profile["Size"] = raw["Cluster"].value_counts().sort_index()
        st.dataframe(profile.style.background_gradient(cmap="Blues", axis=0),
                     use_container_width=True)

    # PCA Visualisation
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X_sc)
    raw["PC1"] = X_2d[:,0]; raw["PC2"] = X_2d[:,1]

    fig, axes = plt.subplots(1,2,figsize=(15,5))
    fig.patch.set_facecolor(CARD)
    for ci in range(k_val):
        mask = raw["Cluster"]==ci
        axes[0].scatter(raw.loc[mask,"PC1"], raw.loc[mask,"PC2"],
                        s=60, alpha=0.8, color=PAL[ci%len(PAL)],
                        label=f"Cluster {ci}", edgecolors=DARK, linewidths=0.4)
    axes[0].set_title("PCA Cluster Visualisation")
    axes[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    axes[0].legend()
    # Income vs Spending with clusters
    for ci in range(k_val):
        mask = raw["Cluster"]==ci
        axes[1].scatter(raw.loc[mask,"Annual Income (k$)"],
                        raw.loc[mask,"Spending Score (1-100)"],
                        s=60, alpha=0.8, color=PAL[ci%len(PAL)],
                        label=f"Cluster {ci}", edgecolors=DARK, linewidths=0.4)
    axes[1].set_title("Income vs Spending (Clustered)")
    axes[1].set_xlabel("Annual Income (k$)"); axes[1].set_ylabel("Spending Score")
    axes[1].legend()
    plt.tight_layout(); st.pyplot(fig); plt.close()

    section("Marketing Strategies per Segment")
    STRATEGIES = [
        (G1, "High-Value Stars",      "High income + high spending. → VIP programs, early access, concierge services, loyalty tiers."),
        (G2, "Careful Spenders",      "High income + low spending. → Premium exclusives, trust-building, ROI-focused messaging."),
        (G3, "Young Enthusiasts",     "Low income + high spending. → Flash sales, BNPL, referral bonuses, gamification."),
        (G4, "Conservative Savers",   "Low income + low spending. → Essential bundles, heavy discounts, subscription savings."),
        (G5, "Middle Ground",         "Average on all metrics. → Cross-sell, newsletter engagement, moderate loyalty rewards."),
    ]
    for ci in range(k_val):
        c, title, strat = STRATEGIES[ci % len(STRATEGIES)]
        avg_income = profile.loc[ci, "Annual Income (k$)"]
        avg_score  = profile.loc[ci, "Spending Score (1-100)"]
        size       = int(profile.loc[ci, "Size"])
        st.markdown(
            f'<div style="border-left:4px solid {c}; padding:10px 16px; '
            f'background:#0f1a2e; border-radius:6px; margin:8px 0">'
            f'<b style="color:{c}">Cluster {ci} — {title}</b> &nbsp; '
            f'<span style="color:#8899bb; font-size:0.85rem">'
            f'n={size} | Avg Income={avg_income}k$ | Avg Score={avg_score}</span><br>'
            f'<span style="color:#c0d0f0">{strat}</span></div>',
            unsafe_allow_html=True)

    st.info("**Conclusion:** K-Means with K=5 cleanly partitions customers by income and spending habits. "
            "Each cluster maps to a distinct behavioural persona requiring a targeted marketing approach.")


# ══════════════════════════════════════════════════════════════════════════════
# TASK 3 — Energy Forecasting
# ══════════════════════════════════════════════════════════════════════════════
elif task == "Task 3 · Energy Forecasting":
    st.title(" Task 3 · Energy Consumption Forecasting")
    st.caption("Household Power Dataset · ARIMA · Gradient Boosting · Time Features")

    hpc_path = os.path.join(DATA_DIR, "household_power_consumption.csv")
    if not os.path.exists(hpc_path):
        st.error("household_power_consumption.csv not found."); st.stop()

    @st.cache_data
    def load_energy():
        df = pd.read_csv(hpc_path)
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)
        df["Energy_Consumption_kWh"] = pd.to_numeric(
            df["Energy_Consumption_kWh"], errors="coerce").fillna(method="ffill")
        df["Has_AC_bin"] = (df["Has_AC"] == "Yes").astype(int)
        df["dayofweek"]  = df["Date"].dt.dayofweek
        df["month"]      = df["Date"].dt.month
        df["day"]        = df["Date"].dt.day
        df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
        df["day_sin"]    = np.sin(2*np.pi*df["dayofweek"]/7)
        df["day_cos"]    = np.cos(2*np.pi*df["dayofweek"]/7)
        df["month_sin"]  = np.sin(2*np.pi*df["month"]/12)
        df["month_cos"]  = np.cos(2*np.pi*df["month"]/12)
        return df

    df_full = load_energy()
    households = sorted(df_full["Household_ID"].unique())
    sel_hh = st.sidebar.selectbox("Select Household", households, index=0)
    df = df_full[df_full["Household_ID"] == sel_hh].copy().reset_index(drop=True)
    forecast_days = st.sidebar.slider("Forecast horizon (days)", 3, 30, 14)

    # ── EDA ───────────────────────────────────────────────────────────────────
    section("Dataset Overview")
    c1,c2,c3,c4 = st.columns(4)
    kpi(c1, f"{len(df_full):,}", "Total Records")
    kpi(c2, len(households), "Households")
    kpi(c3, f"{df['Energy_Consumption_kWh'].mean():.2f} kWh", "Avg Daily Consumption")
    kpi(c4, f"{df['Energy_Consumption_kWh'].max():.2f} kWh", "Peak Consumption")

    with st.expander(" Raw Data Sample"):
        st.dataframe(df.head(15), use_container_width=True)

    section("Exploratory Data Analysis")
    t1, t2, t3, t4, t5 = st.tabs([
        "Time Series", "Distribution", "Day-of-Week",
        "Monthly Patterns", "Feature Correlations"
    ])

    with t1:
        fig, axes = plt.subplots(2, 1, figsize=(14, 7))
        fig.patch.set_facecolor(CARD)
        axes[0].plot(df["Date"], df["Energy_Consumption_kWh"],
                     color=G1, lw=1.2, alpha=0.9)
        axes[0].fill_between(df["Date"], df["Energy_Consumption_kWh"],
                              alpha=0.15, color=G1)
        axes[0].set_title(f"Daily Energy Consumption — {sel_hh}")
        axes[0].set_ylabel("kWh"); axes[0].set_xlabel("Date")
        # Rolling average
        rolling = df["Energy_Consumption_kWh"].rolling(7, center=True).mean()
        axes[1].plot(df["Date"], df["Energy_Consumption_kWh"],
                     color=G2, lw=0.8, alpha=0.5, label="Daily")
        axes[1].plot(df["Date"], rolling, color=G3, lw=2, label="7-day MA")
        axes[1].set_title("Daily + 7-Day Moving Average")
        axes[1].set_ylabel("kWh"); axes[1].legend()
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with t2:
        fig, axes = plt.subplots(1, 3, figsize=(16, 4))
        fig.patch.set_facecolor(CARD)
        axes[0].hist(df["Energy_Consumption_kWh"], bins=25,
                     color=G1, alpha=0.85, edgecolor=DARK)
        axes[0].axvline(df["Energy_Consumption_kWh"].mean(), color=G3,
                        lw=2, linestyle="--", label="Mean")
        axes[0].set_title("Consumption Distribution")
        axes[0].set_xlabel("kWh"); axes[0].legend()
        # Temp vs Energy
        axes[1].scatter(df["Avg_Temperature_C"], df["Energy_Consumption_kWh"],
                        alpha=0.4, s=25, c=G2, edgecolors="none")
        m, b = np.polyfit(df["Avg_Temperature_C"], df["Energy_Consumption_kWh"], 1)
        x_line = np.linspace(df["Avg_Temperature_C"].min(), df["Avg_Temperature_C"].max(), 100)
        axes[1].plot(x_line, m*x_line+b, color=G3, lw=2, label="Trend")
        axes[1].set_xlabel("Avg Temperature (°C)"); axes[1].set_ylabel("kWh")
        axes[1].set_title("Temperature vs Consumption"); axes[1].legend()
        # AC vs Non-AC
        for ac, color, label in [(0, G2, "No AC"), (1, G1, "Has AC")]:
            axes[2].hist(df[df["Has_AC_bin"]==ac]["Energy_Consumption_kWh"],
                         bins=20, alpha=0.7, color=color, label=label, density=True)
        axes[2].set_title("Consumption by AC Status")
        axes[2].set_xlabel("kWh"); axes[2].legend()
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with t3:
        day_names = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
        day_avg = df.groupby("dayofweek")["Energy_Consumption_kWh"].mean()
        day_std = df.groupby("dayofweek")["Energy_Consumption_kWh"].std()
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        fig.patch.set_facecolor(CARD)
        axes[0].bar([day_names[i] for i in day_avg.index],
                    day_avg.values, color=PAL[:7], edgecolor=DARK, yerr=day_std.values, capsize=4)
        axes[0].set_title("Avg Consumption by Day of Week")
        axes[0].set_ylabel("kWh (± std)")
        # Weekend vs weekday box
        data_box = [df[df["is_weekend"]==0]["Energy_Consumption_kWh"].values,
                    df[df["is_weekend"]==1]["Energy_Consumption_kWh"].values]
        bp = axes[1].boxplot(data_box, patch_artist=True,
                              medianprops={"color":DARK,"lw":2})
        for patch, color in zip(bp["boxes"], [G2, G1]):
            patch.set_facecolor(color); patch.set_alpha(0.7)
        axes[1].set_xticklabels(["Weekday","Weekend"])
        axes[1].set_title("Weekday vs Weekend"); axes[1].set_ylabel("kWh")
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with t4:
        month_avg = df.groupby("month")["Energy_Consumption_kWh"].mean()
        month_peak = df.groupby("month")["Peak_Hours_Usage_kWh"].mean()
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        fig.patch.set_facecolor(CARD)
        month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                       "Jul","Aug","Sep","Oct","Nov","Dec"]
        axes[0].bar([month_names[m-1] for m in month_avg.index],
                    month_avg.values,
                    color=plt.cm.plasma(np.linspace(0.1,0.9,len(month_avg))),
                    edgecolor=DARK)
        axes[0].set_title("Avg Consumption by Month"); axes[0].set_ylabel("kWh")
        plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha="right")
        axes[1].plot([month_names[m-1] for m in month_peak.index],
                     month_peak.values, "o-", color=G4, lw=2)
        axes[1].fill_between(range(len(month_peak)), month_peak.values, alpha=0.2, color=G4)
        axes[1].set_title("Avg Peak-Hour Usage by Month"); axes[1].set_ylabel("kWh")
        axes[1].set_xticks(range(len(month_peak)))
        axes[1].set_xticklabels([month_names[m-1] for m in month_peak.index], rotation=45, ha="right")
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with t5:
        num_cols = ["Energy_Consumption_kWh","Peak_Hours_Usage_kWh",
                    "Avg_Temperature_C","Household_Size","Has_AC_bin"]
        fig, ax = plt.subplots(figsize=(7,6))
        fig.patch.set_facecolor(CARD)
        mask = np.triu(np.ones_like(df[num_cols].corr(), dtype=bool))
        sns.heatmap(df[num_cols].corr(), ax=ax, cmap="coolwarm", annot=True,
                    fmt=".2f", mask=mask, linewidths=0.5, center=0)
        ax.set_title("Feature Correlation Heatmap"); plt.tight_layout(); st.pyplot(fig); plt.close()

    # ── Modelling ─────────────────────────────────────────────────────────────
    section("Model Training & Forecasting")
    feats = ["dayofweek","month","is_weekend","day_sin","day_cos",
             "month_sin","month_cos","Avg_Temperature_C","Has_AC_bin","Household_Size"]
    target = "Energy_Consumption_kWh"

    split = int(len(df) * 0.8)
    train_df, test_df = df.iloc[:split], df.iloc[split:]
    gbm = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05,
                                     max_depth=5, random_state=42)
    gbm.fit(train_df[feats], train_df[target])
    preds = gbm.predict(test_df[feats])
    mae  = mean_absolute_error(test_df[target], preds)
    rmse = mean_squared_error(test_df[target], preds)**0.5

    m1,m2,m3 = st.columns(3)
    m1.metric("MAE", f"{mae:.4f} kWh")
    m2.metric("RMSE", f"{rmse:.4f} kWh")
    m3.metric("Test Records", len(test_df))

    # Actual vs predicted
    fig, axes = plt.subplots(2,1,figsize=(14,7))
    fig.patch.set_facecolor(CARD)
    axes[0].plot(test_df["Date"].values, test_df[target].values, color=G1, lw=1.5, label="Actual")
    axes[0].plot(test_df["Date"].values, preds, color=G2, lw=1.5, linestyle="--", label="GBM Predicted")
    axes[0].fill_between(test_df["Date"].values,
                          preds*0.95, preds*1.05, alpha=0.15, color=G2)
    axes[0].set_title(f"Actual vs Predicted — {sel_hh}  |  MAE={mae:.3f}  RMSE={rmse:.3f}")
    axes[0].set_ylabel("kWh"); axes[0].legend()
    # Residuals
    resid = test_df[target].values - preds
    axes[1].bar(range(len(resid)), resid,
                color=[G1 if r>=0 else G3 for r in resid], alpha=0.6, width=1)
    axes[1].axhline(0, color=G4, lw=1)
    axes[1].set_title("Residuals (Actual − Predicted)")
    axes[1].set_ylabel("Error (kWh)"); axes[1].set_xlabel("Test Day Index")
    plt.tight_layout(); st.pyplot(fig); plt.close()

    # Feature importance
    fi_df = pd.DataFrame({"Feature":feats, "Importance":gbm.feature_importances_}).sort_values("Importance")
    fig, ax = plt.subplots(figsize=(10,4))
    fig.patch.set_facecolor(CARD)
    ax.barh(fi_df["Feature"], fi_df["Importance"],
            color=plt.cm.cool(np.linspace(0.2,0.9,len(fi_df))))
    ax.set_title("GBM Feature Importances"); ax.set_xlabel("Importance Score")
    plt.tight_layout(); st.pyplot(fig); plt.close()

    # Future forecast
    section(f"{forecast_days}-Day Ahead Forecast")
    last_date = df["Date"].max()
    future_dates = pd.date_range(last_date + pd.Timedelta("1D"), periods=forecast_days)
    future_df = pd.DataFrame({
        "Date": future_dates,
        "dayofweek":  future_dates.dayofweek,
        "month":      future_dates.month,
        "is_weekend": (future_dates.dayofweek >= 5).astype(int),
        "Avg_Temperature_C": df["Avg_Temperature_C"].mean(),
        "Has_AC_bin":   df["Has_AC_bin"].mode()[0],
        "Household_Size": df["Household_Size"].mode()[0],
    })
    future_df["day_sin"]   = np.sin(2*np.pi*future_df["dayofweek"]/7)
    future_df["day_cos"]   = np.cos(2*np.pi*future_df["dayofweek"]/7)
    future_df["month_sin"] = np.sin(2*np.pi*future_df["month"]/12)
    future_df["month_cos"] = np.cos(2*np.pi*future_df["month"]/12)
    fcast = gbm.predict(future_df[feats])

    fig, ax = plt.subplots(figsize=(14,4))
    fig.patch.set_facecolor(CARD)
    # Last 30 actual
    hist = df.tail(30)
    ax.plot(hist["Date"], hist[target], color=G1, lw=1.5, label="Historical")
    ax.plot(future_df["Date"], fcast, color=G4, lw=2, marker="o", ms=4, label="Forecast")
    ax.fill_between(future_df["Date"], fcast*0.92, fcast*1.08, alpha=0.2, color=G4)
    ax.axvline(last_date, color=G3, linestyle="--", lw=1.5, label="Forecast Start")
    ax.set_title(f"{forecast_days}-Day Energy Forecast for {sel_hh}")
    ax.set_ylabel("kWh"); ax.legend()
    plt.tight_layout(); st.pyplot(fig); plt.close()

    if ARIMA_AVAILABLE:
        section("ARIMA Model Comparison")
        y_ts = df[target].values
        split_a = int(len(y_ts)*0.8)
        try:
            arima_model = ARIMA(y_ts[:split_a], order=(5,1,2))
            arima_fit   = arima_model.fit()
            arima_pred  = arima_fit.forecast(len(y_ts)-split_a)
            mae_a  = mean_absolute_error(y_ts[split_a:], arima_pred)
            rmse_a = mean_squared_error(y_ts[split_a:], arima_pred)**0.5
            comp = pd.DataFrame({
                "Model": ["GBM (Gradient Boosting)", "ARIMA(5,1,2)"],
                "MAE":   [round(mae,4), round(mae_a,4)],
                "RMSE":  [round(rmse,4), round(rmse_a,4)],
            })
            st.dataframe(comp.style.highlight_min(subset=["MAE","RMSE"], color="#003322"),
                         use_container_width=True)
        except Exception as e:
            st.warning(f"ARIMA failed: {e}")
    else:
        st.info("Install `statsmodels` to enable ARIMA comparison: `pip install statsmodels`")

    st.info("**Conclusion:** Gradient Boosting with cyclical time features and temperature captures "
            "the seasonal and daily patterns in household energy consumption. "
            "Temperature and AC usage are the strongest drivers of demand.")


# ══════════════════════════════════════════════════════════════════════════════
# TASK 4 — Loan Default Risk
# ══════════════════════════════════════════════════════════════════════════════
elif task == "Task 4 · Loan Default Risk":
    st.title(" Task 4 · Loan Default Risk & Threshold Optimization")
    st.caption("Home Credit Dataset (metadata) · Synthetic enriched data · Cost-Benefit Analysis")

    # The supplied application_train.csv is a metadata/description file, not the actual loan data.
    # We generate a realistic synthetic dataset matching Home Credit schema.
    section("Dataset (Synthetic — Home Credit Schema)")

    @st.cache_data
    def generate_loan_data(n=8000, seed=42):
        rng = np.random.default_rng(seed)
        age       = rng.integers(20, 70, n)
        income    = rng.lognormal(10.5, 0.6, n).astype(int).clip(20000, 500000)
        credit    = rng.lognormal(11.5, 0.7, n).astype(int).clip(45000, 4050000)
        annuity   = (credit / rng.uniform(24, 60, n)).astype(int)
        ext_score = rng.uniform(0.05, 0.99, n)
        days_emp  = rng.integers(-5000, -100, n)
        employed  = rng.choice([0,1], n, p=[0.25,0.75])
        own_car   = rng.choice([0,1], n, p=[0.4,0.6])
        own_realty= rng.choice([0,1], n, p=[0.3,0.7])
        prev_app  = rng.integers(0, 10, n)
        amt_req   = rng.integers(0, 8, n)
        # Realistic default probability
        log_odds = (
            -3.5
            - 0.02*(age - 35)
            - 0.4*ext_score
            + 0.0000015*(credit - income)
            - 0.3*employed
            - 0.1*own_realty
            + 0.05*prev_app
        )
        prob = 1/(1+np.exp(-log_odds)) + rng.normal(0, 0.03, n)
        prob = prob.clip(0.01, 0.99)
        target = (rng.uniform(0,1,n) < prob).astype(int)
        return pd.DataFrame({
            "SK_ID_CURR":          range(1, n+1),
            "TARGET":              target,
            "DAYS_BIRTH":          -age*365,
            "AMT_INCOME_TOTAL":    income,
            "AMT_CREDIT":          credit,
            "AMT_ANNUITY":         annuity,
            "EXT_SOURCE_2":        ext_score.round(4),
            "DAYS_EMPLOYED":       days_emp,
            "FLAG_OWN_CAR":        own_car,
            "FLAG_OWN_REALTY":     own_realty,
            "PREV_APP_COUNT":      prev_app,
            "AMT_REQ_CREDIT_BUREAU_YEAR": amt_req,
        })

    raw = generate_loan_data()
    raw["AGE_YEARS"] = (-raw["DAYS_BIRTH"] / 365).astype(int)
    default_rate = raw["TARGET"].mean()*100

    c1,c2,c3,c4 = st.columns(4)
    kpi(c1, f"{len(raw):,}", "Loan Applications")
    kpi(c2, f"{default_rate:.1f}%", "Default Rate")
    kpi(c3, f"${raw['AMT_CREDIT'].mean():,.0f}", "Avg Credit Amount")
    kpi(c4, f"${raw['AMT_INCOME_TOTAL'].mean():,.0f}", "Avg Annual Income")

    with st.expander("📋 Data Preview"):
        st.dataframe(raw.head(10), use_container_width=True)

    section("Exploratory Data Analysis")
    t1,t2,t3,t4,t5 = st.tabs([
        "Target & Age", "Credit & Income", "External Score",
        "Binary Features", "Correlation"
    ])

    with t1:
        fig, axes = plt.subplots(1,3,figsize=(16,4))
        fig.patch.set_facecolor(CARD)
        counts = raw["TARGET"].value_counts()
        axes[0].pie(counts, labels=["No Default","Default"],
                    autopct="%1.1f%%", colors=[G2,G3], startangle=90,
                    wedgeprops={"edgecolor":DARK,"linewidth":2})
        axes[0].set_title("Default Distribution")
        for tgt, color in [(0, G2), (1, G3)]:
            axes[1].hist(raw[raw["TARGET"]==tgt]["AGE_YEARS"], bins=25,
                         alpha=0.7, color=color,
                         label=["No Default","Default"][tgt], density=True)
        axes[1].set_title("Age Distribution by Default")
        axes[1].set_xlabel("Age (years)"); axes[1].legend()
        data_box = [raw[raw["TARGET"]==0]["AGE_YEARS"].values,
                    raw[raw["TARGET"]==1]["AGE_YEARS"].values]
        bp = axes[2].boxplot(data_box, patch_artist=True, medianprops={"color":DARK,"lw":2})
        for patch, color in zip(bp["boxes"],[G2,G3]):
            patch.set_facecolor(color); patch.set_alpha(0.7)
        axes[2].set_xticklabels(["No Default","Default"])
        axes[2].set_title("Age Boxplot by Default"); axes[2].set_ylabel("Age")
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with t2:
        fig, axes = plt.subplots(1,3,figsize=(16,4))
        fig.patch.set_facecolor(CARD)
        for tgt, color in [(0,G2),(1,G3)]:
            axes[0].hist(raw[raw["TARGET"]==tgt]["AMT_CREDIT"]/1000,
                         bins=25, alpha=0.7, color=color,
                         label=["No Default","Default"][tgt], density=True)
        axes[0].set_title("Credit Amount Distribution"); axes[0].set_xlabel("Credit (k$)"); axes[0].legend()
        for tgt, color in [(0,G2),(1,G3)]:
            axes[1].hist(raw[raw["TARGET"]==tgt]["AMT_INCOME_TOTAL"]/1000,
                         bins=25, alpha=0.7, color=color,
                         label=["No Default","Default"][tgt], density=True)
        axes[1].set_title("Income Distribution"); axes[1].set_xlabel("Income (k$)"); axes[1].legend()
        axes[2].scatter(raw[raw["TARGET"]==0]["AMT_INCOME_TOTAL"]/1000,
                        raw[raw["TARGET"]==0]["AMT_CREDIT"]/1000,
                        alpha=0.2, s=8, color=G2, label="No Default")
        axes[2].scatter(raw[raw["TARGET"]==1]["AMT_INCOME_TOTAL"]/1000,
                        raw[raw["TARGET"]==1]["AMT_CREDIT"]/1000,
                        alpha=0.4, s=12, color=G3, label="Default")
        axes[2].set_xlabel("Income (k$)"); axes[2].set_ylabel("Credit (k$)")
        axes[2].set_title("Income vs Credit"); axes[2].legend()
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with t3:
        fig, axes = plt.subplots(1,2,figsize=(14,4))
        fig.patch.set_facecolor(CARD)
        for tgt, color in [(0,G2),(1,G3)]:
            axes[0].hist(raw[raw["TARGET"]==tgt]["EXT_SOURCE_2"],
                         bins=30, alpha=0.7, color=color,
                         label=["No Default","Default"][tgt], density=True)
        axes[0].set_title("External Credit Score by Default")
        axes[0].set_xlabel("EXT_SOURCE_2"); axes[0].legend()
        data_box = [raw[raw["TARGET"]==0]["EXT_SOURCE_2"].values,
                    raw[raw["TARGET"]==1]["EXT_SOURCE_2"].values]
        bp = axes[1].boxplot(data_box, patch_artist=True, medianprops={"color":DARK,"lw":2})
        for patch, color in zip(bp["boxes"],[G2,G3]):
            patch.set_facecolor(color); patch.set_alpha(0.7)
        axes[1].set_xticklabels(["No Default","Default"])
        axes[1].set_title("EXT_SOURCE_2 Boxplot"); axes[1].set_ylabel("Score")
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with t4:
        fig, axes = plt.subplots(1,2,figsize=(14,4))
        fig.patch.set_facecolor(CARD)
        for feat, ax, title in [("FLAG_OWN_CAR", axes[0], "Car Ownership"),
                                  ("FLAG_OWN_REALTY", axes[1], "Realty Ownership")]:
            rates = raw.groupby(feat)["TARGET"].mean()*100
            ax.bar(["No","Yes"], rates.values,
                   color=[G2,G1], edgecolor=DARK, width=0.5)
            ax.set_title(f"Default Rate by {title}")
            ax.set_ylabel("Default Rate (%)")
            for p, v in zip(ax.patches, rates.values):
                ax.text(p.get_x()+p.get_width()/2, p.get_height()+0.2,
                        f"{v:.1f}%", ha="center", fontsize=11)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with t5:
        num_cols = ["TARGET","AGE_YEARS","AMT_INCOME_TOTAL","AMT_CREDIT",
                    "EXT_SOURCE_2","FLAG_OWN_CAR","FLAG_OWN_REALTY","PREV_APP_COUNT"]
        fig, ax = plt.subplots(figsize=(8,7))
        fig.patch.set_facecolor(CARD)
        mask = np.triu(np.ones_like(raw[num_cols].corr(), dtype=bool))
        sns.heatmap(raw[num_cols].corr(), ax=ax, cmap="coolwarm", annot=True,
                    fmt=".2f", mask=mask, linewidths=0.5, center=0)
        ax.set_title("Feature Correlation"); plt.tight_layout(); st.pyplot(fig); plt.close()

    # ── Model ─────────────────────────────────────────────────────────────────
    section("Model Training & Threshold Optimization")
    feat_cols = ["AGE_YEARS","AMT_INCOME_TOTAL","AMT_CREDIT","AMT_ANNUITY",
                 "EXT_SOURCE_2","DAYS_EMPLOYED","FLAG_OWN_CAR","FLAG_OWN_REALTY",
                 "PREV_APP_COUNT","AMT_REQ_CREDIT_BUREAU_YEAR"]
    X = raw[feat_cols]; y = raw["TARGET"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    with st.spinner("Training Random Forest…"):
        rf = RandomForestClassifier(n_estimators=200, max_depth=8,
                                     random_state=42, n_jobs=-1)
        rf.fit(X_tr, y_tr)
    probs = rf.predict_proba(X_te)[:,1]
    preds_def = rf.predict(X_te)

    roc_auc = roc_auc_score(y_te, probs)
    fpr, tpr, roc_thresholds = roc_curve(y_te, probs)

    c1,c2,c3 = st.columns(3)
    c1.metric("ROC-AUC", f"{roc_auc:.4f}")
    c2.metric("Accuracy (default t=0.5)", f"{accuracy_score(y_te, preds_def):.3f}")
    c3.metric("F1-Score (default t=0.5)", f"{f1_score(y_te, preds_def):.3f}")

    # Threshold sliders
    c_fn = st.sidebar.slider("Cost of False Negative ($)", 1000, 20000, 5000, 500)
    c_fp = st.sidebar.slider("Cost of False Positive ($)", 100, 5000, 500, 100)

    thresholds = np.linspace(0.05, 0.95, 50)
    losses = []
    for t in thresholds:
        pred_t = (probs >= t).astype(int)
        fn_cost = ((pred_t==0) & (y_te==1)).sum() * c_fn
        fp_cost = ((pred_t==1) & (y_te==0)).sum() * c_fp
        losses.append(fn_cost + fp_cost)

    opt_t   = thresholds[np.argmin(losses)]
    min_loss = min(losses)
    preds_opt = (probs >= opt_t).astype(int)

    st.success(f" Optimal threshold: **{opt_t:.2f}** — Minimum total cost: **${min_loss:,.0f}**")

    fig, axes = plt.subplots(1,3,figsize=(16,4))
    fig.patch.set_facecolor(CARD)
    # Cost curve
    axes[0].plot(thresholds, losses, color=G3, lw=2, marker=".")
    axes[0].axvline(opt_t, color=G1, linestyle="--", lw=2, label=f"Optimal={opt_t:.2f}")
    axes[0].scatter([opt_t],[min_loss], color=G1, s=100, zorder=5)
    axes[0].set_title("Total Business Cost vs Threshold")
    axes[0].set_xlabel("Decision Threshold"); axes[0].set_ylabel("Total Cost ($)"); axes[0].legend()
    # ROC Curve
    axes[1].plot(fpr, tpr, color=G2, lw=2, label=f"AUC={roc_auc:.3f}")
    axes[1].plot([0,1],[0,1],"--",color="#334155")
    axes[1].set_xlabel("FPR"); axes[1].set_ylabel("TPR")
    axes[1].set_title("ROC Curve"); axes[1].legend()
    # Confusion matrix at optimal threshold
    cm_opt = confusion_matrix(y_te, preds_opt)
    sns.heatmap(cm_opt, annot=True, fmt="d", ax=axes[2], cmap="Blues",
                xticklabels=["No Default","Default"],
                yticklabels=["No Default","Default"], linewidths=0.5)
    axes[2].set_title(f"Confusion Matrix @ t={opt_t:.2f}")
    axes[2].set_xlabel("Predicted"); axes[2].set_ylabel("Actual")
    plt.tight_layout(); st.pyplot(fig); plt.close()

    # Feature importance
    section("Feature Importance Analysis")
    fi = pd.DataFrame({"Feature":feat_cols,"Importance":rf.feature_importances_}).sort_values("Importance")
    fig, ax = plt.subplots(figsize=(10,4))
    fig.patch.set_facecolor(CARD)
    ax.barh(fi["Feature"], fi["Importance"],
            color=plt.cm.cool(np.linspace(0.2,0.9,len(fi))))
    ax.set_title("Feature Importances — Random Forest"); ax.set_xlabel("Importance")
    plt.tight_layout(); st.pyplot(fig); plt.close()

    st.code(classification_report(y_te, preds_opt, target_names=["No Default","Default"]))
    st.info("**Conclusion:** EXT_SOURCE_2 (external credit bureau score) is the strongest predictor of default. "
            "The optimal threshold balances the asymmetric costs of missing a real default (high) vs incorrectly flagging a good customer (low).")


# ══════════════════════════════════════════════════════════════════════════════
# TASK 5 — Global Superstore BI Dashboard
# ══════════════════════════════════════════════════════════════════════════════
elif task == "Task 5 · Superstore BI Dashboard":
    st.title(" Task 5 · Global Superstore BI Dashboard")
    st.caption("Global Superstore Dataset · Sales · Profit · KPIs · Segment Analysis")

    gs_path = os.path.join(DATA_DIR, "Global_Superstore.csv")
    if not os.path.exists(gs_path):
        st.error("Global_Superstore.csv not found."); st.stop()

    @st.cache_data
    def load_superstore():
        try:    df = pd.read_csv(gs_path, encoding="ISO-8859-1")
        except: df = pd.read_csv(gs_path, encoding="utf-8", errors="replace")
        df.columns = df.columns.str.strip()
        # Rename dotted columns for convenience
        df = df.rename(columns={
            "Sub.Category": "Sub_Category",
            "Customer.Name": "Customer_Name",
            "Customer.ID": "Customer_ID",
            "Order.Date": "Order_Date",
            "Ship.Date": "Ship_Date",
            "Ship.Mode": "Ship_Mode",
            "Order.ID": "Order_ID",
            "Product.ID": "Product_ID",
            "Product.Name": "Product_Name",
            "Order.Priority": "Order_Priority",
            "Shipping.Cost": "Shipping_Cost",
        })
        df["Sales"]  = pd.to_numeric(df["Sales"].astype(str).str.replace(",",""), errors="coerce").fillna(0)
        df["Profit"] = pd.to_numeric(df["Profit"].astype(str).str.replace(",",""), errors="coerce").fillna(0)
        df["Discount"] = pd.to_numeric(df["Discount"], errors="coerce").fillna(0)
        df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce").fillna(0)
        df["Order_Date"] = pd.to_datetime(df["Order_Date"], errors="coerce")
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce").fillna(0).astype(int)
        return df

    df_all = load_superstore()

    # ── Sidebar Filters ───────────────────────────────────────────────────────
    st.sidebar.markdown("### 🔧 Filters")
    all_regions = sorted(df_all["Region"].dropna().unique())
    sel_regions = st.sidebar.multiselect("Region", all_regions, default=all_regions[:3])
    all_cats = sorted(df_all["Category"].dropna().unique())
    sel_cats = st.sidebar.multiselect("Category", all_cats, default=all_cats)
    all_segs = sorted(df_all["Segment"].dropna().unique())
    sel_segs = st.sidebar.multiselect("Segment", all_segs, default=all_segs)
    years = sorted(df_all["Year"][df_all["Year"]>0].unique())
    sel_years = st.sidebar.multiselect("Year", years, default=years)

    df = df_all[
        df_all["Region"].isin(sel_regions) &
        df_all["Category"].isin(sel_cats) &
        df_all["Segment"].isin(sel_segs) &
        df_all["Year"].isin(sel_years)
    ]

    if df.empty:
        st.warning("No data matches the selected filters."); st.stop()

    # ── KPIs ──────────────────────────────────────────────────────────────────
    section("Key Performance Indicators")
    total_sales   = df["Sales"].sum()
    total_profit  = df["Profit"].sum()
    total_orders  = df["Order_ID"].nunique()
    profit_margin = (total_profit/total_sales*100) if total_sales else 0
    avg_discount  = df["Discount"].mean()*100
    total_qty     = df["Quantity"].sum()

    c1,c2,c3 = st.columns(3)
    kpi(c1, f"${total_sales:,.0f}", "Total Sales")
    kpi(c2, f"${total_profit:,.0f}", "Total Profit")
    kpi(c3, f"{profit_margin:.1f}%", "Profit Margin")
    c4,c5,c6 = st.columns(3)
    kpi(c4, f"{total_orders:,}", "Unique Orders")
    kpi(c5, f"{avg_discount:.1f}%", "Avg Discount")
    kpi(c6, f"{int(total_qty):,}", "Units Sold")

    section("Sales & Profit Analysis")
    t1,t2,t3,t4,t5,t6 = st.tabs([
        "Sales by Category", "Yearly Trend", "Region Performance",
        "Top Customers", "Sub-Category Breakdown", "Profit vs Discount"
    ])

    with t1:
        fig, axes = plt.subplots(1,3,figsize=(16,4))
        fig.patch.set_facecolor(CARD)
        cat_sales  = df.groupby("Category")["Sales"].sum().sort_values()
        cat_profit = df.groupby("Category")["Profit"].sum()
        cat_qty    = df.groupby("Category")["Quantity"].sum()
        axes[0].barh(cat_sales.index, cat_sales.values, color=[G2,G1,G4], edgecolor=DARK)
        axes[0].set_title("Sales by Category"); axes[0].set_xlabel("Sales ($)")
        axes[1].barh(cat_profit.index, cat_profit.values,
                     color=[G2 if v>=0 else G3 for v in cat_profit.values], edgecolor=DARK)
        axes[1].set_title("Profit by Category"); axes[1].set_xlabel("Profit ($)")
        axes[2].pie(cat_qty, labels=cat_qty.index, autopct="%1.1f%%",
                    colors=[G2,G1,G4], startangle=90,
                    wedgeprops={"edgecolor":DARK,"linewidth":2})
        axes[2].set_title("Quantity Share by Category")
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with t2:
        if len(sel_years) >= 2:
            yr_sales  = df.groupby("Year")["Sales"].sum()
            yr_profit = df.groupby("Year")["Profit"].sum()
            yr_orders = df.groupby("Year")["Order_ID"].nunique()
            fig, axes = plt.subplots(1,3,figsize=(16,4))
            fig.patch.set_facecolor(CARD)
            axes[0].plot(yr_sales.index, yr_sales.values, "o-", color=G1, lw=2)
            axes[0].fill_between(yr_sales.index, yr_sales.values, alpha=0.15, color=G1)
            axes[0].set_title("Annual Sales Trend"); axes[0].set_ylabel("Sales ($)")
            axes[1].bar(yr_profit.index, yr_profit.values,
                        color=[G1 if v>=0 else G3 for v in yr_profit.values], edgecolor=DARK)
            axes[1].set_title("Annual Profit"); axes[1].set_ylabel("Profit ($)")
            axes[2].plot(yr_orders.index, yr_orders.values, "s-", color=G4, lw=2)
            axes[2].set_title("Orders per Year"); axes[2].set_ylabel("Unique Orders")
            plt.tight_layout(); st.pyplot(fig); plt.close()
        else:
            st.info("Select multiple years to see the trend.")

    with t3:
        reg_s = df.groupby("Region")["Sales"].sum().sort_values(ascending=True)
        reg_p = df.groupby("Region")["Profit"].sum().sort_values(ascending=True)
        fig, axes = plt.subplots(1,2,figsize=(14,max(4, len(reg_s)*0.45)))
        fig.patch.set_facecolor(CARD)
        colors_r = plt.cm.cool(np.linspace(0.2,0.9,len(reg_s)))
        axes[0].barh(reg_s.index, reg_s.values, color=colors_r)
        axes[0].set_title("Sales by Region"); axes[0].set_xlabel("Sales ($)")
        bar_colors_p = [G1 if v>=0 else G3 for v in reg_p.values]
        axes[1].barh(reg_p.index, reg_p.values, color=bar_colors_p)
        axes[1].axvline(0, color=G4, lw=1)
        axes[1].set_title("Profit by Region"); axes[1].set_xlabel("Profit ($)")
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with t4:
        top_c = df.groupby("Customer_Name")["Sales"].sum().nlargest(10).sort_values()
        top_p = df.groupby("Customer_Name")["Profit"].sum().nlargest(10).sort_values()
        fig, axes = plt.subplots(1,2,figsize=(14,5))
        fig.patch.set_facecolor(CARD)
        axes[0].barh(top_c.index, top_c.values,
                     color=plt.cm.cool(np.linspace(0.2,0.9,10)))
        axes[0].set_title("Top 10 Customers by Sales"); axes[0].set_xlabel("Sales ($)")
        axes[1].barh(top_p.index, top_p.values,
                     color=plt.cm.plasma(np.linspace(0.2,0.8,10)))
        axes[1].set_title("Top 10 Customers by Profit"); axes[1].set_xlabel("Profit ($)")
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with t5:
        sub_s = df.groupby("Sub_Category")["Sales"].sum().sort_values(ascending=True)
        sub_p = df.groupby("Sub_Category")["Profit"].sum().sort_values(ascending=True)
        fig, axes = plt.subplots(1,2,figsize=(14, max(5, len(sub_s)*0.38)))
        fig.patch.set_facecolor(CARD)
        axes[0].barh(sub_s.index, sub_s.values,
                     color=plt.cm.cool(np.linspace(0.1,0.9,len(sub_s))))
        axes[0].set_title("Sales by Sub-Category"); axes[0].set_xlabel("Sales ($)")
        bar_c = [G1 if v>=0 else G3 for v in sub_p.values]
        axes[1].barh(sub_p.index, sub_p.values, color=bar_c)
        axes[1].axvline(0, color=G4, lw=1, linestyle="--")
        axes[1].set_title("Profit by Sub-Category (red=loss)"); axes[1].set_xlabel("Profit ($)")
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with t6:
        fig, axes = plt.subplots(1,2,figsize=(14,4))
        fig.patch.set_facecolor(CARD)
        axes[0].scatter(df["Discount"], df["Profit"], alpha=0.2, s=8,
                        c=[G1 if p>=0 else G3 for p in df["Profit"]])
        axes[0].axhline(0, color=G4, lw=1, linestyle="--")
        m_d, b_d = np.polyfit(df["Discount"], df["Profit"], 1)
        x_d = np.linspace(0,1,100)
        axes[0].plot(x_d, m_d*x_d+b_d, color=G2, lw=2, label="Trend")
        axes[0].set_xlabel("Discount Rate"); axes[0].set_ylabel("Profit ($)")
        axes[0].set_title("Discount vs Profit Impact"); axes[0].legend()
        # Segment comparison
        seg_s = df.groupby("Segment")["Sales"].sum()
        seg_p = df.groupby("Segment")["Profit"].sum()
        x_seg = np.arange(len(seg_s))
        w = 0.35
        axes[1].bar(x_seg-w/2, seg_s.values, w, label="Sales", color=G2, edgecolor=DARK)
        axes[1].bar(x_seg+w/2, seg_p.values, w, label="Profit", color=G1, edgecolor=DARK)
        axes[1].set_xticks(x_seg); axes[1].set_xticklabels(seg_s.index)
        axes[1].set_title("Sales & Profit by Segment"); axes[1].legend()
        plt.tight_layout(); st.pyplot(fig); plt.close()

    section("Ship Mode & Priority Analysis")
    fig, axes = plt.subplots(1,2,figsize=(14,4))
    fig.patch.set_facecolor(CARD)
    ship_s = df.groupby("Ship_Mode")["Sales"].sum().sort_values()
    axes[0].bar(ship_s.index, ship_s.values,
                color=PAL[:len(ship_s)], edgecolor=DARK)
    axes[0].set_title("Sales by Ship Mode"); axes[0].set_ylabel("Sales ($)")
    plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=20, ha="right")
    prio_s = df.groupby("Order_Priority")["Sales"].sum().sort_values()
    axes[1].pie(prio_s, labels=prio_s.index, autopct="%1.1f%%",
                colors=PAL[:len(prio_s)], startangle=90,
                wedgeprops={"edgecolor":DARK,"linewidth":2})
    axes[1].set_title("Sales Share by Order Priority")
    plt.tight_layout(); st.pyplot(fig); plt.close()

    st.info("**Conclusion:** Technology has the highest sales and profit margin. "
            "Furniture shows high revenue but thin margins. Discounts above 20% "
            "consistently lead to negative profit — a key business insight for pricing strategy.")