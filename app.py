import streamlit as st

st.set_page_config(
    page_title="Data Science Portal",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }

    .stApp {
        background-color: #0f1117;
        color: #e8eaf0;
    }

    section[data-testid="stSidebar"] {
        background-color: #161b27;
        border-right: 1px solid #2a3040;
    }

    .main-header {
        background: linear-gradient(135deg, #1a1f2e 0%, #0f1117 100%);
        border: 1px solid #2a3040;
        border-radius: 12px;
        padding: 2rem 2.5rem;
        margin-bottom: 2rem;
    }

    .main-header h1 {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 2rem;
        font-weight: 600;
        color: #64ffda;
        margin: 0 0 0.5rem 0;
        letter-spacing: -0.5px;
    }

    .main-header p {
        color: #8892a4;
        font-size: 0.95rem;
        margin: 0;
    }

    .task-card {
        background: #161b27;
        border: 1px solid #2a3040;
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 0.75rem;
        transition: border-color 0.2s;
    }

    .task-card:hover {
        border-color: #64ffda;
    }

    .badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.72rem;
        font-family: 'IBM Plex Mono', monospace;
        font-weight: 600;
        margin-left: 8px;
        vertical-align: middle;
    }

    .badge-clf { background: #1a3a4a; color: #64ffda; }
    .badge-reg { background: #2d1f3d; color: #c792ea; }
    .badge-eda { background: #1f3320; color: #80cfa9; }

    div[data-testid="stMetricValue"] {
        font-family: 'IBM Plex Mono', monospace;
        color: #64ffda !important;
        font-size: 1.6rem !important;
    }

    .stButton > button {
        background: linear-gradient(135deg, #64ffda, #00bfa5);
        color: #0f1117;
        border: none;
        border-radius: 8px;
        font-weight: 700;
        font-family: 'IBM Plex Sans', sans-serif;
        letter-spacing: 0.3px;
        padding: 0.6rem 1.5rem;
        transition: opacity 0.2s;
    }

    .stButton > button:hover {
        opacity: 0.88;
        color: #0f1117;
    }

    .stTabs [data-baseweb="tab"] {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.8rem;
        color: #8892a4;
    }

    .stTabs [aria-selected="true"] {
        color: #64ffda !important;
    }

    h2, h3, h4 {
        color: #ccd6f6;
        font-family: 'IBM Plex Sans', sans-serif;
    }

    .stSuccess { background-color: #0d2b22 !important; border-color: #64ffda !important; }
    .stError   { background-color: #2b0d0d !important; }
    .stWarning { background-color: #2b1f0d !important; }
    .stInfo    { background-color: #0d1a2b !important; }

    .sidebar-title {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.75rem;
        color: #64ffda;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 1rem;
    }

    /* matplotlib figures dark bg */
    .stImage img { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "tasks"))

from tasks import task1_iris, task2_credit, task3_churn, task4_insurance, task5_loan

# Sidebar nav
with st.sidebar:
    st.markdown('<p class="sidebar-title">📁 DevelopersHub Corp</p>', unsafe_allow_html=True)
    st.markdown("**Data Science Internship**")
    st.markdown("*Due: 15th May 2026*")
    st.divider()
    st.markdown('<p class="sidebar-title">Navigate</p>', unsafe_allow_html=True)

    task = st.radio(
        "Select Task",
        options=[
            "🏠  Overview",
            "🌸  Task 1 — Iris Classification",
            "💰  Task 2 — Credit Risk",
            "🏦  Task 3 — Customer Churn",
            "🏥  Task 4 — Insurance Cost",
            "💳  Task 5 — Loan Acceptance",
        ],
        label_visibility="collapsed"
    )

    st.divider()
    st.markdown("""
    <div style="font-size:0.78rem; color:#8892a4; line-height:1.7">
    <b style="color:#ccd6f6">Stack</b><br>
    Python · scikit-learn<br>
    pandas · seaborn<br>
    matplotlib · Streamlit<br><br>
    <b style="color:#ccd6f6">Models used</b><br>
    Random Forest<br>
    Logistic Regression<br>
    Decision Tree<br>
    Gradient Boosting<br>
    Linear Regression
    </div>
    """, unsafe_allow_html=True)

# ── Pages ──────────────────────────────────────────────────

if "Overview" in task:
    st.markdown("""
    <div class="main-header">
        <h1>⬡ Data Science Portal</h1>
        <p>DevelopersHub Corporation · Analytics Internship · 5 end-to-end ML tasks</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📋 Project Overview")
    st.markdown("""
    This portal contains **5 complete data science tasks** — each with full EDA,
    trained machine learning models, and a live prediction interface.
    Pick a task from the sidebar to get started.
    """)

    tasks_info = [
        ("🌸", "Task 1", "Iris Species Classification",
         "EDA + visualization on the Iris dataset. Scatter, histogram, box plots. RF classifier.",
         "EDA + Classification", "#64ffda"),
        ("💰", "Task 2", "Credit Risk / Loan Default",
         "Predict loan approval from applicant profile. Logistic Regression + Decision Tree.",
         "Binary Classification", "#c792ea"),
        ("🏦", "Task 3", "Customer Churn Prediction",
         "Identify bank customers likely to churn. Random Forest + feature importance.",
         "Binary Classification", "#c792ea"),
        ("🏥", "Task 4", "Insurance Cost Prediction",
         "Estimate annual medical insurance charges. Gradient Boosting Regressor.",
         "Regression", "#ffb86c"),
        ("💳", "Task 5", "Personal Loan Acceptance",
         "Predict whether a customer will accept a loan offer. RF + Logistic Regression.",
         "Binary Classification", "#c792ea"),
    ]

    for icon, num, title, desc, label, color in tasks_info:
        st.markdown(f"""
        <div class="task-card">
            <span style="font-size:1.4rem">{icon}</span>
            <span style="color:#8892a4; font-size:0.8rem; font-family:'IBM Plex Mono',monospace"> {num}</span>
            <span style="font-weight:700; font-size:1.05rem; color:#ccd6f6; margin-left:6px">{title}</span>
            <span class="badge" style="background:#1a2030; color:{color}">{label}</span>
            <p style="color:#8892a4; font-size:0.88rem; margin:0.5rem 0 0 0">{desc}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🚀 Quick Start")
    st.code("""
# Install dependencies
pip install streamlit pandas numpy matplotlib seaborn scikit-learn

# Run the app
streamlit run app.py
    """, language="bash")

    st.markdown("### 📦 Dataset Sources")
    data = {
        "Task": ["Task 1 — Iris", "Task 2 — Credit Risk", "Task 3 — Churn",
                 "Task 4 — Insurance", "Task 5 — Loan"],
        "Dataset": ["Iris Dataset", "Loan Prediction Dataset", "Churn Modelling",
                    "Medical Cost Personal", "Bank Personal Loan Modelling"],
        "Source": ["seaborn built-in", "Kaggle", "Kaggle", "Kaggle", "Kaggle"],
        "File": ["auto-loaded", "train.csv", "Churn_Modelling.csv",
                 "insurance.csv", "Bank_Personal_Loan_Modelling.csv"],
    }
    import pandas as pd
    st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)

elif "Task 1" in task:
    task1_iris.run()

elif "Task 2" in task:
    task2_credit.run()

elif "Task 3" in task:
    task3_churn.run()

elif "Task 4" in task:
    task4_insurance.run()

elif "Task 5" in task:
    task5_loan.run()
