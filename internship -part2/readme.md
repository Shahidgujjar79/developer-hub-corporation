🧬 DevelopersHub — Advanced Data Science PortfolioA fully interactive Streamlit web application showcasing solutions for the DevelopersHub Data Science & Analytics internship. This application provides Exploratory Data Analysis (EDA), model training, and live predictive capabilities for five distinct data science use cases.📁 Project StructurePlaintextinternship-part2/
├── app.py                  ← Main Streamlit entry point
├── requirements.txt        ← Python dependencies
├── README.md               ← Project documentation
├── Global_Superstore.csv   ← Dataset for Task 5
├── Mall_Customers.csv      ← Dataset for Task 2
├── application_train.csv   ← Dataset for Task 4
├── bank.csv                ← Dataset for Task 1
└── household_power_consumption.csv ← Dataset for Task 3
🚀 How to Run LocallyClone the repository:Bashgit clone https://github.com/Shahidgujjar79/developer-hub-corporation.git
cd internship-part2
Install dependencies:Bashpip install -r requirements.txt
Run the application:Bashstreamlit run app.py
The app will open automatically in your browser at http://localhost:8501.📋 Tasks CompletedTaskObjectiveMethodologyTask 1Term Deposit PredictionClassification ModelingTask 2Customer SegmentationK-Means Clustering + PCATask 3Energy ForecastingGBM + ARIMATask 4Loan Default RiskThreshold OptimizationTask 5BI DashboardData Visualization🛠️ Tech Stack & FeaturesFramework: Streamlit for the web interface.Data Processing: pandas, numpy.Visualization: matplotlib, seaborn for interactive EDA (scatter plots, heatmaps, box plots).Machine Learning: scikit-learn for training and evaluation.Time Series: statsmodels for forecasting.DevelopersHub Corporation · Data Science & Analytics Internship
