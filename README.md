🕵️‍♂️ Transactional Fraud Detection Analysis

📅 Project: Month 1 — Data Analytics & Machine Learning Capstone

Objective: Detect fraudulent financial transactions using data analysis and machine learning.

🧠 Project Overview

Financial fraud is a major concern for banks and e-commerce companies, leading to significant financial losses and reputational damage.

This project aims to analyze historical transaction data to uncover fraudulent patterns and build a baseline fraud detection model using machine learning.

By leveraging EDA, feature engineering, and classification modeling, the project identifies suspicious transactions and provides visual insights through a Streamlit dashboard.

🧩 Problem Statement

Financial institutions handle millions of transactions daily, making it difficult to manually detect fraudulent activities.

The challenge is to develop an automated analytical system that can distinguish between legitimate and fraudulent transactions accurately and efficiently.

⚙️ Tech Stack Used

| Category                      | Tools & Technologies                                     |
| ----------------------------- | -------------------------------------------------------- |
| **Programming Language**      | Python                                                   |
| **Libraries**                 | Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Plotly |
| **Database**                  | PostgreSQL *(for optional data extraction)*              |
| **Visualization & Reporting** | Streamlit                                                |
| **Environment**               | Jupyter Notebook, VS Code                                |
| **Deployment**                | Render (Streamlit Cloud Hosting)                         |

🧱 Project Structure

Transactional_Fraud_Detection_Analysis/
│
├── app/
│   ├── streamlit_app.py              # Main Streamlit dashboard app
│   ├── requirements.txt              # Libraries required for deployment
│   └── Procfile                      # Render app startup file
│
├── data/
│   ├── creditcard.csv                # Original dataset
│   ├── clean_creditcard.csv          # Cleaned version (after preprocessing)
│   ├── eda_ready_data.csv            # Data prepared for visualization
│
├── models/
│   └── fraud_pipeline_v1.joblib      # Trained ML model (Logistic Regression)
│
├── notebooks/
│   ├── 01_Data_Preparation.ipynb
│   ├── 02_Exploratory_Data_Analysis.ipynb
│   ├── 03_Model_Training.ipynb
│
├── README.md                         # Project documentation
└── .gitignore

🚀 Project Development Phases

🗓️ Week 1 — Data Preparation & Exploration

Load data from creditcard.csv.

Perform initial profiling using .info() and .describe().

Handle missing values and data inconsistencies.

Analyze class imbalance (fraud vs. non-fraud).

🗓️ Week 2 — Exploratory Data Analysis (EDA)

Univariate and bivariate analysis of transaction features.

Visualize distribution of transaction amounts, time, and frequency using Matplotlib, Seaborn, Plotly.

Identify correlations and anomalies linked to fraud.

Save processed data as eda_ready_data.csv.

🗓️ Week 3 — Feature Engineering & Baseline Modeling

Create new features (e.g., transaction frequency, time since last transaction).

Split data into train-test sets.

Build a baseline Logistic Regression model with scaling and pipeline.

Evaluate model using Precision, Recall, F1-score, ROC-AUC.

Save trained model as fraud_pipeline_v1.joblib.

🗓️ Week 4 — Reporting & Streamlit Dashboard

Develop an interactive Streamlit dashboard (streamlit_app.py).

Display fraud vs. non-fraud distribution, transaction patterns, and key indicators.

Integrate model predictions for test samples.

Deploy the app on Render for live access.

📂 Dataset Sources

| File                   | Description                           | Link                                                                                           |
| ---------------------- | ------------------------------------- | ---------------------------------------------------------------------------------------------- |
| `creditcard.csv`       | Original transaction dataset          | [Download](https://drive.google.com/file/d/18F13C4FhUGn22ZwvuN0w1NYzEWxznYj-/view?usp=sharing) |
| `clean_creditcard.csv` | Cleaned dataset (after preprocessing) | [Download](https://drive.google.com/file/d/1VKx5NhSRvKhZOojgAbfYmgSASg2N0NpJ/view?usp=sharing) |
| `eda_ready_data.csv`   | Dataset prepared for visualizations   | [Download](https://drive.google.com/file/d/1nFSSB-AkT_DpxRK4hKi1Gro0BSq7RtNw/view?usp=sharing) |


💻 How to Run Locally

1️⃣ Clone the repository

git clone https://github.com/pranavb200/Transactional-Fraud-Detection-Analysis.git

cd Transactional-Fraud-Detection-Analysis

2️⃣ Create and activate a virtual environment

python -m venv venv
venv\Scripts\activate   # For Windows
# or
source venv/bin/activate  # For Mac/Linux

3️⃣ Install dependencies
pip install -r app/requirements.txt

4️⃣ Run the Streamlit app
streamlit run app/streamlit_app.py  (inside venv)

☁️ Deployment on Render

Push your project to GitHub.
Go to Render
.
Create a New Web Service.
Connect your GitHub repo.
Select the branch and set these options:

Build Command: pip install -r app/requirements.txt

Start Command: streamlit run app/streamlit_app.py --server.port 10000

Click Deploy.

Render will automatically build and host your Streamlit dashboard online.

🧾 Key Insights

Fraudulent transactions tend to have smaller amounts but higher frequency.

Certain transaction times (e.g., late night) show higher fraud likelihood.

The dataset is highly imbalanced (frauds ≈ 0.17% of total).

Logistic Regression with class weights performs reasonably well as a baseline.

📊 Model Evaluation Summary
| Metric    | Value (approx.) |
| --------- | --------------- |
| Accuracy  | 99.8%           |
| Precision | 0.84            |
| Recall    | 0.91            |
| F1-Score  | 0.87            |
| ROC-AUC   | 0.97            |

(Exact metrics may vary depending on dataset split and preprocessing.)

🧰 Future Enhancements

Implement SMOTE or ADASYN to handle class imbalance.

Experiment with advanced models like Random Forest, XGBoost, or Neural Networks.

Add a real-time API layer for live fraud detection.

Integrate Tableau/Power BI dashboards for executive reporting.

👨‍💻 Author

Pranav B
Data Analyst | Machine Learning Enthusiast
📧 pranavbgowda0@gmail.com
📍 India
