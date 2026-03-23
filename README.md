# 💳 Credit Risk Assessment Model

The project is a machine learning project aimed at predicting the creditworthiness of small businesses, with the dataset being the SBA loan dataset. The project is aimed at helping financial institutions make informed decisions when it comes to lending to small businesses.

## 🚀 Overview

This project involves building a **Credit Risk Assessment Model** to evaluate whether a small business is likely to default on a loan. By leveraging classification techniques and analyzing historical loan data, the model helps improve **loan portfolio management** in banking systems.

The approach combines **data preprocessing, feature engineering, and machine learning algorithms** to deliver reliable predictions of high-risk borrowers.

## 📊 Key Features

- **Exploratory Data Analysis (EDA)**  
  Performed in-depth analysis to understand data distribution, correlations, and key influencing factors.

- **Data Cleaning & Preprocessing**  
  - Handled missing values  
  - Treated outliers  
  - Ensured consistency across features  

- **Feature Engineering**  
  Created new meaningful features to improve model performance and capture hidden patterns.

- **Model Development**  
  Implemented multiple classification algorithms to identify the best-performing model.

- **Model Evaluation**  
  Evaluated models using standard performance metrics:
  - Accuracy  
  - Precision  
  - Recall  
  - F1 Score  

## 🧠 Machine Learning Models Used

- **Logistic Regression**  
- **Decision Tree**  
- **Random Forest**  

## 🛠️ Tech Stack

- **Python** – Core programming language  
- **Pandas & NumPy** – Data manipulation and numerical computation  
- **Matplotlib** – Data visualization  
- **Scikit-learn** – Machine learning models and evaluation  
- **Jupyter Notebook** – Development and experimentation  

## 📂 Dataset

The project uses the **SBA (Small Business Administration) Loan Dataset**, which contains:

- Loan details of small businesses  
- Applicant financial information  
- Loan status (paid / default)  

This dataset enables the model to learn patterns associated with loan repayment behavior.

## 📈 Results

- The final model is able to **effectively classify high-risk and low-risk borrowers**  
- **Logistic Regression** provided more consistent and reliable predictions compared to other models  
- The system can assist banks in:
- The system can assist banks in:
  - Reducing default rates  
  - Improving credit decision-making  
  - Managing loan portfolios more efficiently  

## ⚙️ Installation & Setup

1. Clone the repository  
```bash
git clone https://github.com/AkshatKumar10/Credit-Risk-Analysis.git
cd Credit-Risk-Analysis
```

2. Setup for Model Development (Jupyter Notebook)
```bash
cd 'Other Files'
pip install -r requirements.txt
```

3. Run the Streamlit Web App
```bash
cd ../project_code
pip install -r requirements.txt
streamlit run streamlit_app.py
```
