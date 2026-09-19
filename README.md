# AI Salary Prediction

An end-to-end machine learning project that predicts job salaries based on professional and company-related features. The project uses Random Forest Regression with hyperparameter tuning and provides an interactive Streamlit web application for salary prediction.

## 🚀 Project Overview

Salary prediction can help estimate expected compensation based on factors such as job title, experience, education, Number of skills, industry, company size, location, remote-work status and Number of certifications.

This project builds a complete machine learning pipeline from data exploration and preprocessing to model training, evaluation, hyperparameter tuning, and deployment using Streamlit.

## 📊 Dataset

The project uses the **250K Job Salary Prediction Dataset** from Kaggle.

**Dataset:**  
https://www.kaggle.com/datasets/rhythmghai/250k-job-salary-prediction-dataset/data

The dataset contains approximately **250,000 job records** with information about job roles, experience, education, skills, industry, company size, location, remote work, certifications, and salary.

## 🧠 Features Used

The model uses the following features:

- Job Title
- Experience Years
- Education Level
- Skills Count
- Industry
- Company Size
- Location
- Remote Work
- Certifications

### Target Variable

- Salary

## 🔧 Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Joblib
- Streamlit
- Jupyter Notebook

## 🔄 Machine Learning Workflow

```text
Dataset
   ↓
Data Cleaning & Exploration
   ↓
Exploratory Data Analysis
   ↓
Feature & Target Separation
   ↓
Train-Test Split
   ↓
One-Hot Encoding
   ↓
Random Forest Regression
   ↓
Model Evaluation
   ↓
Hyperparameter Tuning
   ↓
Final Model
   ↓
Model & Encoder Serialization
   ↓
Streamlit Web Application
```


## Data Preprocessing

Categorical features were converted into numerical features using One-Hot Encoding (OHE).

The encoder was fitted only on the training data and then used to transform the test data to avoid data leakage.

The dataset was divided into:

80% Training Data
20% Testing Data

## Model

The project uses Random Forest Regressor for salary prediction.

Random Forest was selected because it combines multiple Decision Trees and can capture nonlinear relationships between different job and company features.

## Hyperparameter Tuning

RandomizedSearchCV was used to search for better Random Forest hyperparameters, including:

n_estimators
max_depth
min_samples_split
min_samples_leaf
max_features

Cross-validation was performed on the training data while keeping the test set separate for final evaluation.

## Model Performance

The final tuned Random Forest achieved:

Metric	Score
Training R² =	0.9768
Testing R² = 0.9646

## Streamlit Application

The trained model and encoder are saved using Joblib and loaded into a Streamlit application.

The application allows users to enter:

Job Title
Experience
Education Level
Number of Skills
Industry
Company Size
Location
Remote Work
Number of Certifications

The application then generates a predicted salary based on the trained Random Forest model.

## Key Highlights
250K-row salary dataset
Exploratory Data Analysis
Categorical feature encoding using One-Hot Encoding
Random Forest Regression
Hyperparameter tuning using RandomizedSearchCV
Model evaluation using MAE, RMSE, and R2
Saved ML model and encoder using Joblib
Interactive Streamlit web application
