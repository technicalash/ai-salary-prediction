import streamlit as st
import pandas as pd
import joblib

model = joblib.load("salary_prediction_model.pkl")
encoder = joblib.load("salary_encoder.pkl")


st.set_page_config(
    page_title="Salary Prediction System",
    page_icon="💼",
    layout="centered"
)
st.title("Salary Prediction System")
st.write("Enter the job details to predict the expected salary.")

job_title = st.selectbox(
    "Job Title",
    [
        "AI Engineer",
        "Data Analyst",
        "Frontend Developer",
        "Business Analyst",
        "Product Manager",
        "Backend Developer",
        "Machine Learning Engineer",
        "DevOps Engineer",
        "Software Engineer",
        "Cybersecurity Analyst",
        "Data Scientist",
        "Cloud Engineer"
    ]
)

experience_years = st.number_input(
    "Experience (Years)",
    min_value=0.0,
    max_value=50.0,
    value=2.0,
    step=0.5
)

education_level = st.selectbox(
    "Education Level",
    [
        "Bachelor",
        "PhD",
        "High School",
        "Diploma",
        "Master"
    ]
)

skills_count = st.number_input(
    "Number of Skills",
    min_value=0,
    max_value=50,
    value=5,
    step=1
)

industry = st.selectbox(
    "Industry",
    [
        "Healthcare",
        "Telecom",
        "Media",
        "Retail",
        "Manufacturing",
        "Education",
        "Finance",
        "Technology",
        "Consulting",
        "Government"
    ]
)

company_size = st.selectbox(
    "Company Size",
    [
        "Medium",
        "Small",
        "Large",
        "Enterprise",
        "Startup"
    ]
)

location = st.selectbox(
    "Location",
    [
        "India",
        "Australia",
        "Singapore",
        "Canada",
        "Sweden",
        "USA",
        "Netherlands",
        "Remote",
        "Germany",
        "UK"
    ]
)

remote_work = st.selectbox(
    "Remote Work",
    [
        "Hybrid",
        "No",
        "Yes"
    ]
)

certifications = st.number_input(
    "Number of Certifications",
    min_value=0,
    max_value=20,
    value=1,
    step=1
)

# prediction 
if st.button("Predict Salary"):

    # Create input DataFrame
    input_data = pd.DataFrame({
        "job_title": [job_title],
        "experience_years": [experience_years],
        "education_level": [education_level],
        "skills_count": [skills_count],
        "industry": [industry],
        "company_size": [company_size],
        "location": [location],
        "remote_work": [remote_work],
        "certifications": [certifications]
    })
    
    categorical_cols = [
        "job_title",
        "education_level",
        "industry",
        "company_size",
        "location",
        "remote_work"
    ]
    
    numerical_cols = [
        "experience_years",
        "skills_count",
        "certifications"
    ]
    
    encoded_data = encoder.transform(input_data[categorical_cols])
    encoded_df = pd.DataFrame(
        encoded_data,
        columns=encoder.get_feature_names_out(categorical_cols)
    )
    final_input = pd.concat(
        [
            input_data[numerical_cols].reset_index(drop=True),
            encoded_df.reset_index(drop=True)
        ],
        axis=1
    )
    
    prediction = model.predict(final_input)[0]
    st.success(
        f"### Predicted Salary: ₹{prediction:,.2f}"
    )