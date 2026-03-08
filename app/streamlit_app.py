import streamlit as st
import pandas as pd
import numpy as np
from src.model_utils import load_model
from src.feature_engineering import build_features
from src.data_processing import prepare_monthly_data, create_targets
from src.data_loader import load_data

# Load the trained model
@st.cache_resource
def load_trained_model():
    return load_model("xgboost_enrollment_model.pkl")

# Load sample data for feature engineering
@st.cache_data
def load_sample_data():
    courses, teachers, transactions = load_data("data/raw/EduPro Online Platform.xlsx")
    monthly = prepare_monthly_data(courses, teachers, transactions)
    dataset = create_targets(monthly)
    dataset = build_features(dataset)
    dataset = dataset.dropna()
    return dataset

st.title("EduPro Enrollment & Revenue Forecasting")

st.markdown("""
This app forecasts next-month course enrollments and revenue using a trained XGBoost model.
Enter the course details and historical data below.
""")

# Load model and sample data
model = load_trained_model()
sample_data = load_sample_data()

# Input form
st.header("Course Information")
course_id = st.text_input("Course ID", value="C001")
course_name = st.text_input("Course Name", value="Sample Course")
course_category = st.selectbox("Course Category", sample_data["CourseCategory"].unique())
course_price = st.number_input("Course Price", min_value=0.0, value=100.0)
course_rating = st.slider("Course Rating", 0.0, 5.0, 4.0)

st.header("Historical Data (Last Month)")
enrollment_count = st.number_input("Current Month Enrollments", min_value=0, value=10)
revenue = st.number_input("Current Month Revenue", min_value=0.0, value=1000.0)

# Lagged features (simplified inputs)
enrollment_lag1 = st.number_input("Enrollments 1 Month Ago", min_value=0, value=8)
enrollment_lag2 = st.number_input("Enrollments 2 Months Ago", min_value=0, value=6)
enrollment_lag3 = st.number_input("Enrollments 3 Months Ago", min_value=0, value=5)

revenue_lag1 = st.number_input("Revenue 1 Month Ago", min_value=0.0, value=800.0)
revenue_lag2 = st.number_input("Revenue 2 Months Ago", min_value=0.0, value=600.0)
revenue_lag3 = st.number_input("Revenue 3 Months Ago", min_value=0.0, value=500.0)

# Category data (simplified)
category_enrollment = st.number_input("Category Total Enrollments (Current Month)", min_value=0, value=100)
category_enrollment_lag1 = st.number_input("Category Enrollments 1 Month Ago", min_value=0, value=90)

# Create a sample row for prediction
if st.button("Predict Next Month"):
    # Create a DataFrame with the inputs
    input_data = pd.DataFrame({
        "CourseID": [course_id],
        "CourseCategory": [course_category],
        "CoursePrice": [course_price],
        "CourseRating": [course_rating],
        "Enrollment_count": [enrollment_count],
        "Revenue": [revenue],
        "Enrollment_lag1": [enrollment_lag1],
        "Enrollment_lag2": [enrollment_lag2],
        "Enrollment_lag3": [enrollment_lag3],
        "Revenue_lag1": [revenue_lag1],
        "Revenue_lag2": [revenue_lag2],
        "Revenue_lag3": [revenue_lag3],
        "Category_Enrollment": [category_enrollment],
        "Category_Enrollment_lag1": [category_enrollment_lag1],
        "YearMonth": [pd.Period("2025-11", freq="M")],  # Dummy
    })

    # Add missing features by copying from sample data or computing
    # For simplicity, merge with a sample row to get all features
    sample_row = sample_data.iloc[0:1].copy()
    for col in input_data.columns:
        if col in sample_row.columns:
            sample_row[col] = input_data[col].values[0]

    # Rebuild features
    sample_row = build_features(sample_row)

    # Select numeric features
    feature_cols = [col for col in sample_row.select_dtypes(include=[np.number]).columns if col not in [
        "Enrollments_next_month", "Revenue_next_month", "Course_Share_next_month", "Category_Enrollment_next_month"
    ]]
    X_pred = sample_row[feature_cols]

    # Predict
    pred_log = model.predict(X_pred)
    pred_enrollment = np.expm1(pred_log)[0]  # Inverse log transform
    pred_revenue = pred_enrollment * course_price

    st.success(f"Predicted Next-Month Enrollments: {pred_enrollment:.1f}")
    st.success(f"Predicted Next-Month Revenue: ${pred_revenue:.2f}")

st.markdown("---")
st.markdown("Note: This is a simplified demo. In production, integrate with real-time data and full feature engineering.")