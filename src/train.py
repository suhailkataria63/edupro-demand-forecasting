from data_loader import load_data
from data_processing import prepare_monthly_data, create_targets
from feature_engineering import build_features

DATA_PATH = "data/raw/EduPro Online Platform.xlsx"

def main():
    courses, teachers, transactions = load_data(DATA_PATH)

    monthly = prepare_monthly_data(courses, teachers, transactions)
    dataset = create_targets(monthly)

    dataset = build_features(dataset)

    # Drop rows where lag features are missing (first few months per course)
    dataset = dataset.dropna(subset=["Enrollment_lag1", "Revenue_lag1"])

    print("Final dataset shape (with lags):", dataset.shape)
    print(dataset[[
        "CourseID", "YearMonth",
        "Enrollment_count", "Revenue",
        "Enrollment_lag1", "Revenue_lag1",
        "Enrollments_next_month", "Revenue_next_month"
    ]].head(10))

if __name__ == "__main__":
    main()