from data_loader import load_data
from data_processing import prepare_monthly_data, create_targets
from feature_engineering import build_features

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

DATA_PATH = "data/raw/EduPro Online Platform.xlsx"

def time_split(df):

    df = df.sort_values("YearMonth")

    train = df[df["YearMonth"] <= "2025-08"]
    val = df[(df["YearMonth"] >= "2025-09") & (df["YearMonth"] <= "2025-10")]
    test = df[df["YearMonth"] == "2025-11"]

    return train, val, test


def evaluate(y_true, y_pred, label="Model"):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    print(f"\n{label} Performance:")
    print("MAE:", round(mae, 3))
    print("RMSE:", round(rmse, 3))
    print("MAPE:", round(mape, 2), "%")

def main():

    courses, teachers, transactions = load_data(DATA_PATH)

    monthly = prepare_monthly_data(courses, teachers, transactions)
    dataset = create_targets(monthly)
    dataset = build_features(dataset)

    dataset = dataset.dropna()

    train, val, test = time_split(dataset)

    print("Train shape:", train.shape)
    print("Validation shape:", val.shape)
    print("Test shape:", test.shape)

    # Baseline: predict next month = last month (lag1)
    y_test = test["Enrollments_next_month"]
    y_pred_baseline = test["Enrollment_lag1"]

    evaluate(y_test, y_pred_baseline, label="Naive Baseline (Enrollment)")


if __name__ == "__main__":
    main()