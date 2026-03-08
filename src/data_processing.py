import pandas as pd
import numpy as np

def prepare_monthly_data(courses, teachers, transactions):
    transactions['TransactionDate'] = pd.to_datetime(transactions['TransactionDate'])
    transactions['YearMonth'] = transactions['TransactionDate'].dt.to_period('M')

    monthly = transactions.groupby(['CourseID', 'YearMonth']).agg(
        Enrollment_count=('TransactionID', 'count'),
        Revenue=('Amount', 'sum')
    ).reset_index()

    # Merge course info (teacher mapping not available in dataset)
    monthly = monthly.merge(courses, on='CourseID', how='left')

    return monthly


def create_targets(df):
    df = df.copy()

    # Sort for course-level targets
    df = df.sort_values(["CourseID", "YearMonth"])

    # Course-level next month targets
    df["Enrollments_next_month"] = df.groupby("CourseID")["Enrollment_count"].shift(-1)
    df["Revenue_next_month"] = df.groupby("CourseID")["Revenue"].shift(-1)

    # ----- Category totals per month (ground truth) -----
    cat_monthly = (
        df.groupby(["CourseCategory", "YearMonth"])["Enrollment_count"]
        .sum()
        .reset_index()
        .rename(columns={"Enrollment_count": "Category_Enrollment"})
    )
    cat_monthly = cat_monthly.sort_values(["CourseCategory", "YearMonth"])
    cat_monthly["Category_Enrollment_next_month"] = (
        cat_monthly.groupby("CourseCategory")["Category_Enrollment"].shift(-1)
    )
    df = df.merge(cat_monthly, on=["CourseCategory", "YearMonth"], how="left")

    # ----- Category revenue per month -----
    cat_revenue_monthly = (
        df.groupby(["CourseCategory", "YearMonth"])["Revenue"]
        .sum()
        .reset_index()
        .rename(columns={"Revenue": "Category_Revenue"})
    )
    cat_revenue_monthly = cat_revenue_monthly.sort_values(["CourseCategory", "YearMonth"])
    cat_revenue_monthly["Category_Revenue_next_month"] = (
        cat_revenue_monthly.groupby("CourseCategory")["Category_Revenue"].shift(-1)
    )
    df = df.merge(cat_revenue_monthly, on=["CourseCategory", "YearMonth"], how="left")

    # Drop current Category_Revenue as it's not needed for features (target is computed)
    df = df.drop(columns=["Category_Revenue"])

    # ----- Course share (current month) -----
    df = df.sort_values(["CourseID", "YearMonth"])
    df["Course_Share"] = df["Enrollment_count"] / df["Category_Enrollment"].replace(0, np.nan)
    df["Course_Share"] = df["Course_Share"].fillna(0)

    # Course share lag (previous month) — SAFE feature
    df["Course_Share_lag1"] = df.groupby("CourseID")["Course_Share"].shift(1)

    # Course share next month (target)
    df["Course_Share_next_month"] = df.groupby("CourseID")["Course_Share"].shift(-1)

    # Drop rows where any required target is missing
    df = df.dropna(subset=[
        "Enrollments_next_month",
        "Revenue_next_month"
        ])

    return df
