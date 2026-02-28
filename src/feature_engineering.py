import pandas as pd

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # YearMonth is a Period[M]; convert to timestamp for extracting month/quarter
    ym_ts = df["YearMonth"].dt.to_timestamp()
    df["month"] = ym_ts.dt.month
    df["quarter"] = ym_ts.dt.quarter
    return df

def add_lag_features(df: pd.DataFrame, lags=(1, 2, 3)) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["CourseID", "YearMonth"])

    for lag in lags:
        df[f"Enrollment_lag{lag}"] = df.groupby("CourseID")["Enrollment_count"].shift(lag)
        df[f"Revenue_lag{lag}"] = df.groupby("CourseID")["Revenue"].shift(lag)

    # Safe derived feature (uses only past values)
    df["Revenue_per_enrollment_lag1"] = df["Revenue_lag1"] / df["Enrollment_lag1"].replace(0, pd.NA)

    return df
def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["CourseID", "YearMonth"])

    df["Enrollment_roll3"] = (
        df.groupby("CourseID")["Enrollment_count"]
        .rolling(3)
        .mean()
        .reset_index(level=0, drop=True)
    )

    df["Revenue_roll3"] = (
        df.groupby("CourseID")["Revenue"]
        .rolling(3)
        .mean()
        .reset_index(level=0, drop=True)
    )

    return df


def add_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Enrollment_trend"] = df["Enrollment_lag1"] - df["Enrollment_lag2"]
    df["Revenue_trend"] = df["Revenue_lag1"] - df["Revenue_lag2"]
    return df
def add_global_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Compute global monthly averages
    global_monthly = (
        df.groupby("YearMonth")[["Enrollment_count", "Revenue"]]
        .mean()
        .reset_index()
        .rename(columns={
            "Enrollment_count": "Global_Enrollment",
            "Revenue": "Global_Revenue"
        })
    )

    df = df.merge(global_monthly, on="YearMonth", how="left")

    # Sort properly
    df = df.sort_values(["YearMonth"])

    # Add lag
    df["Global_Enrollment_lag1"] = df["Global_Enrollment"].shift(1)

    # Rolling
    df["Global_Enrollment_roll3"] = (
        df["Global_Enrollment"]
        .rolling(3)
        .mean()
    )

    return df
def add_category_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Category totals per month (macro demand inside category)
    cat_monthly = (
        df.groupby(["CourseCategory", "YearMonth"])[["Enrollment_count", "Revenue"]]
        .sum()
        .reset_index()
        .rename(columns={
            "Enrollment_count": "Category_Enrollment",
            "Revenue": "Category_Revenue"
        })
    )

    df = df.merge(cat_monthly, on=["CourseCategory", "YearMonth"], how="left")
    df = df.sort_values(["CourseCategory", "YearMonth", "CourseID"])

    # Lagged category features (avoid future leakage)
    df["Category_Enrollment_lag1"] = df.groupby("CourseCategory")["Category_Enrollment"].shift(1)
    df["Category_Revenue_lag1"] = df.groupby("CourseCategory")["Category_Revenue"].shift(1)

    # Rolling (3-month) category trends
    df["Category_Enrollment_roll3"] = (
        df.groupby("CourseCategory")["Category_Enrollment"]
        .rolling(3)
        .mean()
        .reset_index(level=0, drop=True)
    )

    df["Category_Revenue_roll3"] = (
        df.groupby("CourseCategory")["Category_Revenue"]
        .rolling(3)
        .mean()
        .reset_index(level=0, drop=True)
    )

    return df

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_time_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_trend_features(df)
    df = add_global_features(df)
    df= add_category_features(df)
    return df