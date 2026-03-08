import pandas as pd
import numpy as np


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
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

    # Safe derived feature: uses only lagged values
    df["Revenue_per_enrollment_lag1"] = (
        df["Revenue_lag1"] / df["Enrollment_lag1"].replace(0, np.nan)
    )

    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["CourseID", "YearMonth"])

    # rolling features must not look into the future; use shift(1) then rolling
    df["Enrollment_roll3"] = (
        df.groupby("CourseID")["Enrollment_count"]
        .shift(1)
        .rolling(3)
        .mean()
        .reset_index(level=0, drop=True)
    )

    df["Revenue_roll3"] = (
        df.groupby("CourseID")["Revenue"]
        .shift(1)
        .rolling(3)
        .mean()
        .reset_index(level=0, drop=True)
    )

    return df


def add_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # uses only lag values (safe)
    df["Enrollment_trend"] = df["Enrollment_lag1"] - df["Enrollment_lag2"]
    df["Revenue_trend"] = df["Revenue_lag1"] - df["Revenue_lag2"]
    return df


def add_global_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

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
    df = df.sort_values(["YearMonth", "CourseID"])

    df["Global_Enrollment_lag1"] = df["Global_Enrollment"].shift(1)
    df["Global_Enrollment_roll3"] = df["Global_Enrollment"].shift(1).rolling(3).mean()

    return df


def add_category_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Ensure Category_Enrollment exists
    if "Category_Enrollment" not in df.columns:
        cat_monthly = (
            df.groupby(["CourseCategory", "YearMonth"])["Enrollment_count"]
            .sum()
            .reset_index()
            .rename(columns={"Enrollment_count": "Category_Enrollment"})
        )
        df = df.merge(cat_monthly, on=["CourseCategory", "YearMonth"], how="left")

    df = df.sort_values(["CourseCategory", "YearMonth", "CourseID"])

    # Safe: lag + past rolling
    df["Category_Enrollment_lag1"] = df.groupby("CourseCategory")["Category_Enrollment"].shift(1)
    df["Category_Enrollment_roll3"] = (
        df.groupby("CourseCategory")["Category_Enrollment"]
        .shift(1)
        .rolling(3)
        .mean()
        .reset_index(level=0, drop=True)
    )
    # ===== Momentum Features =====
    df["Enrollment_momentum_1"] = df["Enrollment_lag1"] - df["Enrollment_lag2"]
    df["Enrollment_momentum_2"] = df["Enrollment_lag2"] - df["Enrollment_lag3"]

    # Acceleration (change of change)
    df["Enrollment_acceleration"] = (
        df["Enrollment_momentum_1"] - df["Enrollment_momentum_2"]
    )

    # Growth Ratio (stability safe)
    df["Enrollment_growth_ratio"] = (
        df["Enrollment_lag1"] / (df["Enrollment_lag2"] + 1e-6)
    )

    # ===== Volatility Feature =====
    df["Enrollment_volatility_3"] = (
    df.groupby("CourseID")["Enrollment_count"]
      .transform(lambda x: x.shift(1).rolling(3).std())
    )
    # ===== Cyclical Month Encoding =====
    import numpy as np

    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)


    
    return df



def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_time_features(df)
    df = add_lag_features(df, lags=(1, 2, 3))
    df = add_rolling_features(df)
    df = add_trend_features(df)
    df = add_global_features(df)
    df = add_category_features(df)
    return df