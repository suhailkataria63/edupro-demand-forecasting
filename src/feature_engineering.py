import pandas as pd
import numpy as np


def add_course_static_features(df: pd.DataFrame) -> pd.DataFrame:
    """Bucket static course features per mentor spec."""
    df = df.copy()

    # price bands: low/medium/high via terciles
    if "CoursePrice" in df.columns:
        try:
            df["price_band"] = pd.qcut(df["CoursePrice"], q=3, labels=[0, 1, 2], duplicates="drop")
        except ValueError:
            # fallback to simple equal-width bins
            df["price_band"] = pd.cut(df["CoursePrice"], bins=3, labels=[0,1,2])

    # duration buckets: short/medium/long via tertiles
    if "CourseDuration" in df.columns:
        try:
            df["duration_bucket"] = pd.qcut(df["CourseDuration"], q=3, labels=[0, 1, 2], duplicates="drop")
        except ValueError:
            df["duration_bucket"] = pd.cut(df["CourseDuration"], bins=3, labels=[0,1,2])

    # rating tiers: low/med/high
    if "CourseRating" in df.columns:
        df["rating_tier"] = pd.cut(df["CourseRating"], bins=[-np.inf, 2, 4, np.inf], labels=[0, 1, 2])

    # course level encoding (ordinal)
    level_map = {"Beginner": 0, "Intermediate": 1, "Advanced": 2}
    if "CourseLevel" in df.columns:
        df["course_level_enc"] = df["CourseLevel"].map(level_map).fillna(-1)

    return df


def add_instructor_features(df: pd.DataFrame) -> pd.DataFrame:
    """Placeholder for instructor info; dataset lacks mapping so use defaults."""
    df = df.copy()
    df["instr_experience_bucket"] = 0
    df["instr_rating_score"] = 0.0
    df["instr_expertise_match"] = 0
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ym_ts = df["YearMonth"].dt.to_timestamp()
    df["year"] = ym_ts.dt.year
    df["month"] = ym_ts.dt.month
    df["quarter"] = ym_ts.dt.quarter
    df["is_q_start"] = df["month"].isin([1, 4, 7, 10]).astype(int)
    return df


def add_lag_features(df: pd.DataFrame, lags=(1, 2, 3)) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["CourseID", "YearMonth"])
    for lag in lags:
        df[f"Enrollment_lag{lag}"] = df.groupby("CourseID")["Enrollment_count"].shift(lag)
        df[f"Revenue_lag{lag}"] = df.groupby("CourseID")["Revenue"].shift(lag)
    df["Revenue_per_enrollment_lag1"] = (
        df["Revenue_lag1"] / df["Enrollment_lag1"].replace(0, np.nan)
    )
    return df


def add_historical_summary(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["CourseID", "YearMonth"])
    df["Enroll_mean_3"] = df.groupby("CourseID")["Enrollment_count"].transform(
        lambda s: s.shift(1).rolling(3).mean()
    )
    df["Rev_mean_3"] = df.groupby("CourseID")["Revenue"].transform(
        lambda s: s.shift(1).rolling(3).mean()
    )
    df["RevperEnroll_mean_3"] = (
        df["Rev_mean_3"] / df["Enroll_mean_3"].replace(0, np.nan)
    )
    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["CourseID", "YearMonth"])
    df["Enrollment_roll3"] = df.groupby("CourseID")["Enrollment_count"].transform(
        lambda s: s.shift(1).rolling(3).mean()
    )
    df["Revenue_roll3"] = df.groupby("CourseID")["Revenue"].transform(
        lambda s: s.shift(1).rolling(3).mean()
    )
    return df


def add_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Enrollment_trend"] = df["Enrollment_lag1"] - df["Enrollment_lag2"]
    df["Revenue_trend"] = df["Revenue_lag1"] - df["Revenue_lag2"]
    return df


def add_category_lags(df: pd.DataFrame) -> pd.DataFrame:
    """Add lagged category totals which might capture shifting demand."""
    df = df.copy()
    df = df.sort_values(["CourseCategory", "YearMonth"])
    df["cat_enroll_lag1"] = df.groupby("CourseCategory")["Category_Enrollment"].shift(1)
    df["cat_enroll_lag2"] = df.groupby("CourseCategory")["Category_Enrollment"].shift(2)
    # revenue per category isn't directly stored but we could derive it if needed
    # (could be added if `Category_Revenue` were retained)
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_course_static_features(df)
    df = add_instructor_features(df)
    df = add_time_features(df)
    df = add_lag_features(df, lags=(1, 2, 3))
    df = add_historical_summary(df)
    df = add_rolling_features(df)
    df = add_trend_features(df)
    df = add_category_lags(df)
    # global/category omitted per mentor guidance
    return df
