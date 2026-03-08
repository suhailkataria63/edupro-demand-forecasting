from data_loader import load_data
from data_processing import prepare_monthly_data, create_targets
from feature_engineering import build_features
from model_utils import save_model, save_metadata

from datetime import datetime, timezone
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_PATH = "data/raw/EduPro Online Platform.xlsx"
TRAIN_END = "2025-08"
VAL_START = "2025-09"
VAL_END = "2025-10"
TEST_MONTH = "2025-11"

PARAM_GRID = [
    {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 2, "subsample": 0.9, "colsample_bytree": 0.9},
    {"n_estimators": 400, "learning_rate": 0.03, "max_depth": 3, "subsample": 0.9, "colsample_bytree": 0.9},
    {"n_estimators": 500, "learning_rate": 0.05, "max_depth": 3, "subsample": 1.0, "colsample_bytree": 1.0},
    {"n_estimators": 800, "learning_rate": 0.03, "max_depth": 2, "subsample": 0.9, "colsample_bytree": 0.9},
]

ENROLLMENT_BLEND_ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
MIN_CV_TRAIN_MONTHS = 3


def time_split(df: pd.DataFrame):
    df = df.sort_values("YearMonth")
    train = df[df["YearMonth"] <= TRAIN_END]
    val = df[(df["YearMonth"] >= VAL_START) & (df["YearMonth"] <= VAL_END)]
    test = df[df["YearMonth"] == TEST_MONTH]
    return train, val, test


def safe_mape(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    y_true_safe = np.where(y_true == 0, np.nan, y_true)
    return np.nanmean(np.abs((y_true - y_pred) / y_true_safe)) * 100


def evaluate(y_true, y_pred, label="Model"):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = safe_mape(y_true, y_pred)

    print(f"\n{label} Performance:")
    print("MAE:", round(mae, 3))
    print("RMSE:", round(rmse, 3))
    print("R2:", round(r2, 4))
    print("MAPE:", round(mape, 2), "%")

    return {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape}


def transform_target(y_raw, use_log_target):
    y_raw = np.array(y_raw, dtype=float)
    if use_log_target:
        return np.log1p(y_raw)
    return y_raw


def inverse_target(pred_model_space, use_log_target):
    pred_model_space = np.array(pred_model_space, dtype=float)
    if use_log_target:
        return np.expm1(pred_model_space)
    return pred_model_space


def build_X_y(df: pd.DataFrame, target_col: str):
    drop_cols = {"CourseID", "CourseCategory", "YearMonth", target_col}
    drop_cols.update([c for c in df.columns if c.endswith("_next_month") and c != target_col])

    X = df.drop(columns=[c for c in drop_cols if c in df.columns]).select_dtypes(include=["number"])
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    y = df[target_col].values.astype(float)
    return X, y


def tune_xgboost(train, val, target_col, use_log_target=True):
    X_train, y_train = build_X_y(train, target_col)
    X_val, y_val = build_X_y(val, target_col)

    X_val = X_val.reindex(columns=X_train.columns, fill_value=0)
    y_train_model = transform_target(y_train, use_log_target)

    best = {
        "model": None,
        "params": None,
        "val_mape": float("inf"),
        "val_pred": None,
        "feature_columns": X_train.columns.tolist(),
    }

    for params in PARAM_GRID:
        model = xgb.XGBRegressor(
            objective="reg:squarederror",
            random_state=42,
            tree_method="hist",
            n_jobs=4,
            **params,
        )
        model.fit(X_train, y_train_model, verbose=False)

        pred_val = inverse_target(model.predict(X_val), use_log_target)
        pred_val = np.clip(pred_val, 0, None)
        val_mape = safe_mape(y_val, pred_val)

        if val_mape < best["val_mape"]:
            best.update(
                {
                    "model": model,
                    "params": params,
                    "val_mape": val_mape,
                    "val_pred": pred_val,
                }
            )

    return best


def fit_final_xgboost(train, val, test, target_col, params, use_log_target=True):
    X_train, y_train = build_X_y(train, target_col)
    X_val, y_val = build_X_y(val, target_col)
    X_test, y_test = build_X_y(test, target_col)

    X_dev = pd.concat([X_train, X_val], axis=0, ignore_index=True)
    y_dev = np.concatenate([y_train, y_val], axis=0)
    X_test = X_test.reindex(columns=X_dev.columns, fill_value=0)

    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        tree_method="hist",
        n_jobs=4,
        **params,
    )
    model.fit(X_dev, transform_target(y_dev, use_log_target), verbose=False)

    pred_test = inverse_target(model.predict(X_test), use_log_target)
    pred_test = np.clip(pred_test, 0, None)
    return model, y_test, pred_test


def make_expanding_time_folds(df: pd.DataFrame, min_train_months=MIN_CV_TRAIN_MONTHS):
    """Expanding-window folds over YearMonth for strict time-ordered CV."""
    months = sorted(df["YearMonth"].unique())
    folds = []

    for idx in range(min_train_months, len(months)):
        val_month = months[idx]
        train_months = months[:idx]
        train_fold = df[df["YearMonth"].isin(train_months)]
        val_fold = df[df["YearMonth"] == val_month]

        if train_fold.empty or val_fold.empty:
            continue
        folds.append((train_fold, val_fold, val_month))

    return folds


def tune_enrollment_timeseries_cv(dev_df: pd.DataFrame, use_log_target=True):
    """
    Tune enrollment model using expanding-window CV.
    Uses recency-weighted fold MAPE to prioritize nearer-term generalization.
    """
    folds = make_expanding_time_folds(dev_df, min_train_months=MIN_CV_TRAIN_MONTHS)
    if not folds:
        raise ValueError("No valid time-series CV folds could be created for enrollment tuning.")

    fold_weights = np.arange(1, len(folds) + 1, dtype=float)
    fold_weights = fold_weights / fold_weights.sum()

    best = {
        "params": None,
        "blend_alpha": 1.0,
        "cv_weighted_mape": float("inf"),
        "cv_mean_mape": float("inf"),
        "fold_months": [str(month) for _, _, month in folds],
        "fold_mapes": None,
    }

    for params in PARAM_GRID:
        for alpha in ENROLLMENT_BLEND_ALPHAS:
            fold_mapes = []

            for train_fold, val_fold, _ in folds:
                X_train, y_train = build_X_y(train_fold, "Enrollments_next_month")
                X_val, y_val = build_X_y(val_fold, "Enrollments_next_month")
                X_val = X_val.reindex(columns=X_train.columns, fill_value=0)

                model = xgb.XGBRegressor(
                    objective="reg:squarederror",
                    random_state=42,
                    tree_method="hist",
                    n_jobs=4,
                    **params,
                )
                model.fit(X_train, transform_target(y_train, use_log_target), verbose=False)

                pred_val_direct = inverse_target(model.predict(X_val), use_log_target)
                pred_val_direct = np.clip(pred_val_direct, 0, None)

                lag_baseline = np.clip(val_fold["Enrollment_lag1"].values.astype(float), 0, None)
                pred_val = np.clip(alpha * pred_val_direct + (1 - alpha) * lag_baseline, 0, None)
                fold_mapes.append(safe_mape(y_val, pred_val))

            fold_mapes = np.array(fold_mapes, dtype=float)
            weighted_mape = float(np.dot(fold_mapes, fold_weights))
            mean_mape = float(np.mean(fold_mapes))

            if weighted_mape < best["cv_weighted_mape"]:
                best.update(
                    {
                        "params": params,
                        "blend_alpha": alpha,
                        "cv_weighted_mape": weighted_mape,
                        "cv_mean_mape": mean_mape,
                        "fold_mapes": fold_mapes.tolist(),
                    }
                )

    return best


def category_revenue_current_baseline(df):
    return np.clip(
        df.groupby(["CourseCategory", "YearMonth"])["Revenue"].transform("sum").values.astype(float),
        0,
        None,
    )


def main():
    courses, teachers, transactions = load_data(DATA_PATH)

    monthly = prepare_monthly_data(courses, teachers, transactions)
    dataset = create_targets(monthly)
    dataset = build_features(dataset)
    dataset = dataset.dropna().copy()

    train, val, test = time_split(dataset)
    train = train.copy()
    val = val.copy()
    test = test.copy()

    category_map = {
        category: idx for idx, category in enumerate(sorted(dataset["CourseCategory"].dropna().unique()))
    }
    for split_df in (train, val, test):
        split_df["CourseCategory_enc"] = (
            split_df["CourseCategory"].map(category_map).fillna(-1).astype(int)
        )

    print("Train shape:", train.shape)
    print("Validation shape:", val.shape)
    print("Test shape:", test.shape)

    # ==========================
    # ENROLLMENT FORECASTING
    # ==========================
    print("\n" + "=" * 50)
    print("ENROLLMENT FORECASTING")
    print("=" * 50)

    y_test_enroll = test["Enrollments_next_month"].values.astype(float)
    y_pred_enroll_baseline = np.clip(test["Enrollment_lag1"].values.astype(float), 0, None)
    enrollment_baseline_metrics = evaluate(
        y_test_enroll, y_pred_enroll_baseline, label="Naive Baseline (Enrollment lag1)"
    )

    enrollment_dev = pd.concat([train, val], axis=0, ignore_index=True)
    enrollment_tuned = tune_enrollment_timeseries_cv(enrollment_dev, use_log_target=True)
    print("Best TS-CV weighted MAPE (Enrollment):", round(enrollment_tuned["cv_weighted_mape"], 2), "%")
    print("Best TS-CV mean MAPE (Enrollment):", round(enrollment_tuned["cv_mean_mape"], 2), "%")
    print("Best params (Enrollment):", enrollment_tuned["params"])
    print("Best blend alpha (Enrollment):", enrollment_tuned["blend_alpha"])
    print("Enrollment fold months:", enrollment_tuned["fold_months"])
    print("Enrollment fold MAPEs:", [round(v, 2) for v in enrollment_tuned["fold_mapes"]])

    enrollment_model, y_test_enroll, y_pred_enroll_direct = fit_final_xgboost(
        train,
        val,
        test,
        target_col="Enrollments_next_month",
        params=enrollment_tuned["params"],
        use_log_target=True,
    )
    y_pred_enroll = np.clip(
        enrollment_tuned["blend_alpha"] * y_pred_enroll_direct
        + (1 - enrollment_tuned["blend_alpha"]) * np.clip(test["Enrollment_lag1"].values.astype(float), 0, None),
        0,
        None,
    )
    enrollment_metrics = evaluate(y_test_enroll, y_pred_enroll, label="Tuned XGBoost (Enrollment)")
    save_model(enrollment_model, "xgboost_enrollment_model.pkl")

    # ==========================
    # COURSE REVENUE FORECASTING
    # ==========================
    print("\n" + "=" * 50)
    print("COURSE REVENUE FORECASTING")
    print("=" * 50)

    y_test_rev = test["Revenue_next_month"].values.astype(float)
    y_pred_rev_baseline = np.clip(test["Revenue_lag1"].values.astype(float), 0, None)
    revenue_baseline_metrics = evaluate(
        y_test_rev, y_pred_rev_baseline, label="Naive Baseline (Revenue lag1)"
    )

    # Direct revenue model
    revenue_tuned = tune_xgboost(train, val, target_col="Revenue_next_month", use_log_target=True)
    print("Best Validation MAPE (Course Revenue):", round(revenue_tuned["val_mape"], 2), "%")
    print("Best params (Course Revenue):", revenue_tuned["params"])

    course_revenue_model, y_test_rev, y_pred_rev_direct = fit_final_xgboost(
        train,
        val,
        test,
        target_col="Revenue_next_month",
        params=revenue_tuned["params"],
        use_log_target=True,
    )

    # Structural revenue from enrollment prediction.
    # Fit a train-only enrollment model for validation predictions to avoid validation leakage.
    X_val_enroll, _ = build_X_y(val, "Enrollments_next_month")
    X_train_enroll, y_train_enroll = build_X_y(train, "Enrollments_next_month")
    X_val_enroll = X_val_enroll.reindex(columns=X_train_enroll.columns, fill_value=0)
    enrollment_val_model = xgb.XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        tree_method="hist",
        n_jobs=4,
        **enrollment_tuned["params"],
    )
    enrollment_val_model.fit(X_train_enroll, transform_target(y_train_enroll, True), verbose=False)
    y_pred_enroll_val_direct = inverse_target(enrollment_val_model.predict(X_val_enroll), use_log_target=True)
    y_pred_enroll_val_direct = np.clip(y_pred_enroll_val_direct, 0, None)
    y_pred_enroll_val = np.clip(
        enrollment_tuned["blend_alpha"] * y_pred_enroll_val_direct
        + (1 - enrollment_tuned["blend_alpha"]) * np.clip(val["Enrollment_lag1"].values.astype(float), 0, None),
        0,
        None,
    )
    y_pred_rev_struct_val = np.clip(y_pred_enroll_val * val["CoursePrice"].values.astype(float), 0, None)
    y_true_rev_val = val["Revenue_next_month"].values.astype(float)

    blend_alpha = 0.0
    best_blend_val_mape = float("inf")
    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        y_pred_blend_val = alpha * revenue_tuned["val_pred"] + (1 - alpha) * y_pred_rev_struct_val
        curr_mape = safe_mape(y_true_rev_val, y_pred_blend_val)
        if curr_mape < best_blend_val_mape:
            best_blend_val_mape = curr_mape
            blend_alpha = alpha

    print("Best validation blend alpha (direct vs structural revenue):", blend_alpha)
    print("Best Validation MAPE (Blended Revenue):", round(best_blend_val_mape, 2), "%")

    y_pred_rev_struct_test = np.clip(y_pred_enroll * test["CoursePrice"].values.astype(float), 0, None)
    y_pred_rev_final = np.clip(
        blend_alpha * y_pred_rev_direct + (1 - blend_alpha) * y_pred_rev_struct_test,
        0,
        None,
    )
    revenue_metrics = evaluate(y_test_rev, y_pred_rev_final, label="Final Course Revenue (Blended)")
    save_model(course_revenue_model, "xgboost_course_revenue_model.pkl")

    # ==========================
    # CATEGORY REVENUE FORECASTING
    # ==========================
    print("\n" + "=" * 50)
    print("CATEGORY REVENUE FORECASTING")
    print("=" * 50)

    y_test_cat_rev = test["Category_Revenue_next_month"].values.astype(float)
    y_pred_cat_rev_baseline = category_revenue_current_baseline(test)
    cat_revenue_baseline_metrics = evaluate(
        y_test_cat_rev,
        y_pred_cat_rev_baseline,
        label="Naive Baseline (Current Category Revenue)",
    )

    cat_revenue_tuned = tune_xgboost(
        train,
        val,
        target_col="Category_Revenue_next_month",
        use_log_target=True,
    )
    print(
        "Best Validation MAPE (Category Revenue):",
        round(cat_revenue_tuned["val_mape"], 2),
        "%",
    )
    print("Best params (Category Revenue):", cat_revenue_tuned["params"])

    category_revenue_model, y_test_cat_rev, y_pred_cat_rev = fit_final_xgboost(
        train,
        val,
        test,
        target_col="Category_Revenue_next_month",
        params=cat_revenue_tuned["params"],
        use_log_target=True,
    )
    cat_revenue_metrics = evaluate(
        y_test_cat_rev,
        y_pred_cat_rev,
        label="Tuned XGBoost (Category Revenue)",
    )
    save_model(category_revenue_model, "xgboost_category_revenue_model.pkl")

    # Persist inference metadata so the app can enforce a stable feature contract.
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_path": DATA_PATH,
        "train_window": {
            "train_end": TRAIN_END,
            "val_start": VAL_START,
            "val_end": VAL_END,
            "test_month": TEST_MONTH,
        },
        "category_map": category_map,
        "blending": {
            "enrollment_blend_alpha": float(enrollment_tuned["blend_alpha"]),
            "course_revenue_blend_alpha": float(blend_alpha),
            "course_revenue_structural_formula": "pred_enrollment * course_price",
        },
        "models": {
            "enrollment": {
                "filename": "xgboost_enrollment_model.pkl",
                "feature_columns": list(enrollment_model.feature_names_in_),
                "target_transform": "log1p",
                "best_params": enrollment_tuned["params"],
            },
            "course_revenue": {
                "filename": "xgboost_course_revenue_model.pkl",
                "feature_columns": list(course_revenue_model.feature_names_in_),
                "target_transform": "log1p",
                "best_params": revenue_tuned["params"],
            },
            "category_revenue": {
                "filename": "xgboost_category_revenue_model.pkl",
                "feature_columns": list(category_revenue_model.feature_names_in_),
                "target_transform": "log1p",
                "best_params": cat_revenue_tuned["params"],
            },
        },
    }
    save_metadata(metadata, "prediction_metadata.json")

    # ==========================
    # SUMMARY
    # ==========================
    print("\n" + "=" * 50)
    print("TEST METRICS SUMMARY (MAPE)")
    print("=" * 50)
    print(
        "Enrollment:",
        f"{enrollment_baseline_metrics['MAPE']:.2f}% -> {enrollment_metrics['MAPE']:.2f}%",
    )
    print(
        "Course Revenue:",
        f"{revenue_baseline_metrics['MAPE']:.2f}% -> {revenue_metrics['MAPE']:.2f}%",
    )
    print(
        "Category Revenue:",
        f"{cat_revenue_baseline_metrics['MAPE']:.2f}% -> {cat_revenue_metrics['MAPE']:.2f}%",
    )


if __name__ == "__main__":
    main()
