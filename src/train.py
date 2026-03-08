from data_loader import load_data
from data_processing import prepare_monthly_data, create_targets
from feature_engineering import build_features
from model_utils import save_model
from model_utils import save_model

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
import lightgbm as lgb
import pandas as pd

DATA_PATH = "data/raw/EduPro Online Platform.xlsx"

# Toggle: model enrollment in log space (recommended for count-like targets)
USE_LOG_TARGET = True


def time_split(df):
    df = df.sort_values("YearMonth")

    train = df[df["YearMonth"] <= "2025-08"]
    val = df[(df["YearMonth"] >= "2025-09") & (df["YearMonth"] <= "2025-10")]
    test = df[df["YearMonth"] == "2025-11"]

    return train, val, test


def safe_mape(y_true, y_pred):
    """MAPE that ignores y_true == 0 to avoid division explosions."""
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


def get_X_y(df, drop_cols, target_col, use_log_target):
    """Build numeric feature matrix X, plus raw y and model-space y."""
    X = df.drop(columns=[c for c in drop_cols if c in df.columns]).select_dtypes(include=["number"])
    y_raw = df[target_col].values.astype(float)

    if use_log_target:
        y_model = np.log1p(y_raw)
    else:
        y_model = y_raw

    return X, y_raw, y_model


def inverse_target(pred_model_space, use_log_target):
    """Convert model predictions back to raw enrollment counts."""
    pred_model_space = np.array(pred_model_space, dtype=float)
    if use_log_target:
        return np.expm1(pred_model_space)
    return pred_model_space


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

    # ==========================
    # ENROLLMENT FORECASTING
    # ==========================
    target = "Enrollments_next_month"

    drop_cols = [
        "CourseID",
        "YearMonth",
        "Enrollments_next_month",
        "Revenue_next_month",
        "Course_Share_next_month",
        "Category_Enrollment_next_month",
    ]

    X_train, y_train_raw, y_train = get_X_y(train, drop_cols, target, USE_LOG_TARGET)
    X_val, y_val_raw, y_val = get_X_y(val, drop_cols, target, USE_LOG_TARGET)
    X_test, y_test_raw, y_test = get_X_y(test, drop_cols, target, USE_LOG_TARGET)

    # ---- Baseline (raw-scale) ----
    y_pred_baseline = test["Enrollment_lag1"].values.astype(float)
    y_pred_baseline = np.clip(y_pred_baseline, 0, None)
    evaluate(y_test_raw, y_pred_baseline, label="Naive Baseline (Enrollment)")

    # ---- Linear Regression ----
    linear_model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ])
    linear_model.fit(X_train, y_train)

    pred_lin_model_space = linear_model.predict(X_test)
    y_pred_linear = inverse_target(pred_lin_model_space, USE_LOG_TARGET)
    y_pred_linear = np.clip(y_pred_linear, 0, None)
    evaluate(y_test_raw, y_pred_linear, label="Linear Regression (Enrollment)")

    # ---- Ridge Regression ----
    ridge_model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0))
    ])
    ridge_model.fit(X_train, y_train)

    pred_ridge_model_space = ridge_model.predict(X_test)
    y_pred_ridge = inverse_target(pred_ridge_model_space, USE_LOG_TARGET)
    y_pred_ridge = np.clip(y_pred_ridge, 0, None)
    evaluate(y_test_raw, y_pred_ridge, label="Ridge Regression (Enrollment)")

    # ---- GBR Default (kept for comparison) ----
    gbr_model = GradientBoostingRegressor(
        n_estimators=1500,
        learning_rate=0.001,
        max_depth=3,
        random_state=42
    )
    gbr_model.fit(X_train, y_train)

    

    imp = pd.Series(gbr_model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    print("\nTop 15 Feature Importances (GBR):")
    print(imp.head(15))

    pred_gbr_model_space = gbr_model.predict(X_test)
    y_pred_gbr = inverse_target(pred_gbr_model_space, USE_LOG_TARGET)
    y_pred_gbr = np.clip(y_pred_gbr, 0, None)
    evaluate(y_test_raw, y_pred_gbr, label="Gradient Boosting (Enrollment)")

    # ---- GBR Tuning on VALIDATION (raw-scale MAPE) ----
    best_model = None
    best_mape = float("inf")
    best_params = None

    for lr in [0.01, 0.05, 0.1]:
        for n in [200, 500, 800]:
            for depth in [2, 3, 4]:
                model = GradientBoostingRegressor(
                    learning_rate=lr,
                    n_estimators=n,
                    max_depth=depth,
                    random_state=42
                )
                model.fit(X_train, y_train)

                pred_val_model_space = model.predict(X_val)
                pred_val_raw = inverse_target(pred_val_model_space, USE_LOG_TARGET)
                pred_val_raw = np.clip(pred_val_raw, 0, None)

                mape = safe_mape(y_val_raw, pred_val_raw)

                if mape < best_mape:
                    best_mape = mape
                    best_model = model
                    best_params = (lr, n, depth)

    print("\nBest Validation MAPE (GBR):", round(best_mape, 2), "%")
    print("Best GBR params (lr, n_estimators, max_depth):", best_params)

    # Evaluate best tuned model on TEST (raw-scale)
    pred_test_model_space = best_model.predict(X_test)
    y_pred_gbr_tuned = inverse_target(pred_test_model_space, USE_LOG_TARGET)
    y_pred_gbr_tuned = np.clip(y_pred_gbr_tuned, 0, None)
    evaluate(y_test_raw, y_pred_gbr_tuned, label="Tuned Gradient Boosting (Enrollment)")

    # ---- XGBoost ----
    xgb_model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
        objective='reg:squarederror'
    )
    xgb_model.fit(X_train, y_train)

    pred_xgb_model_space = xgb_model.predict(X_test)
    y_pred_xgb = inverse_target(pred_xgb_model_space, USE_LOG_TARGET)
    y_pred_xgb = np.clip(y_pred_xgb, 0, None)
    evaluate(y_test_raw, y_pred_xgb, label="XGBoost (Enrollment)")

    # Save the XGBoost model
    save_model(xgb_model, "xgboost_enrollment_model.pkl")

    # ---- XGBoost Tuning on VALIDATION ----
    best_xgb_model = None
    best_xgb_mape = float("inf")
    best_xgb_params = None

    for lr in [0.01, 0.05, 0.1]:
        for n in [200, 500, 800]:
            for depth in [2, 3, 4]:
                model = xgb.XGBRegressor(
                    learning_rate=lr,
                    n_estimators=n,
                    max_depth=depth,
                    random_state=42,
                    objective='reg:squarederror'
                )
                model.fit(X_train, y_train)

                pred_val_model_space = model.predict(X_val)
                pred_val_raw = inverse_target(pred_val_model_space, USE_LOG_TARGET)
                pred_val_raw = np.clip(pred_val_raw, 0, None)

                mape = safe_mape(y_val_raw, pred_val_raw)

                if mape < best_xgb_mape:
                    best_xgb_mape = mape
                    best_xgb_model = model
                    best_xgb_params = (lr, n, depth)

    print("\nBest Validation MAPE (XGBoost):", round(best_xgb_mape, 2), "%")
    print("Best XGBoost params (lr, n_estimators, max_depth):", best_xgb_params)

    # Evaluate best tuned XGBoost on TEST
    pred_test_xgb_model_space = best_xgb_model.predict(X_test)
    y_pred_xgb_tuned = inverse_target(pred_test_xgb_model_space, USE_LOG_TARGET)
    y_pred_xgb_tuned = np.clip(y_pred_xgb_tuned, 0, None)
    evaluate(y_test_raw, y_pred_xgb_tuned, label="Tuned XGBoost (Enrollment)")

    # ---- LightGBM ----
    lgb_model = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
        verbosity=-1  # suppress warnings
    )
    lgb_model.fit(X_train, y_train)

    pred_lgb_model_space = lgb_model.predict(X_test)
    y_pred_lgb = inverse_target(pred_lgb_model_space, USE_LOG_TARGET)
    y_pred_lgb = np.clip(y_pred_lgb, 0, None)
    evaluate(y_test_raw, y_pred_lgb, label="LightGBM (Enrollment)")

    # ---- Ensemble: XGBoost + GBR ----
    # Use the best XGBoost and tuned GBR
    pred_ensemble_model_space = (pred_xgb_model_space + pred_gbr_model_space) / 2
    y_pred_ensemble = inverse_target(pred_ensemble_model_space, USE_LOG_TARGET)
    y_pred_ensemble = np.clip(y_pred_ensemble, 0, None)
    evaluate(y_test_raw, y_pred_ensemble, label="Ensemble XGBoost+GBR (Enrollment)")

    # ==========================
    # HIERARCHICAL ENROLLMENT (Category × Share)
    # ==========================
    print("\n" + "=" * 50)
    print("HIERARCHICAL ENROLLMENT (Category × Share)")
    print("=" * 50)

    # Category model (predict category total next month)
    cat_drop = [
        "CourseID", "YearMonth",
        "Enrollments_next_month", "Revenue_next_month",
        "Course_Share_next_month", "Category_Enrollment_next_month"
    ]
    X_train_cat = train.drop(columns=[c for c in cat_drop if c in train.columns]).select_dtypes(include=["number"])
    X_test_cat = test.drop(columns=[c for c in cat_drop if c in test.columns]).select_dtypes(include=["number"])
    y_train_cat = train["Category_Enrollment_next_month"].values.astype(float)
    y_test_cat = test["Category_Enrollment_next_month"].values.astype(float)

    cat_model = xgb.XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
        objective='reg:squarederror'
    )
    cat_model.fit(X_train_cat, y_train_cat)
    cat_pred = np.clip(cat_model.predict(X_test_cat), 0, None)

    # Share model (predict course share next month)
    share_drop = cat_drop  # same drops
    X_train_share = train.drop(columns=[c for c in share_drop if c in train.columns]).select_dtypes(include=["number"])
    X_test_share = test.drop(columns=[c for c in share_drop if c in test.columns]).select_dtypes(include=["number"])
    y_train_share = train["Course_Share_next_month"].values.astype(float)
    y_test_share = test["Course_Share_next_month"].values.astype(float)

    share_model = xgb.XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
        objective='reg:squarederror'
    )
    share_model.fit(X_train_share, y_train_share)
    share_pred = np.clip(share_model.predict(X_test_share), 0, 1)

    # Final hierarchical enrollment prediction (raw-scale)
    y_pred_hier = cat_pred * share_pred
    evaluate(test["Enrollments_next_month"].values, y_pred_hier, label="Hierarchical Enrollment (Category×Share)")
    evaluate(y_test_cat, cat_pred, label="Category Model")
    evaluate(y_test_share, share_pred, label="Share Model")

    # ==========================
    # REVENUE FORECASTING (STRUCTURAL)
    # Revenue_next_month ≈ Predicted_Enrollments_next_month × CoursePrice
    # ==========================
    print("\n" + "=" * 50)
    print("REVENUE FORECASTING (STRUCTURAL: Enrollment × Price)")
    print("=" * 50)

    y_test_rev = test["Revenue_next_month"].values.astype(float)

    price = test["CoursePrice"].values.astype(float)

    # Revenue from GBR-log enrollment
    rev_from_gbr = np.clip(y_pred_gbr * price, 0, None)
    evaluate(y_test_rev, rev_from_gbr, label="Revenue via GBR(log) Enrollment × Price")

    # Revenue from Hierarchical enrollment
    rev_from_hier = np.clip(y_pred_hier * price, 0, None)
    evaluate(y_test_rev, rev_from_hier, label="Revenue via Hier Enrollment × Price")

    # Optional: blended enrollment -> blended revenue
    blend_enroll = 0.6 * y_pred_gbr + 0.4 * y_pred_hier
    rev_from_blend = np.clip(blend_enroll * price, 0, None)
    evaluate(y_test_rev, rev_from_blend, label="Revenue via Blended Enrollment × Price")
    
if __name__ == "__main__":
    main()