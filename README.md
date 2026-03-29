# EduPro Demand Forecasting

## Overview
AI-based forecasting system that predicts:
- course enrollment
- revenue

Uses:
- Gradient Boosting (XGBoost regressors)
- hierarchical forecasting signals (course and category level)
- feature-engineered time-series signals (lags, rolling means, trends, calendar features)

## Project Structure
- `src/`: data loading, preprocessing, feature engineering, model training, utilities
- `app/`: Streamlit dashboard (`streamlit_app.py`)
- `models/`: trained model artifacts and prediction metadata
- `data/`: raw and processed datasets

## How to Run
1. Clone repo
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run Streamlit:
   ```bash
   streamlit run app/streamlit_app.py
   ```

If model files are missing, train first:
```bash
python src/train.py
```

## Model Performance
Evaluation setup: time-based split (`train <= 2025-08`, `validation = 2025-09..2025-10`, `test = 2025-11`).

### Enrollment Forecasting
- Baseline: previous month enrollment (`Enrollment_lag1`)
- Final: tuned XGBoost + lag-blended prediction (`alpha = 0.75`)
- Test results:
  - MAE: `3.767 -> 2.932`
  - RMSE: `4.913 -> 3.740`
  - R2: `-0.6531 -> 0.0417`
  - MAPE: `29.63% -> 24.88%`

### Revenue Forecasting
- Baseline: previous month revenue (`Revenue_lag1`)
- Final: blended forecast of direct XGBoost revenue and structural revenue (`predicted enrollment x course price`, blend `alpha = 0.5`)
- Test results:
  - MAE: `332.630 -> 193.213`
  - RMSE: `866.289 -> 431.411`
  - R2: `0.8242 -> 0.9564`
  - MAPE: `30.38% -> 27.15%`

### Category Revenue Forecasting
- Baseline: current category revenue
- Final: tuned XGBoost category-level revenue model
- Test results:
  - MAE: `1312.300 -> 1006.419`
  - RMSE: `1916.522 -> 1387.918`
  - R2: `0.8796 -> 0.9368`
  - MAPE: `26.53% -> 19.97%`

## Dashboard
The Streamlit app allows interactive prediction for enrollment, course revenue, and category revenue using the trained models and metadata contract.
