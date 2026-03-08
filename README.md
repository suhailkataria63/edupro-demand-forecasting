# EduPro Demand & Revenue Forecasting

Time-aware predictive modeling to forecast next-month course demand (enrollments) and revenue for an online learning platform.

## Setup
1) Place the dataset in: `data/raw/EduPro Online Platform.xlsx` (not tracked in git)
2) Install deps: `pip install -r requirements.txt`
3) Train models and generate inference metadata: `python src/train.py`
4) Run app: `streamlit run app/streamlit_app.py`

## Production Streamlit Notes
- The app requires these artifacts in `models/`:
  - `xgboost_enrollment_model.pkl`
  - `xgboost_course_revenue_model.pkl`
  - `xgboost_category_revenue_model.pkl`
  - `prediction_metadata.json`
- The app enforces feature-contract checks between model artifacts and metadata at startup.
- You can override dataset location with environment variable:
  - `EDUPRO_DATA_PATH=/path/to/EduPro Online Platform.xlsx`

## Project Structure
- `src/`: Core modules
  - `data_loader.py`: Load data from Excel
  - `data_processing.py`: Process and create targets
  - `feature_engineering.py`: Build features
  - `train.py`: Train models and save models + metadata to `models/`
  - `model_utils.py`: Utilities for saving/loading models and metadata
- `app/`: Streamlit web app
- `data/`: Data directories (raw, processed)
- `models/`: Saved trained models
- `notebooks/`: (Empty) For exploratory analysis
- `reports/`: (Empty) For outputs and visualizations

## Dataset Note
Note: The provided dataset does not include a course-to-teacher mapping (Courses sheet has no TeacherID).
Teacher-level features are excluded from modeling and listed as a future enhancement.
