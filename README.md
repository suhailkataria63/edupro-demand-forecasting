# EduPro Demand & Revenue Forecasting

Time-aware predictive modeling to forecast next-month course demand (enrollments) and revenue for an online learning platform.

## Setup
1) Place the dataset in: `data/raw/EduPro Online Platform.xlsx` (not tracked in git)
2) Install deps: `pip install -r requirements.txt`
3) Train the model: `python src/train.py` (models will be saved to `models/`)
4) Run app: `streamlit run app/streamlit_app.py`

## Project Structure
- `src/`: Core modules
  - `data_loader.py`: Load data from Excel
  - `data_processing.py`: Process and create targets
  - `feature_engineering.py`: Build features
  - `train.py`: Train models and save to `models/`
  - `model_utils.py`: Utilities for saving/loading models
- `app/`: Streamlit web app
- `data/`: Data directories (raw, processed)
- `models/`: Saved trained models
- `notebooks/`: (Empty) For exploratory analysis
- `reports/`: (Empty) For outputs and visualizations

## Dataset Note
Note: The provided dataset does not include a course-to-teacher mapping (Courses sheet has no TeacherID).
Teacher-level features are excluded from modeling and listed as a future enhancement.
