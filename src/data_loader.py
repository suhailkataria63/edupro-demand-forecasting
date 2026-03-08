import pandas as pd

def _clean_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.replace(" ", "", regex=False)
        .str.replace("-", "", regex=False)
    )
    return df

def load_data(path):
    courses = pd.read_excel(path, sheet_name="Courses")
    teachers = pd.read_excel(path, sheet_name="Teachers")
    transactions = pd.read_excel(path, sheet_name="Transactions")

    return _clean_cols(courses), _clean_cols(teachers), _clean_cols(transactions)