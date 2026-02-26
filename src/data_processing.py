import pandas as pd

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
    df = df.sort_values(['CourseID', 'YearMonth'])

    df['Enrollments_next_month'] = df.groupby('CourseID')['Enrollment_count'].shift(-1)
    df['Revenue_next_month'] = df.groupby('CourseID')['Revenue'].shift(-1)

    df = df.dropna(subset=['Enrollments_next_month', 'Revenue_next_month'])

    return df