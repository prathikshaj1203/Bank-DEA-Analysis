import pandas as pd

def load_data(path: str) -> pd.DataFrame:
    """Load Excel dataset."""
    df = pd.read_excel(path)
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean column names and handle missing values."""
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    df = df.dropna(how="all")  # drop empty rows
    return df
