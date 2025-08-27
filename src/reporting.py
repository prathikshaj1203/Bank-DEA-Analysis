"""
Reporting utilities for Bank Efficiency DEA project.
Provides helper functions to summarize and rank banks
based on efficiency metrics.
"""

import pandas as pd


def top_efficient_banks(df: pd.DataFrame, metric: str, year: int, n: int = 10) -> pd.DataFrame:
    """
    Return top N efficient banks for a given year.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with efficiency results.
    metric : str
        Column name for efficiency score ("efficiency_ccr", "efficiency_bcc", or "scale_efficiency").
    year : int
        Year to filter by.
    n : int
        Number of top banks to return.

    Returns
    -------
    pd.DataFrame
        DataFrame of top N banks with their efficiency scores.
    """
    df_year = df[df["year"] == year]
    return (
        df_year[["bank_name", metric]]
        .sort_values(by=metric, ascending=False)
        .head(n)
        .reset_index(drop=True)
    )


def average_efficiency_by_year(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """
    Compute average efficiency across all banks per year.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with efficiency results.
    metric : str
        Efficiency metric column.

    Returns
    -------
    pd.DataFrame
        Year-wise average efficiency.
    """
    return (
        df.groupby("year")[metric]
        .mean()
        .reset_index(name=f"avg_{metric}")
    )


def bank_efficiency_trend(df: pd.DataFrame, bank: str, metric: str) -> pd.DataFrame:
    """
    Get efficiency trend of a single bank across years.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with efficiency results.
    bank : str
        Bank name to filter.
    metric : str
        Efficiency metric column.

    Returns
    -------
    pd.DataFrame
        Year vs efficiency for that bank.
    """
    return (
        df[df["bank_name"] == bank][["year", metric]]
        .sort_values("year")
        .reset_index(drop=True)
    )


def save_results(df: pd.DataFrame, path: str):
    """
    Save results DataFrame to CSV.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to save.
    path : str
        File path to save CSV.
    """
    df.to_csv(path, index=False)
