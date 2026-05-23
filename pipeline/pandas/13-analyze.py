#!/usr/bin/env python3
"""Defines a function to compute descriptive statistics of a DataFrame"""


def analyze(df):
    """Computes descriptive statistics for all columns except Timestamp.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: A new DataFrame containing the calculated statistics.
    """
    # Drop the Timestamp column if it is present in the columns
    if 'Timestamp' in df.columns:
        df = df.drop(columns=['Timestamp'])

    # Compute and return the descriptive statistics
    return df.describe()
