#!/usr/bin/env python3
"""Defines a function to slice specific columns and rows from a DataFrame"""


def slice(df):
    """Extracts columns High, Low, Close, and Volume_(BTC) and selects

    every 60th row.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The sliced DataFrame.
    """
    # Select the required columns first
    columns = ['High', 'Low', 'Close', 'Volume_(BTC)']

    # Slice every 60th row using iloc
    return df[columns].iloc[::60]
