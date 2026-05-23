#!/usr/bin/env python3
"""Defines a function to remove rows with missing data from a DataFrame"""


def prune(df):
    """Removes any entries where the Close column has NaN values.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The modified DataFrame with NaN entries removed
                      from the Close column.
    """
    return df.dropna(subset=['Close'])
