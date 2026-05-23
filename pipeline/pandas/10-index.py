#!/usr/bin/env python3
"""Defines a function to set a specific column as the DataFrame index"""


def index(df):
    """Sets the 'Timestamp' column as the index of the dataframe.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The modified DataFrame with 'Timestamp' as its index.
    """
    return df.set_index('Timestamp')
