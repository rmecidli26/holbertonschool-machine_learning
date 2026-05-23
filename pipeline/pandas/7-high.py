#!/usr/bin/env python3
"""Defines a function to sort a pandas DataFrame by a specific column"""


def high(df):
    """Sorts the DataFrame by the 'High' price in descending order.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The sorted DataFrame.
    """
    return df.sort_values(by='High', ascending=False)
