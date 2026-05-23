#!/usr/bin/env python3
"""Defines a function to merge and structure crypto data by hierarchy"""
import pandas as pd
index = __import__('10-index').index


def hierarchy(df1, df2):
    """Indexes, filters, and concatenates two DataFrames while reorganizing

    the MultiIndex level so that Timestamp is the primary level sorted
    chronologically.

    Args:
        df1 (pd.DataFrame): Coinbase DataFrame.
        df2 (pd.DataFrame): Bitstamp DataFrame.

    Returns:
        pd.DataFrame: The concatenated multi-index DataFrame.
    """
    # Index both dataframes on their Timestamp columns
    df1_indexed = index(df1)
    df2_indexed = index(df2)

    # Filter both dataframes between timestamps 1417411980 and 1417417980
    df1_filtered = df1_indexed.loc[1417411980:1417417980]
    df2_filtered = df2_indexed.loc[1417411980:1417417980]

    # Concatenate the selected rows with respective keys
    df = pd.concat(
        [df2_filtered, df1_filtered],
        keys=['bitstamp', 'coinbase']
    )

    # Swap the MultiIndex levels and sort them chronologically
    return df.swaplevel(0, 1).sort_index()
