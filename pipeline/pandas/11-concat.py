#!/usr/bin/env python3
"""Defines a function to concatenate two DataFrames with hierarchical keys"""
import pandas as pd
index = __import__('10-index').index


def concat(df1, df2):
    """Indexes both dataframes on Timestamp, filters df2 up to 1417411920,

    and concatenates df2 on top of df1 with keys 'bitstamp' and 'coinbase'.

    Args:
        df1 (pd.DataFrame): Coinbase DataFrame.
        df2 (pd.DataFrame): Bitstamp DataFrame.

    Returns:
        pd.DataFrame: The concatenated multi-index DataFrame.
    """
    # Index both dataframes on their Timestamp columns
    df1_indexed = index(df1)
    df2_indexed = index(df2)

    # Filter bitstamp data up to and including timestamp 1417411920
    df2_filtered = df2_indexed.loc[:1417411920]

    # Concatenate df2 on top of df1 with the respective keys
    return pd.concat([df2_filtered, df1_indexed], keys=['bitstamp', 'coinbase'])
