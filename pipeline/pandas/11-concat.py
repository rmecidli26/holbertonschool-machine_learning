#!/usr/bin/env python3
"""Defines a function to concatenatrchical keys"""
import pandas as pd
index = __import__('10-index').index


def concat(df1, df2):
    """Indexes both dataframes on Timto 1417411920,

    and concatenates df2 on top of df1 with keynd 'coinbase'.

    Args:
        df1 (pd.DataFrame): Coinbase DataFrame.
        df2 (pd.DataFrame): Bitstamp DataFrame.

    Returns:
        pd.DataFrame: The concatenated mame.
    """
    # Index both dataframes on their Timestamp columns
    df1_indexed = index(df1)
    df2_indexed = index(df2)

    # Filter bitstamp data up to 11920
    df2_filtered = df2_indexed.loc[:1417411920]

    # Concatenate df2 on top of df1 with the respective keys
    return pd.concat([df2_filtered, df1_indexed], keys=['bitstamp', 'coinbase'])
