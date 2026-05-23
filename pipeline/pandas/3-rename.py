#!/usr/bin/env python3
"""Defines a function to rename and convert columns in a pandas DataFrame"""
import pandas as pd


def rename(df):
    """Renames the 'Timestamp' column to 'Datetime', converts its values

    to datetime objects, and slices out only the 'Datetime' and 'Close'
    columns.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The modified DataFrame containing only the Datetime
                      and Close columns.
    """
    # Rename the column
    df = df.rename(columns={'Timestamp': 'Datetime'})

    # Convert the column to datetime formatting using seconds unit
    df['Datetime'] = pd.to_datetime(df['Datetime'], unit='s')

    # Return only the specified columns
    return df[['Datetime', 'Close']]
