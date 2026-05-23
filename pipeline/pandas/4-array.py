#!/usr/bin/env python3
"""Defines a function to convert specific DataFrame rows into a NumPy array"""
import pandas as pd


def array(df):
    """Selects the last 10 rows of the 'High' and 'Close' columns and converts

    them into a numpy.ndarray.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        numpy.ndarray: The array containing the selected high and close values.
    """
    # Select the High and Close columns, take the last 10 rows, and convert
    return df[['High', 'Close']].tail(10).to_numpy()
