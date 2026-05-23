#!/usr/bin/env python3
"""Defines a function to load data from a file into a pandas DataFrame"""
import pandas as pd


def from_file(filename, delimiter):
    """Loads data from a file as a pd.DataFrame.

    Args:
        filename (str): The path to the file to load from.
        delimiter (str): The character separating the file's columns.

    Returns:
        pd.DataFrame: The loaded pandas DataFrame object.
    """
    return pd.read_csv(filename, sep=delimiter)
