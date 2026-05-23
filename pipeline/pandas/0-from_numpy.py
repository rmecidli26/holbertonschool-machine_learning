#!/usr/bin/env python3
"""Defines a function to create a pandas DataFrame from a numpy array"""
import pandas as pd


def from_numpy(array):
    """Creates a pd.DataFrame from a np.ndarray with columns labeled
    in alphabetical order and capitalized.

    Args:
        array: np.ndarray to convert

    Returns:
        pd.DataFrame: the newly created DataFrame
    """
    # Create a list of uppercase letters corresponding to the number of columns
    # ASCII value of 'A' is 65
    column_labels = [chr(65 + i) for i in range(array.shape[1])]

    # Generate and return the DataFrame
    return pd.DataFrame(array, columns=column_labels)
