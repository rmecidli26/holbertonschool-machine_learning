#!/usr/bin/env python3
"""Defines a function to sort and transpose a pandas DataFrame"""


def flip_switch(df):
    """Sorts data in reverse chronological order and transposes the result.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The transformed DataFrame.
    """
    # Sort by the index in descending order to reverse the timeline
    # and then transpose using the .T attribute
    return df.sort_index(ascending=False).T
