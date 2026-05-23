#!/usr/bin/env python3
"""Defines a function to clean and fill missing values in a DataFrame"""


def fill(df):
    """Removes 'Weighted_Price' and propagates missing data based on standard

    crypto OHLCV data-cleaning rules.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The modified DataFrame.
    """
    # 1. Remove the Weighted_Price column
    if 'Weighted_Price' in df.columns:
        df = df.drop(columns=['Weighted_Price'])

    # 2. Fill missing values in Close with the previous row's value (ffill)
    df['Close'] = df['Close'].ffill()

    # 3. Fill missing High, Low, Open values with the Close value of the row
    # Using fillna with the Close series mapped over the columns
    df['High'] = df['High'].fillna(df['Close'])
    df['Low'] = df['Low'].fillna(df['Close'])
    df['Open'] = df['Open'].fillna(df['Close'])

    # 4. Set missing values in Volume columns to 0
    df['Volume_(BTC)'] = df['Volume_(BTC)'].fillna(0)
    df['Volume_(Currency)'] = df['Volume_(Currency)'].fillna(0)

    return df
