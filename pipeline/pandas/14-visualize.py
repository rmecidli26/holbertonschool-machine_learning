#!/usr/bin/env python3
import matplotlib.pyplot as plt
import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

# 1. Remove the Weighted_Price column
df = df.drop(columns=['Weighted_Price'])

# 2. Rename the column Timestamp to Date
df = df.rename(columns={'Timestamp': 'Date'})

# 3. Convert the timestamp values to datetime values
df['Date'] = pd.to_datetime(df['Date'], unit='s')

# 4. Index the dataframe on Date
df = df.set_index('Date')

# 5. Fill missing values in Close with the previous row's value
df['Close'] = df['Close'].ffill()

# 6. Fill missing values in High, Low, Open with the same row's Close value
df['High'] = df['High'].fillna(df['Close'])
df['Low'] = df['Low'].fillna(df['Close'])
df['Open'] = df['Open'].fillna(df['Close'])

# 7. Set missing values in Volume columns to 0
df['Volume_(BTC)'] = df['Volume_(BTC)'].fillna(0)
df['Volume_(Currency)'] = df['Volume_(Currency)'].fillna(0)

# 8. Filter data from 2017 and beyond
df = df.loc['2017-01-01':]

# 9. Resample at daily intervals ('D') with specific aggregations
df = df.resample('D').agg({
    'High': 'max',
    'Low': 'min',
    'Open': 'mean',
    'Close': 'mean',
    'Volume_(BTC)': 'sum',
    'Volume_(Currency)': 'sum'
})

# Plot the results
df.plot()
plt.show()
