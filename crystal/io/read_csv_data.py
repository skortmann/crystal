#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generalized data reading and feature engineering for different energy markets.

Copyright (c) by Institute for High Voltage Equipment and Grids,
Digitalization and Energy Economics (IAEW), RWTH Aachen University,
10.02.2025, s.kortmann. All rights reserved.
"""

import pandas as pd
import numpy as np
from sympy import continued_fraction_reduce


def read_csv_data(path, market):
    # Load data and set the index
    df = pd.read_csv(path)
    df.index = pd.to_datetime(df["von"])
    # df = df[df.index.year == 2021]  # Filter data for year 2021

    # Generalized market-specific processing
    market_config = {
        "daa": {
            "produkt": "Stundenkontrakt",
            "columns": ["preis", "volumen"],
            "value_col": "preis",
        },
        "ida": {"columns": ["preis", "volumen"], "value_col": "preis"},
        "idc": {
            "produkt": "Viertelstundenkontrakt",
            "columns": ["high", "buy_vol", "sell_vol"],
            "value_col": "high",
        },
    }

    config = market_config[market]
    if "produkt" in config:
        df = df[df["produkt"] == config["produkt"]]

    # Select and rename columns
    df = df[config["columns"]].rename(columns={config["value_col"]: "value"})
    df.name = market

    # Set timestamp and remove duplicates
    df["timestamp"] = df.index
    df = df.drop_duplicates(subset=["timestamp"]).reset_index(drop=True)

    # Set frequency and expected length based on the market
    freq = "15min" if market in ["ida", "idc"] else "1h"
    full_range = pd.date_range(
        start=df["timestamp"].min(), end=df["timestamp"].max(), freq=freq
    )

    # Reindex and forward-fill missing values
    df = (
        df.set_index("timestamp")
        .reindex(full_range)
        .ffill()
        .reset_index()
        .rename(columns={"index": "timestamp"})
    )

    # Check length
    if len(df) != len(full_range):
        raise ValueError(
            f"Dataframe has incorrect length after reindexing: {len(df)} entries for {market} market."
        )

    if df.empty:
        raise ValueError("Empty dataframe")

    return df


def feature_engineering(df, market):
    # Convert timestamp to DateTime
    df["Date"] = pd.to_datetime(df["timestamp"])

    # Determine full-day frequency
    full_day_freq = 24 if market == "daa" else 96

    # 🔹 Efficient Lag Features (Using Dictionary Comprehension)
    lag_features = {f"lag_{i}": df["value"].shift(i) for i in range(full_day_freq)}
    lag_features.update(
        {
            "lag_day": df["value"].shift(full_day_freq),
            "lag_week": df["value"].shift(7 * full_day_freq),
        }
    )

    # 🔹 Efficient Rolling Statistics
    rolling_features = {
        "rolling_mean_week": df["value"].rolling(window=7 * full_day_freq).mean(),
        "rolling_mean_day": df["value"].rolling(window=full_day_freq).mean(),
        "rolling_std_week": df["value"].rolling(window=7 * full_day_freq).std(),
        "rolling_std_day": df["value"].rolling(window=full_day_freq).std(),
    }

    # 🔹 Efficient Date-Based Features
    date_features = {
        "year": df["Date"].dt.year,
        "month": df["Date"].dt.month,
        "day": df["Date"].dt.day,
        "dayofweek": df["Date"].dt.dayofweek,
        "hour": df["Date"].dt.hour,
        "minute": df["Date"].dt.minute,
        "weekend": df["Date"].dt.weekday.isin([5, 6]).astype(float),
    }

    # 🔹 Cyclical Encoding for Time Features
    date_features.update(
        {
            "sin_day_of_week": np.sin(2 * np.pi * df["Date"].dt.dayofweek / 7),
            "cos_day_of_week": np.cos(2 * np.pi * df["Date"].dt.dayofweek / 7),
        }
    )

    if full_day_freq == 96:  # 15-minute intervals
        date_features.update(
            {
                "sin_hour_of_day": np.sin(2 * np.pi * df["Date"].dt.minute / (96 * 15)),
                "cos_hour_of_day": np.cos(2 * np.pi * df["Date"].dt.minute / (96 * 15)),
            }
        )
    else:  # Hourly intervals
        date_features.update(
            {
                "sin_hour_of_day": np.sin(
                    2 * np.pi * df["Date"].dt.hour / full_day_freq
                ),
                "cos_hour_of_day": np.cos(
                    2 * np.pi * df["Date"].dt.hour / full_day_freq
                ),
            }
        )

    # 🔹 Combine All Features Efficiently
    df = pd.concat(
        [
            df,
            pd.DataFrame(lag_features),
            pd.DataFrame(rolling_features),
            pd.DataFrame(date_features),
        ],
        axis=1,
    )

    # Drop rows with NaNs (from shifting/rolling)
    # df.dropna(inplace=True)

    return df

def add_volatility_measures(df, market):

    if not market == "idc":
        return df

    # Define the Chaikin Volatility function
    def chaikin_volatility(data, period=10):
        data['HL'] = data['high'] - data['low']
        data['CHV'] = data['HL'].ewm(span=period, adjust=False).mean()
        data['ChaikinVolatility'] = data['CHV'].pct_change(periods=period) * 100
        return data

    # Apply the Chaikin Volatility function to the data
    df = chaikin_volatility(df)

    # Define the Donchian Channels function
    def donchian_channels(data, period=20):
        data['Upper'] = data['high'].rolling(window=period).max()
        data['Lower'] = data['low'].rolling(window=period).min()
        data['Middle'] = (data['Upper'] + data['Lower']) / 2
        return data

    # Apply the Donchian Channels function to the data
    df = donchian_channels(df)

    # Define the Keltner Channels function
    def keltner_channels(data, period=20, atr_multiplier=2):
        data['TR'] = data[['high', 'low', 'last']].apply(
            lambda x: max(
                x['high'] - x['low'],
                abs(x['high'] - x['last']),
                abs(x['low'] - x['last'])
            ), axis=1
        )
        data['ATR'] = data['TR'].rolling(window=period).mean()
        data['Middle'] = data['last'].rolling(window=period).mean()
        data['Upper'] = data['Middle'] + atr_multiplier * data['ATR']
        data['Lower'] = data['Middle'] - atr_multiplier * data['ATR']
        return data

    # Apply the Keltner Channels function to the data
    df = keltner_channels(df)

    # Define the Relative Volatility Index (RVI) function
    def relative_volatility_index(data, period=14):
        # Calculate the difference between the Close and Open prices
        data['Upward Volatility'] = np.where(data['last'] > data['Open'], data['last'] - data['Open'], 0)
        data['Total Volatility'] = abs(data['Close'] - data['Open'])

        # Calculate the rolling sum of Upward Volatility and Total Volatility
        data['Upward Volatility Sum'] = data['Upward Volatility'].rolling(window=period).sum()
        data['Total Volatility Sum'] = data['Total Volatility'].rolling(window=period).sum()

        # Calculate RVI as the ratio of Upward Volatility to Total Volatility
        data['RVI'] = data['Upward Volatility Sum'] / data['Total Volatility Sum'] * 100

        return data

    # Apply the Relative Volatility Index (RVI) function to the data
    # df = relative_volatility_index(df)

    # Define the standard deviation function
    def standard_deviation(data, period=96):
        data['Std_Dev'] = data['last'].rolling(window=period).std()
        return data

    # Apply the standard deviation function to the data
    df = standard_deviation(df)

    return df


if __name__ == "__main__":
    market = "idc"
    df = read_csv_data("../data/idc_price_vector.csv", market)
    df = feature_engineering(df, market)
    print(df)
