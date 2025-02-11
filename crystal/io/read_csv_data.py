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

def read_csv_data(path, market):
    # Load data and set the index
    df = pd.read_csv(path)
    df.index = pd.to_datetime(df["von"])
    df = df[df.index.year == 2021]  # Filter data for year 2021

    # Generalized market-specific processing
    market_config = {
        "daa": {"produkt": "Stundenkontrakt", "columns": ["preis", "volumen"], "value_col": "preis"},
        "ida": {"columns": ["preis", "volumen"], "value_col": "preis"},
        "idc": {"produkt": "Viertelstundenkontrakt", "columns": ["high", "buy_vol", "sell_vol"], "value_col": "high"}
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
    freq = '15min' if market in ["ida", "idc"] else '1h'
    expected_length = 35040 if freq == '15min' else 8760
    full_range = pd.date_range(start=df["timestamp"].min(), end=df["timestamp"].max(), freq=freq)

    # Reindex and forward-fill missing values
    df = df.set_index("timestamp").reindex(full_range).ffill().reset_index().rename(columns={"index": "timestamp"})

    # Check length
    if len(df) != expected_length:
        raise ValueError(f"Dataframe has incorrect length after reindexing: {len(df)} entries for {market} market.")

    if df.empty:
        raise ValueError("Empty dataframe")

    return df

def feature_engineering(df, market):
    # Determine full-day frequency
    full_day_freq = 24 if market == "daa" else 96

    # Lag features
    for i in range(full_day_freq):
        df[f"lag_{i}"] = df["value"].shift(i)
    # df["lag_1"] = df["value"].shift(1)
    df["lag_day"] = df["value"].shift(full_day_freq)
    df["lag_week"] = df["value"].shift(7 * full_day_freq)

    # Moving averages and standard deviation
    df["rolling_mean_week"] = df["value"].rolling(window=7 * full_day_freq).mean()
    df["rolling_mean_day"] = df["value"].rolling(window=full_day_freq).mean()
    df["rolling_std_week"] = df["value"].rolling(window=7 * full_day_freq).std()
    df["rolling_std_day"] = df["value"].rolling(window=full_day_freq).std()

    # Date-based features
    df['Date'] = pd.to_datetime(df["timestamp"])
    # df['year'] = df.Date.dt.year
    df['month'] = df.Date.dt.month
    # df['day'] = df.Date.dt.day
    df['dayofweek'] = df.Date.dt.dayofweek
    # df['hour'] = df.Date.dt.hour
    # df['minute'] = df.Date.dt.minute
    df['weekend'] = df.Date.dt.weekday.isin([5, 6]).astype(float)

    # Cyclical encoding for time-based features
    df["sin_day_of_week"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["cos_day_of_week"] = np.cos(2 * np.pi * df["dayofweek"] / 7)

    if full_day_freq == 96:  # 15-minute intervals
        df["sin_hour_of_day"] = np.sin(2 * np.pi * df["minute"] / (96 * 15))
        df["cos_hour_of_day"] = np.cos(2 * np.pi * df["minute"] / (96 * 15))
    else:  # Hourly intervals
        df["sin_hour_of_day"] = np.sin(2 * np.pi * df["hour"] / full_day_freq)
        df["cos_hour_of_day"] = np.cos(2 * np.pi * df["hour"] / full_day_freq)

    # Drop rows with NaNs (from shifting/rolling)
    df.dropna(inplace=True)

    return df

if __name__ == "__main__":
    market = "idc"
    df = read_csv_data("../data/idc_price_vector.csv", market)
    df = feature_engineering(df, market)
    print(df)
