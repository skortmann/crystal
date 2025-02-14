#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Small description of feature_analysis

Copyright (c) by Institute for High Voltage Equipment and Grids,
Digitalization and Energy Economics (IAEW), RWTH Aachen University,
20.01.2025, s.kortmann. All rights reserved.
"""
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import norm, weibull_min
import matplotlib.pyplot as plt

if "rwth-latex" in plt.style.available:
    plt.style.use(["rwth-latex", "ieee"])
plt.rcParams["font.size"] = 12
PROJECT_ROOT = Path(__file__).resolve().parents[2]

markets = ["day-ahead", "intraday-auction", "intraday-continuous"]

# Split data into separate DataFrames for each market
df_day_ahead = pd.read_csv(PROJECT_ROOT / "crystal/data/daa_price_vector_full.csv")
df_intraday_auction = pd.read_csv(
    PROJECT_ROOT / "crystal/data/ida_price_vector_full.csv"
)
df_intraday_continuous = pd.read_csv(
    PROJECT_ROOT / "crystal/data/idc_price_vector_full.csv"
)

df_day_ahead["timestamp"] = pd.to_datetime(df_day_ahead["timestamp"])
df_intraday_auction["timestamp"] = pd.to_datetime(df_intraday_auction["timestamp"])
df_intraday_continuous["timestamp"] = pd.to_datetime(
    df_intraday_continuous["timestamp"]
)

# filter for year 2021
df_day_ahead = df_day_ahead[df_day_ahead["timestamp"].dt.year == 2021]
df_intraday_auction = df_intraday_auction[
    df_intraday_auction["timestamp"].dt.year == 2021
]
df_intraday_continuous = df_intraday_continuous[
    df_intraday_continuous["timestamp"].dt.year == 2021
]

df_day_ahead.name = "day-ahead"
df_intraday_auction.name = "intraday-auction"
df_intraday_continuous.name = "intraday-continuous"


# Add time-based features
def add_time_features(df):
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    return df


df_day_ahead = add_time_features(df_day_ahead)
df_intraday_auction = add_time_features(df_intraday_auction)
df_intraday_continuous = add_time_features(df_intraday_continuous)


# Aggregate statistics per market
def calculate_market_features(df):
    column_name = [idx for idx in df.columns if "value" in idx][0]
    stats = {
        "mean_price": df[column_name].mean(),
        "std_price": df[column_name].std(),
        "min_price": df[column_name].min(),
        "max_price": df[column_name].max(),
        "price_skewness": df[column_name].skew(),
        "price_kurtosis": df[column_name].kurtosis(),
    }
    return stats


day_ahead_features = calculate_market_features(df_day_ahead)
intraday_auction_features = calculate_market_features(df_intraday_auction)
intraday_continuous_features = calculate_market_features(df_intraday_continuous)

# Output results
print("Day-Ahead Features:")
print(day_ahead_features)
print("\nIntraday Auction Features:")
print(intraday_auction_features)
print("\nIntraday Continuous Features:")
print(intraday_continuous_features)


# Add lag features for each market
def add_lag_features(df, lags=[1, 24]):
    column_name = [idx for idx in df.columns if "value" in idx][0]
    for lag in lags:
        df[f"price_lag_{lag}"] = df[column_name].shift(lag)
    return df


df_day_ahead = add_lag_features(df_day_ahead)
df_intraday_auction = add_lag_features(df_intraday_auction)
df_intraday_continuous = add_lag_features(df_intraday_continuous)


# Example: Calculate volatility (rolling standard deviation)
def add_rolling_features(df, window=24):
    column_name = [idx for idx in df.columns if "value" in idx][0]
    if df.name == "day-ahead":
        window = 24
    else:
        window = 96
    df["rolling_volatility"] = df[column_name].rolling(window).std()
    return df


df_day_ahead = add_rolling_features(df_day_ahead)
df_intraday_auction = add_rolling_features(df_intraday_auction)
df_intraday_continuous = add_rolling_features(df_intraday_continuous)


# Volatility Analysis
def analyze_volatility(df, market_name):
    volatility_stats = {
        "market": market_name,
        "average_volatility": df["rolling_volatility"].mean(),
        "max_volatility": df["rolling_volatility"].max(),
        "min_volatility": df["rolling_volatility"].min(),
    }
    return volatility_stats


volatility_day_ahead = analyze_volatility(df_day_ahead, "day-ahead")
volatility_intraday_auction = analyze_volatility(
    df_intraday_auction, "intraday-auction"
)
volatility_intraday_continuous = analyze_volatility(
    df_intraday_continuous, "intraday-continuous"
)

print("\nVolatility Analysis:")
print(volatility_day_ahead)
print(volatility_intraday_auction)
print(volatility_intraday_continuous)

# Plot distributions of rolling volatility
plt.figure(figsize=(6.6, 5))

for df, label, color in zip(
    [df_day_ahead, df_intraday_auction, df_intraday_continuous],
    ["Day-Ahead", "Intraday Auction", "Intraday Continuous"],
    ["#00549f", "#e30066", "#57ab27"],
):

    # Drop NaN values
    data = df["rolling_volatility"].dropna()
    # Plot histogram
    plt.hist(
        data, bins=50, alpha=0.5, density=True, label=f"{label} Histogram", color=color
    )

    # Fit and plot a normal distribution
    """
    mu, std = norm.fit(data)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, linestyle='--', label=f'{label} Fit ($\mu$={mu:.2f}, $\sigma$={std:.2f})', color=color)
    """
    # Fit and plot a Weibull distribution
    c, loc, scale = weibull_min.fit(data, floc=0)  # Fit Weibull distribution
    mean, var, skew, kurt = weibull_min.stats(c, moments="mvsk")
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = weibull_min.pdf(x, c, loc, scale)
    plt.plot(
        x,
        p,
        linestyle="--",
        label=f"{label} Weibull Fit (c={c:.2f}, scale={scale:.2f})",
        color=color,
    )

# Add plot details

# Add plot details
plt.title("Rolling Volatility Distribution by Market")
plt.xlabel("Volatility", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.legend(loc="upper right", fontsize=12)
# set x and y-ticks to fontsize 12
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlim(0)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(
    PROJECT_ROOT / "crystal/results/rolling_volatility_distribution.pdf", dpi=1200
)
