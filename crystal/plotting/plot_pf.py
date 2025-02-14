#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Small description of plot_pf

Copyright (c) by Institute for High Voltage Equipment and Grids,
Digitalization and Energy Economics (IAEW), RWTH Aachen University,
31.01.2025, s.kortmann. All rights reserved.
"""
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
from entsoe import EntsoePandasClient
from sklearn.metrics import mean_squared_error
import datetime as dt

# --- Settings ---
if "rwth-latex" in plt.style.available:
    plt.style.use(["rwth-latex", "blue"])
plt.rcParams["font.size"] = 12

PROJECT_ROOT = Path(__file__).resolve().parents[2]
data_file_path = PROJECT_ROOT / "crystal/data/daa_price_vector_full.csv"

# --- Step 1: Check if the data file exists ---
if os.path.exists(data_file_path):
    print("Loading data from local file...")
    day_ahead_prices = pd.read_csv(data_file_path, parse_dates=["timestamp"])
else:
    print("Downloading data from ENTSO-E...")
    # --- Step 2: Fetch day-ahead prices ---
    api_key = os.environ["ENTSOE_API_KEY"]
    client = EntsoePandasClient(api_key=api_key)
    start_date = "2021-12-31 00:00"
    end_date = "2024-01-30 23:45"
    country_code = "DE_LU"  # Germany - DE_LU region

    start = pd.Timestamp(start_date, tz="Europe/Berlin")
    end = pd.Timestamp(end_date, tz="Europe/Berlin")
    day_ahead_prices = client.query_day_ahead_prices(country_code, start=start, end=end)

    # Ensure proper frequency and forward-fill missing values if necessary
    day_ahead_prices = day_ahead_prices.asfreq("h").ffill()
    day_ahead_prices.index = day_ahead_prices.index.tz_localize(None)

    # Save to CSV
    print("Saving data to local file...")
    day_ahead_prices.to_csv(
        data_file_path, index=True, header=["value"], index_label="timestamp"
    )

# --- Step 3: Data Preprocessing ---
print("Data ready for AutoGluon-TS.")

# Convert to TimeSeriesDataFrame
day_ahead_prices = day_ahead_prices.assign(item_id=1)  # Avoid SettingWithCopyWarning
train_data = TimeSeriesDataFrame(day_ahead_prices)

print(train_data.head())

# Define where the model will be saved
model_dir = PROJECT_ROOT / "crystal/models/daa_model"

# Check if the model directory exists
if os.path.exists(model_dir):
    print("Loading the latest trained model...")
    # Load the existing predictor
    predictor = TimeSeriesPredictor.load(model_dir)

    # Optionally select the latest or best model
    best_model = predictor.model_best  # Select the best model (based on eval_metric)
    print(f"Using best model: {best_model}")

else:
    print("No existing model found. Training a new model...")
    # Train the predictor
    predictor = TimeSeriesPredictor(
        prediction_length=24,
        target="value",
        freq="h",
        eval_metric="WQL",
        path=model_dir,  # Save the model to this directory
    ).fit(train_data, presets="best_quality", time_limit=360)

    # Get the best model after training
    best_model = predictor.model_best
    print(f"New model trained: {best_model}")

from autogluon.timeseries.utils.forecast import (
    get_forecast_horizon_index_ts_dataframe,
)

future_index = get_forecast_horizon_index_ts_dataframe(train_data, prediction_length=24)
WEEKEND_INDICES = [5, 6]
future_timestamps = future_index.get_level_values("timestamp")
known_covariates = pd.DataFrame(index=future_index)
known_covariates["weekend"] = future_timestamps.weekday.isin(WEEKEND_INDICES).astype(
    float
)
known_covariates["year"] = future_timestamps.year
known_covariates["month"] = future_timestamps.month
known_covariates["day"] = future_timestamps.day
known_covariates["dayofweek"] = future_timestamps.dayofweek
known_covariates["hour"] = future_timestamps.hour
known_covariates["minute"] = future_timestamps.minute

# Cyclical encoding for time-based features
known_covariates["sin_day_of_week"] = np.sin(
    2 * np.pi * known_covariates["dayofweek"] / 7
)
known_covariates["cos_day_of_week"] = np.cos(
    2 * np.pi * known_covariates["dayofweek"] / 7
)

known_covariates["sin_hour_of_day"] = np.sin(2 * np.pi * known_covariates["hour"] / 24)
known_covariates["cos_hour_of_day"] = np.cos(2 * np.pi * known_covariates["hour"] / 24)


# Generate predictions using the best model
def print_all_model_stats(predictor):
    leaderboard = predictor.leaderboard(train_data)
    print(leaderboard)
    models = leaderboard["model"]
    for model in models:
        print(
            predictor.predict(
                train_data, model=model, known_covariates=known_covariates
            )
        )


predictions = predictor.predict(
    train_data, model=best_model, known_covariates=known_covariates
)

# Extract prediction mean and all quantiles for plotting
mean_prediction = predictions["mean"]
quantiles = [col for col in predictions.columns if col not in ["mean"]]


def plot_forecast_error(
    train_data, day_ahead_prices, mean_prediction, predictions, quantiles
):
    # Calculate the Mean Squared Error (MSE) on the forecast horizon
    actual_values = train_data["value"].iloc[-24:].values
    forecasted_values = mean_prediction.values[:24]
    mse = mean_squared_error(actual_values, forecasted_values)

    # Correct the actual data plot using the index as timestamps
    timestamps = train_data["value"].unstack().iloc[0].index

    # Plot actual data and forecast
    plt.figure(figsize=(10, 6))

    # Plot the actual day-ahead prices
    plt.plot(
        timestamps[-24 * 7 :],
        train_data["value"].unstack().iloc[0][-24 * 7 :].values,
        label="Actual Prices",
        color="blue",
    )

    # Plot forecast mean
    forecast_time_range = pd.date_range(
        start=day_ahead_prices.index[-1], periods=24, freq="h"
    )
    plt.plot(
        forecast_time_range,
        mean_prediction,
        label=f"Forecast Mean (MSE: {mse:.2f})",
        color="orange",
    )

    # Plot all quantiles as shaded regions
    for q in quantiles:
        plt.fill_between(
            forecast_time_range,
            predictions[q],
            mean_prediction,
            alpha=0.25,
            label=f"Quantile: {q}",
        )

    # Customize the legend placement
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=True)
    plt.xlabel("Time")
    plt.ylabel("Price (€/MWh)")
    plt.title("Probabilistic Forecast of Day-Ahead Electricity Prices")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    # dtFmt = mdates.DateFormatter("%d %H") # define the formatting
    # plt.gca().xaxis.set_major_formatter(dtFmt) # apply the format to the desired axis
    plt.savefig("day_ahead_price_forecast_with_all_quantiles.png")
    plt.show()


# --- Built-in Plot Function ---
def use_autogluon_plot(train_data, predictions):
    print("\nGenerating the built-in plot...")
    # Limit test data for better plot visualization
    train_data = train_data.tail(24 * 7)

    # Built-in AutoGluon plot with quantiles
    fig = predictor.plot(
        train_data,
        predictions,
        quantile_levels=[i / 10 for i in range(1, 10)],  # Show 10% and 90% quantiles
        max_history_length=24 * 7,
        max_num_item_ids=1,  # Plotting one item (day_ahead_prices)
    )
    # Modify the figure (add a title)
    fig.suptitle(
        "AutoGluon-TS Forecast: Day-Ahead Electricity Prices",
        fontsize=14,
        fontweight="bold",
    )

    # Adjust spacing to prevent title overlap
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    # Show and save the plot
    plt.show()
    fig.savefig("autogluon_ts_forecast_customized.png", bbox_inches="tight")


# --- Step 1: Subtract 0.5 quantile (mean prediction) from all quantiles ---
hours = 24
mean_prediction = predictions["0.5"]
normalized_predictions = {}

for q in quantiles:
    if float(q) > 0.5:  # Quantiles above 0.5
        normalized_predictions[q] = predictions[q] - mean_prediction
    elif float(q) < 0.5:  # Quantiles below 0.5
        normalized_predictions[q] = mean_prediction - predictions[q]

# Set the mean prediction (0.5 quantile) to 0
normalized_predictions["0.5"] = np.zeros(hours)  # Center at 0
normalized_df = pd.DataFrame(normalized_predictions)

# --- Step 2: Create the color palette ---
cmap = plt.get_cmap("viridis", hours)  # Discrete color map with 24 distinct colors

# --- Create primary figure ---
fig, ax1 = plt.subplots(figsize=(6.6, 5))

for hour in range(hours):
    color = cmap(hour)

    x_neg = [
        float(q) for q in predictions.columns if q not in ["mean"] if float(q) >= 0.5
    ]
    y_neg = [
        normalized_df[q].iloc[hour]
        for q in predictions.columns
        if q not in ["mean"]
        if float(q) >= 0.5
    ]

    # Define the x and y values for quantiles < 0.5 (positive direction)
    x_pos = [
        float(q) for q in predictions.columns if q not in ["mean"] if float(q) <= 0.5
    ]
    y_pos = [
        normalized_df[q].iloc[hour]
        for q in predictions.columns
        if q not in ["mean"]
        if float(q) <= 0.5
    ]

    ax1.plot(x_neg, y_neg, marker="o", linestyle="-", color=color, label=f"Hour {hour}")
    ax1.plot(x_pos, y_pos, marker="o", linestyle="-", color=color)

# --- Format primary x-axis (Quantiles) ---
ax1.set_xlabel("Quantiles")
ax1.set_ylabel("Deviation from 0.5 Quantile (€/MWh)")
ax1.set_title("Prediction Uncertainty for Forecasted Hours")
ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.5), ncol=6, frameon=True)
ax1.set_ylim(0, 14)
ax1.set_xlim(0, 1)
ax1.grid(True)

# --- Create secondary x-axis (Power) ---
ax2 = ax1.twiny()

# Define tick positions and labels for power (example: charge/discharge levels)
quantile_ticks = [0.1, 0.3, 0.5, 0.7, 0.9]  # Quantiles
power_ticks = [-1, -0.5, 0, 0.5, 1]  # Example power levels in kW

ax2.set_xticks(quantile_ticks)
ax2.set_xticklabels(power_ticks)

# Position the second x-axis at the bottom
ax2.xaxis.set_ticks_position("bottom")
ax2.xaxis.set_label_position("bottom")
ax2.spines["bottom"].set_position(("outward", 36))

ax2.set_xlabel("Battery Power $P_{batt}$")

# Save figure
plt.tight_layout()
plt.savefig(PROJECT_ROOT / "crystal/results/quantiles_distribution.pdf", dpi=1200)
# plt.show()
