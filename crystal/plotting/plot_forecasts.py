#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Small description of plot_forecasts

Copyright (c) by Institute for High Voltage Equipment and Grids,
Digitalization and Energy Economics (IAEW), RWTH Aachen University,
11.02.2025, s.kortmann. All rights reserved.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Function to plot forecast results with quantiles and actual values
def plot_forecast_results(actual_df, forecast_df, market, save_path=None):
    """
    Plots forecasted quantiles along with actual values.

    Parameters:
    - forecast_df (pd.DataFrame): DataFrame containing forecasted values and quantiles.
    - actual_df (pd.DataFrame): DataFrame containing actual market values.
    - market (str): Market name for title and labeling.
    - save_path (str, optional): Path to save the plot if provided.
    """

    # Convert timestamps to datetime
    forecast_df["timestamp"] = pd.to_datetime(forecast_df["timestamp"])
    actual_df["timestamp"] = pd.to_datetime(actual_df["timestamp"])

    # Extract quantiles
    quantiles = [f"{q:.1f}" for q in np.arange(0.1, 1.0, 0.1)]
    mean_forecast = forecast_df["mean"]
    lower_bound = forecast_df["0.1"]
    upper_bound = forecast_df["0.9"]

    # Create figure
    plt.figure(figsize=(12, 6), dpi=1200)

    # Plot actual values
    plt.plot(
        actual_df["timestamp"],
        actual_df["value"],
        label="Actual Values",
        color="black",
        linestyle="dashed",
        alpha=0.75,
    )

    # Plot mean forecast
    plt.plot(
        forecast_df["timestamp"],
        mean_forecast,
        label="Forecast Mean",
        color="blue",
        alpha=0.75,
    )

    # Fill between quantiles
    plt.fill_between(
        forecast_df["timestamp"],
        lower_bound,
        upper_bound,
        color="blue",
        alpha=0.2,
        label="90% Prediction Interval",
    )

    # Labels and title
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.title(f"Forecasted vs Actual Prices for {market.upper()}")
    plt.legend()
    plt.grid()

    # Save plot if path is provided
    if save_path:
        plt.savefig(save_path, dpi=1200, bbox_inches="tight")

    # Close plot
    plt.close()
