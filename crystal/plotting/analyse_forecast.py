#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Small description of analyse_forecast

Copyright (c) by Institute for High Voltage Equipment and Grids,
Digitalization and Energy Economics (IAEW), RWTH Aachen University,
16.01.2025, s.kortmann. All rights reserved.
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates  # Import for formatting x-axis dates

if "rwth-latex" in plt.style.available:
    plt.style.use(["rwth-latex", "blue"])
plt.rcParams["font.size"] = 12

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def plot_quantile_forecasts(forecasted_df, real_df):
    fig, ax = plt.subplots(figsize=(6.6, 5))
    quantiles = ["0.1", "0.9"]

    # Plot quantiles
    for quantile in quantiles:
        ax.plot(forecasted_df.index, forecasted_df[quantile])

    # Fill between quantiles with transparency
    for quantile in quantiles:
        ax.fill_between(
            forecasted_df.index,
            forecasted_df[quantile],
            forecasted_df["0.5"],
            alpha=0.5,
            label=f"{int(float(quantile) * 100)}-50\%-Quantile",
        )

    # Plot mean (point forecast)
    ax.plot(
        forecasted_df.index,
        forecasted_df["mean"],
        label="Point Forecast",
        color="black",
        linestyle="--",
    )
    ax.plot(
        real_df.index,
        real_df["value"],
        label="Actual Price",
        color="red",
        linestyle="--",
    )

    # Set x-label, y-label, and title
    plt.xlabel("Time", fontsize=12)
    plt.ylabel("Price", fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.title("Forecasted Day-Ahead Price for one Week in Year 2022", fontsize=12)

    # Format x-axis datetime labels
    # ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m %H:%M'))  # Format: DD-MM HH:MM
    # ax.xaxis.set_major_locator(mdates.HourLocator(interval=24))  # Show label every 6 hours
    # plt.xticks(rotation=45)  # Rotate labels for better readability

    ax.set_xlim(
        [forecasted_df.index[0], forecasted_df.index[-1]]
    )  # Set x-axis limits to start and end of data

    # Set legend position to top-right
    plt.legend(fontsize=12, loc="lower right")

    plt.savefig(PROJECT_ROOT / "crystal/results/forecast_ida.pdf", dpi=1200)
    plt.show()


if __name__ == "__main__":
    forecasted_df = pd.read_csv(
        PROJECT_ROOT / "crystal/results/daa_forecast_results.csv"
    )
    forecasted_df["timestamp"] = pd.to_datetime(forecasted_df["timestamp"])
    forecasted_df.set_index("timestamp", inplace=True)
    forecasted_df = forecasted_df.iloc[24 * 7 : 2 * 24 * 7]

    real_df = pd.read_csv(PROJECT_ROOT / "crystal/data/daa_price_vector_full.csv")
    real_df["timestamp"] = pd.to_datetime(real_df["timestamp"])
    real_df.set_index("timestamp", inplace=True)
    real_df = real_df.loc[forecasted_df.index[0] : forecasted_df.index[-1]]

    forecasted_df.reset_index(drop=True, inplace=True)
    real_df.reset_index(drop=True, inplace=True)

    plot_quantile_forecasts(forecasted_df, real_df)
