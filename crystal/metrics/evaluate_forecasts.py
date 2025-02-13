#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Small description of evaluate_forecasts

Copyright (c) by Institute for High Voltage Equipment and Grids,
Digitalization and Energy Economics (IAEW), RWTH Aachen University,
11.02.2025, s.kortmann. All rights reserved.
"""
import numpy as np
import pandas as pd
import sklearn

from crystal.metrics import picp_metric


def compute_metrics(
    ground_truth: pd.DataFrame, forecasts: pd.DataFrame, prediction_length: int
) -> pd.DataFrame:
    """
    Evaluate forecast performance using multiple time series metrics.

    Parameters:
    - ground_truth (pd.DataFrame): The actual observed values with 'timestamp' and 'value' columns.
    - forecasts (pd.DataFrame): The forecasted values with 'timestamp' and 'value' columns.
    - prediction_length (int): The length of the forecast horizon (e.g., 96 for 1 day).

    Returns:
    - pd.DataFrame: A DataFrame containing the computed metrics for the forecast.
    """
    if len(ground_truth) < prediction_length or len(forecasts) < prediction_length:
        raise ValueError(
            "Insufficient data for evaluation. Ensure the ground truth and forecast cover the prediction length."
        )

    # Align ground truth and forecast by timestamp
    merged = pd.merge(
        ground_truth, forecasts, on="timestamp", suffixes=("_actual", "_forecast")
    )

    # Check for alignment issues
    if merged.empty:
        raise ValueError(
            "No common timestamps found between ground truth and forecasts. Ensure proper alignment."
        )

    # Extract actual and predicted values
    y_actual = merged["value"].values
    y_forecast = merged["mean"].values

    # Compute metrics
    metrics = {
        "MAE": sklearn.metrics.mean_absolute_error(y_actual, y_forecast),
        "MSE": sklearn.metrics.mean_squared_error(y_actual, y_forecast),
        "RMSSE": sklearn.metrics.root_mean_squared_error(y_actual, y_forecast),
        "MAPE": sklearn.metrics.mean_absolute_percentage_error(y_actual, y_forecast),
    }

    for quantile in np.arange(0.1, 1, 0.1):
        # metrics[f'Quantile Loss {quantile}'] = sklearn.metrics.quantile_loss(y_actual, y_forecast, quantile)
        metrics[f"Pinball Loss {str(round(quantile, 2))}"] = (
            sklearn.metrics.mean_pinball_loss(
                y_actual, forecasts[str(round(quantile, 2))], alpha=quantile
            )
        )
    metrics[f"PICP"] = picp_metric.picp(y_actual, forecasts["0.1"], forecasts["0.9"])

    # Return metrics as a DataFrame
    return pd.DataFrame([metrics], columns=metrics.keys())
