#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Small description of custom_metric

Copyright (c) by Institute for High Voltage Equipment and Grids, 
Digitalization and Energy Economics (IAEW), RWTH Aachen University, 
11.02.2025, s.kortmann. All rights reserved.
"""
import numpy as np

def custom_quadratic_penalty_metric(y_true, y_pred):
    """
    Custom metric that applies quadratic penalties to deviations from the 0.5 quantile (median).
    The penalty increases as deviations fall outside increasing quantile thresholds.

    Parameters:
    - y_true: Actual values (ground truth)
    - y_pred: Predicted values

    Returns:
    - Mean penalty value (scalar)
    """
    # Deviation between actual and predicted values
    deviation = np.abs(y_true - y_pred)

    # Quantile thresholds from 0.1 to 0.9
    quantile_levels = np.arange(0.1, 1.0, 0.1)
    quantiles = np.quantile(y_pred, quantile_levels)

    # Initialize penalties
    penalties = np.zeros_like(deviation)

    # Apply quadratic penalties based on quantile bands
    for i in range(len(quantiles) - 1):
        lower_quantile = quantiles[i]
        upper_quantile = quantiles[i + 1]
        penalty_factor = (i + 1) ** 2  # Quadratic growth in penalty factor

        # Penalize deviations within each quantile band
        penalties += np.where((deviation > lower_quantile) & (deviation <= upper_quantile), (deviation**2) * penalty_factor, 0)

    # Severe penalty for deviations beyond the highest quantile (outside 0.1 or 0.9 quantiles)
    penalties += np.where((deviation > quantiles[-1]), (deviation**2) * 10, 0)

    # Return the mean penalty
    return np.mean(penalties)