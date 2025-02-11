#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Small description of picp

Copyright (c) by Institute for High Voltage Equipment and Grids, 
Digitalization and Energy Economics (IAEW), RWTH Aachen University, 
10.02.2025, s.kortmann. All rights reserved.
"""
import numpy as np

#falls nötig, für PICP-Berechnung
def picp(y_true, lower_bound, upper_bound):
    """
    Berechnet die Prediction Interval Coverage Probability (PICP).

    Parameters:
    - y_true: Tatsächliche Werte.
    - lower_bound: Untere Grenzen der Vorhersageintervalle.
    - upper_bound: Obere Grenzen der Vorhersageintervalle.

    Returns:
    - PICP-Wert.
    """
    y_true = np.array(y_true)
    lower_bound = np.array(lower_bound)
    upper_bound = np.array(upper_bound)

    n = len(y_true)
    coverage = ((y_true >= lower_bound) & (y_true <= upper_bound)).sum()
    return coverage / n

