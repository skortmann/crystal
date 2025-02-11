#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Small description of csv_parser

Copyright (c) by Institute for High Voltage Equipment and Grids, 
Digitalization and Energy Economics (IAEW), RWTH Aachen University, 
16.01.2025, s.kortmann. All rights reserved.
"""
import csv

def parse_csv_to_vector(filepath, key_column, value_column, filters=None, limit=None):
    vector = []
    with open(filepath, mode='r', newline='') as csvfile:
        csvreader = csv.DictReader(csvfile)
        for row in csvreader:
            if not filters or row[key_column] in filters:
                vector.append(float(row[value_column]))
    if limit and len(vector) > limit:
        vector = vector[:limit]
    return vector
