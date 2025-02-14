#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Small description of csv_parser

Copyright (c) by Institute for High Voltage Equipment and Grids,
Digitalization and Energy Economics (IAEW), RWTH Aachen University,
16.01.2025, s.kortmann. All rights reserved.
"""
import csv
import ast


def parse_csv_to_vector(filepath, key_column, value_column, filters=None, limit=None):
    vector = []
    with open(filepath, mode="r", newline="") as csvfile:
        csvreader = csv.DictReader(csvfile)
        for row in csvreader:
            if not filters or row[key_column] in filters:
                vector.append(float(row[value_column]))
    if limit and len(vector) > limit:
        vector = vector[:limit]
    return vector


# Define a function to convert string representations of lists into actual lists
def parse_list_column(column_value):
    try:
        return (
            ast.literal_eval(column_value)
            if isinstance(column_value, str)
            else column_value
        )
    except (ValueError, SyntaxError):
        return column_value  # If parsing fails, return original value
