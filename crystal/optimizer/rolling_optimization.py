#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Small description of rolling_optimization

Copyright (c) by Institute for High Voltage Equipment and Grids,
Digitalization and Energy Economics (IAEW), RWTH Aachen University,
13.01.2025, s.kortmann. All rights reserved.
"""
import datetime

start_date = datetime.datetime(2020, 1, 1, 0, 0)
end_date = datetime.datetime(2020, 1, 3, 0, 0)

# 1. Create range of 96 time steps in 15 min
current_date = start_date
while current_date < end_date:
    current_date += datetime.timedelta(minutes=15)
    print(current_date)

    if current_date.hour == 12 and current_date.minute == 0:
        print("Step 1: DAA")
        # forecasted_prices = darts.forecast()
        # optimizer.step_1(forecasted_prices)

    if current_date.hour == 16 and current_date.minute == 0:
        print("Step 2: IDA")
