#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Small description of SequentialEnergyArbitrage

Copyright (c) by Institute for High Voltage Equipment and Grids, 
Digitalization and Energy Economics (IAEW), RWTH Aachen University, 
11.02.2025, s.kortmann. All rights reserved.
"""
import gurobipy as gp

class SequentialEnergyArbitrage:
    def __init__(self,
                 energy_capacity: int = 100,
                 power_capacity: int = 10,
                 n_cycles: int = 1):
        self.risk_factor = 1
        self.energy_capacity = energy_capacity
        self.power_capacity = power_capacity
        self.n_cycles = n_cycles

        self.volume_limit = self.energy_capacity * self.n_cycles

    def optimizeDAA(self):

        model_daa = gp.Model("DAA")

        # Set parameters
        HOURS = 24
        QUARTERS = 4

        # Create variables
        soc = model_daa.addVars(HOURS, QUARTERS, lb=0, ub=1, vtype=gp.GRB.CONTINUOUS, name="soc" + str(HOURS) + str(QUARTERS))

        model_daa.update()

        print(model_daa.display())

if __name__ == "__main__":
    sea = SequentialEnergyArbitrage()
    sea.optimizeDAA()