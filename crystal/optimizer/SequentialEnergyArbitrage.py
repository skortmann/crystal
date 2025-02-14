#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Small description of SequentialEnergyArbitrage

Copyright (c) by Institute for High Voltage Equipment and Grids,
Digitalization and Energy Economics (IAEW), RWTH Aachen University,
11.02.2025, s.kortmann. All rights reserved.
"""
import gurobipy as gp

gp.setParam("OutputFlag", 0)
import numpy as np

np.seed = 42

DAY_IN_HOURS = 24
TIME_STEPS = 96


class SequentialEnergyArbitrage:
    def __init__(
        self,
        energy_capacity: int = 1,
        power_capacity: int = 1,
        n_cycles: int = 1,
        risk_factor: float = 0.25,
        optimization_method="Risk High",
        cyclic_constraint=True,
        objective="simple",
    ):

        self.optimization_method = optimization_method
        self.risk_factor = risk_factor
        self.objective = objective

        self.energy_capacity = energy_capacity
        self.power_capacity = power_capacity
        self.efficiency = 0.95

        self.cyclic_constraint = cyclic_constraint
        self.n_cycles = n_cycles
        self.volume_limit = self.energy_capacity * self.n_cycles
        self.dt = DAY_IN_HOURS / TIME_STEPS

    def optimizeDAA(self, quantile_forecasts, daa_price_vector_true):

        # Ensure forecasts are provided for all required quantiles
        if len(quantile_forecasts) != 9:
            raise ValueError("Expected 9 quantile forecasts (0.1 to 0.9).")

        # Unpack the quantile forecasts
        (
            daa_price_vector_01,
            daa_price_vector_02,
            daa_price_vector_03,
            daa_price_vector_04,
            daa_price_vector_05,
            daa_price_vector_06,
            daa_price_vector_07,
            daa_price_vector_08,
            daa_price_vector_09,
        ) = quantile_forecasts

        model_daa = gp.Model("DAA")

        # Set variables
        soc_daa = {
            t: model_daa.addVar(
                lb=0, ub=1, vtype=gp.GRB.CONTINUOUS, name=f"soc_daa_{t}"
            )
            for t in range(TIME_STEPS)
        }
        p_charge_daa = {
            t: model_daa.addVar(
                lb=0.1,
                ub=self.power_capacity,
                vtype=gp.GRB.SEMICONT,
                name=f"p_charge_daa_{t}",
            )
            for t in range(TIME_STEPS)
        }
        p_discharge_daa = {
            t: model_daa.addVar(
                lb=0.1,
                ub=self.power_capacity,
                vtype=gp.GRB.SEMICONT,
                name=f"p_discharge_daa_{t}",
            )
            for t in range(TIME_STEPS)
        }
        p_batt_daa = {
            t: model_daa.addVar(
                lb=-self.power_capacity,
                ub=self.power_capacity,
                vtype=gp.GRB.CONTINUOUS,
                name=f"p_batt_daa_{t}",
            )
            for t in range(TIME_STEPS)
        }

        x_bin_charge_daa = {
            t: model_daa.addVar(vtype=gp.GRB.BINARY, name=f"x_bin_charge_daa_{t}")
            for t in range(TIME_STEPS)
        }
        x_bin_discharge_daa = {
            t: model_daa.addVar(vtype=gp.GRB.BINARY, name=f"x_bin_discharge_daa_{t}")
            for t in range(TIME_STEPS)
        }

        # Set constraints
        # Correcting SOC update equation
        model_daa.addConstrs(
            (
                soc_daa[t]
                == soc_daa[t - 1]
                + self.dt
                * (
                    (p_charge_daa[t] * self.efficiency)
                    - (p_discharge_daa[t] / self.efficiency)
                )
                / self.energy_capacity
                for t in range(1, TIME_STEPS)
            ),
            name="soc_update",
        )
        model_daa.addConstr((soc_daa[0] == 0.5), name="initial_soc")
        model_daa.addConstr((soc_daa[TIME_STEPS - 1] == 0.5), name="final_soc")

        model_daa.addConstrs(
            (
                p_charge_daa[t] <= self.power_capacity * x_bin_charge_daa[t]
                for t in range(TIME_STEPS)
            ),
            name="charge_limit",
        )
        model_daa.addConstrs(
            (
                p_discharge_daa[t] <= self.power_capacity * x_bin_discharge_daa[t]
                for t in range(TIME_STEPS)
            ),
            name="discharge_limit",
        )
        model_daa.addConstrs(
            x_bin_charge_daa[t] + x_bin_discharge_daa[t] <= 1 for t in range(TIME_STEPS)
        )

        model_daa.addConstrs(
            (
                p_batt_daa[t] == p_charge_daa[t] - p_discharge_daa[t]
                for t in range(TIME_STEPS)
            ),
            name="batt_power",
        )

        model_daa.addConstrs(
            (
                p_charge_daa[t] + p_discharge_daa[t] <= self.power_capacity
                for t in range(TIME_STEPS)
            ),
            name="power_limit",
        )

        if self.cyclic_constraint:
            WINDOW_SIZE = int(TIME_STEPS / 6)  # 6 hours

            model_daa.addConstrs(
                gp.quicksum(
                    p_charge_daa[t + k]
                    for k in range(WINDOW_SIZE)
                    if t + k < TIME_STEPS
                )
                <= (self.volume_limit / self.dt) / 6
                for t in range(0, TIME_STEPS - WINDOW_SIZE, 12)  # Apply every 3 hours
            )

            model_daa.addConstrs(
                gp.quicksum(
                    p_discharge_daa[t + k]
                    for k in range(WINDOW_SIZE)
                    if t + k < TIME_STEPS
                )
                <= (self.volume_limit / self.dt) / 6
                for t in range(0, TIME_STEPS - WINDOW_SIZE, 12)
            )

            model_daa.addConstr(
                gp.quicksum(p_charge_daa[t] for t in range(TIME_STEPS))
                <= self.volume_limit / self.dt,
                name="volume_limit",
            )
            model_daa.addConstr(
                gp.quicksum(p_discharge_daa[t] for t in range(TIME_STEPS))
                <= self.volume_limit / self.dt,
                name="volume_limit",
            )

        # Ensure every 4 consecutive charging powers are the same
        model_daa.addConstrs(
            (
                p_charge_daa[t] == p_charge_daa[t + 1]
                for t in range(TIME_STEPS - 1)
                if t % 4 != 0
            ),
            name="charging_consistency",
        )

        # Ensure every 4 consecutive discharging powers are the same
        model_daa.addConstrs(
            (
                p_discharge_daa[t] == p_discharge_daa[t + 1]
                for t in range(TIME_STEPS - 1)
                if t % 4 != 0
            ),
            name="discharging_consistency",
        )

        if self.risk_factor == "adaptive":
            # Estimate potential profit from arbitrage (difference between min and max quantile forecasts)
            profit_opportunity = np.max(daa_price_vector_05) - np.min(
                daa_price_vector_05
            )

            # Normalize using a logistic function to keep risk factor in range [0,1]
            self.risk_factor = 1 / (
                1 + np.exp(-profit_opportunity / np.mean(daa_price_vector_05))
            )

        # print("Risk factor in DAA-Market: ", self.risk_factor)

        # Initialize objective as a list of terms
        objective_terms = []

        # Basic revenue maximization (Simple objective)
        objective_terms.append(
            gp.quicksum(
                self.dt
                * daa_price_vector_05[t]
                * (p_discharge_daa[t] - p_charge_daa[t])
                for t in range(TIME_STEPS)
            )
        )

        # Add risk-aware term if applicable
        if "risk-aware" in self.objective.lower():
            objective_terms.append(
                gp.quicksum(
                    self.dt
                    * (
                        -self.risk_factor
                        * (daa_price_vector_09[t] - daa_price_vector_05[t])
                        * p_charge_daa[t]
                        - self.risk_factor
                        * (daa_price_vector_05[t] - daa_price_vector_01[t])
                        * p_discharge_daa[t]
                    )
                    for t in range(TIME_STEPS)
                )
            )

        # Add more terms based on the selected objective strategy (e.g., smooth operation, penalty terms)
        if "smooth" in self.objective.lower():
            objective_terms.append(
                gp.quicksum(
                    -0.01 * (p_charge_daa[t] - p_charge_daa[t - 1]) ** 2
                    - 0.01 * (p_discharge_daa[t] - p_discharge_daa[t - 1]) ** 2
                    for t in range(1, TIME_STEPS)  # Start at t=1 to avoid index error
                )
            )

        if "piece-wise" in self.objective.lower():
            # Auxiliary variables for piecewise-linear transformation
            z_charge = {
                t: model_daa.addVar(lb=0, vtype=gp.GRB.CONTINUOUS, name=f"z_charge_{t}")
                for t in range(TIME_STEPS)
            }
            z_discharge = {
                t: model_daa.addVar(
                    lb=0, vtype=gp.GRB.CONTINUOUS, name=f"z_discharge_{t}"
                )
                for t in range(TIME_STEPS)
            }

            for t in range(TIME_STEPS):
                # PWL function for discharging
                model_daa.addGenConstrPWL(
                    p_discharge_daa[t],
                    z_discharge[t],
                    [
                        0,
                        0.2 * self.power_capacity,
                        0.4 * self.power_capacity,
                        0.6 * self.power_capacity,
                        0.8 * self.power_capacity,
                        1 * self.power_capacity,
                    ],
                    [
                        0,
                        self.risk_factor
                        * self.dt
                        * (daa_price_vector_05[t] - daa_price_vector_04[t]),
                        self.risk_factor
                        * self.dt
                        * (daa_price_vector_05[t] - daa_price_vector_03[t]),
                        self.risk_factor
                        * self.dt
                        * (daa_price_vector_05[t] - daa_price_vector_02[t]),
                        self.risk_factor
                        * self.dt
                        * (daa_price_vector_05[t] - daa_price_vector_01[t]),
                        self.risk_factor * self.dt * (daa_price_vector_05[t]),
                    ],
                    name=f"PWL_Discharge_{t}",
                )

                # PWL function for charging
                model_daa.addGenConstrPWL(
                    p_charge_daa[t],
                    z_charge[t],
                    [
                        0,
                        0.2 * self.power_capacity,
                        0.4 * self.power_capacity,
                        0.6 * self.power_capacity,
                        0.8 * self.power_capacity,
                        1 * self.power_capacity,
                    ],
                    [
                        0,
                        self.risk_factor
                        * self.dt
                        * (daa_price_vector_06[t] - daa_price_vector_05[t]),
                        self.risk_factor
                        * self.dt
                        * (daa_price_vector_07[t] - daa_price_vector_05[t]),
                        self.risk_factor
                        * self.dt
                        * (daa_price_vector_08[t] - daa_price_vector_05[t]),
                        self.risk_factor
                        * self.dt
                        * (daa_price_vector_09[t] - daa_price_vector_05[t]),
                        self.risk_factor * self.dt * (daa_price_vector_05[t]),
                    ],
                    name=f"PWL_Charge_{t}",
                )

            objective_terms.append(
                gp.quicksum(-(z_discharge[t] + z_charge[t]) for t in range(TIME_STEPS))
            )

        if self.objective == "perfect-foresight":
            model_daa.setObjective(
                gp.quicksum(
                    daa_price_vector_true[t] * (p_discharge_daa[t] - p_charge_daa[t])
                    for t in range(TIME_STEPS)
                ),
                sense=gp.GRB.MAXIMIZE,
            )
        else:
            # Combine all selected terms
            model_daa.setObjective(
                gp.quicksum(objective_terms),  # Summing all terms dynamically
                sense=gp.GRB.MAXIMIZE,
            )

        model_daa.optimize()

        self.results_daa = {
            "p_charge_daa": [p_charge_daa[t].x for t in range(TIME_STEPS)],
            "p_discharge_daa": [p_discharge_daa[t].x for t in range(TIME_STEPS)],
            "p_batt_daa": [p_batt_daa[t].x for t in range(TIME_STEPS)],
            "soc_daa": [soc_daa[t].x for t in range(TIME_STEPS)],
            "x_bin_charge_daa": [x_bin_charge_daa[t].x for t in range(TIME_STEPS)],
            "x_bin_discharge_daa": [
                x_bin_discharge_daa[t].x for t in range(TIME_STEPS)
            ],
        }

        return self.results_daa

    def optimizeIDA(self, quantile_forecasts, results_daa, ida_price_vector_true):

        # Ensure forecasts are provided for all required quantiles
        if len(quantile_forecasts) != 9:
            raise ValueError("Expected 9 quantile forecasts (0.1 to 0.9).")

        # Unpack the quantile forecasts
        (
            ida_price_vector_01,
            ida_price_vector_02,
            ida_price_vector_03,
            ida_price_vector_04,
            ida_price_vector_05,
            ida_price_vector_06,
            ida_price_vector_07,
            ida_price_vector_08,
            ida_price_vector_09,
        ) = quantile_forecasts

        model_ida = gp.Model("IDA")

        # Set variables
        soc_ida = {
            t: model_ida.addVar(
                lb=0, ub=1, vtype=gp.GRB.CONTINUOUS, name=f"soc_ida_{t}"
            )
            for t in range(TIME_STEPS)
        }
        p_charge_ida = {
            t: model_ida.addVar(
                lb=0.1,
                ub=self.power_capacity,
                vtype=gp.GRB.SEMICONT,
                name=f"p_charge_ida_{t}",
            )
            for t in range(TIME_STEPS)
        }
        p_discharge_ida = {
            t: model_ida.addVar(
                lb=0.1,
                ub=self.power_capacity,
                vtype=gp.GRB.SEMICONT,
                name=f"p_discharge_ida_{t}",
            )
            for t in range(TIME_STEPS)
        }
        p_charge_ida_close = {
            t: model_ida.addVar(
                lb=0.1,
                ub=self.power_capacity,
                vtype=gp.GRB.SEMICONT,
                name=f"p_charge_ida_close_{t}",
            )
            for t in range(TIME_STEPS)
        }
        p_discharge_ida_close = {
            t: model_ida.addVar(
                lb=0.1,
                ub=self.power_capacity,
                vtype=gp.GRB.SEMICONT,
                name=f"p_discharge_ida_close_{t}",
            )
            for t in range(TIME_STEPS)
        }

        p_batt_ida = {
            t: model_ida.addVar(
                lb=-self.power_capacity,
                ub=self.power_capacity,
                vtype=gp.GRB.CONTINUOUS,
                name=f"p_batt_{t}",
            )
            for t in range(TIME_STEPS)
        }

        x_bin_charge_ida = {
            t: model_ida.addVar(vtype=gp.GRB.BINARY, name=f"x_bin_charge_{t}")
            for t in range(TIME_STEPS)
        }
        x_bin_discharge_ida = {
            t: model_ida.addVar(vtype=gp.GRB.BINARY, name=f"x_bin_discharge_{t}")
            for t in range(TIME_STEPS)
        }

        # Set constraints
        # Correcting SOC update equation
        model_ida.addConstrs(
            (
                soc_ida[t]
                == soc_ida[t - 1]
                + self.dt
                * (
                    (p_charge_ida[t] * self.efficiency)
                    + p_charge_ida_close[t]
                    + results_daa["p_charge_daa"][t] * self.efficiency
                    - (p_discharge_ida[t] / self.efficiency)
                    - p_discharge_ida_close[t]
                    - results_daa["p_discharge_daa"][t] / self.efficiency
                )
                / self.energy_capacity
                for t in range(1, TIME_STEPS)
            ),
            name="soc_update",
        )
        model_ida.addConstr((soc_ida[0] == 0.5), name="initial_soc")
        model_ida.addConstr((soc_ida[TIME_STEPS - 1] == 0.5), name="final_soc")

        model_ida.addConstrs(
            (
                p_charge_ida[t] <= self.power_capacity * x_bin_charge_ida[t]
                for t in range(TIME_STEPS)
            ),
            name="charge_limit",
        )
        model_ida.addConstrs(
            (
                p_discharge_ida[t] <= self.power_capacity * x_bin_discharge_ida[t]
                for t in range(TIME_STEPS)
            ),
            name="discharge_limit",
        )
        model_ida.addConstrs(
            x_bin_charge_ida[t] + x_bin_discharge_ida[t] <= 1 for t in range(TIME_STEPS)
        )

        model_ida.addConstrs(
            (
                p_batt_ida[t] == p_charge_ida[t] - p_discharge_ida[t]
                for t in range(TIME_STEPS)
            ),
            name="batt_power",
        )

        model_ida.addConstrs(
            (
                p_charge_ida[t] + p_discharge_ida[t] <= self.power_capacity
                for t in range(TIME_STEPS)
            ),
            name="power_limit",
        )

        if self.cyclic_constraint:
            WINDOW_SIZE = int(TIME_STEPS / 6)  # 6 hours

            model_ida.addConstrs(
                gp.quicksum(
                    p_charge_ida[t + k]
                    + results_daa["p_charge_daa"][t + k]
                    - p_discharge_ida_close[t + k]
                    for k in range(WINDOW_SIZE)
                    if t + k < TIME_STEPS
                )
                <= (self.volume_limit / self.dt) / 6
                for t in range(0, TIME_STEPS - WINDOW_SIZE, 12)  # Apply every 3 hours
            )

            model_ida.addConstrs(
                gp.quicksum(
                    p_discharge_ida[t + k]
                    + results_daa["p_discharge_daa"][t + k]
                    - p_charge_ida_close[t + k]
                    for k in range(WINDOW_SIZE)
                    if t + k < TIME_STEPS
                )
                <= (self.volume_limit / self.dt) / 6
                for t in range(0, TIME_STEPS - WINDOW_SIZE, 12)
            )

            model_ida.addConstr(
                (
                    gp.quicksum(
                        (
                            p_charge_ida[t]
                            + results_daa["p_charge_daa"][t]
                            - p_discharge_ida_close[t]
                        )
                        for t in range(TIME_STEPS)
                    )
                    <= self.volume_limit / self.dt
                ),
                name="volume_limit",
            )
            model_ida.addConstr(
                (
                    gp.quicksum(
                        (
                            p_discharge_ida[t]
                            + results_daa["p_discharge_daa"][t]
                            - p_charge_ida_close[t]
                        )
                        for t in range(TIME_STEPS)
                    )
                    <= self.volume_limit / self.dt
                ),
                name="volume_limit",
            )

        model_ida.addConstrs(
            (
                p_charge_ida_close[t] <= results_daa["p_discharge_daa"][t]
                for t in range(TIME_STEPS)
            ),
            name="close_charge",
        )
        model_ida.addConstrs(
            (
                p_discharge_ida_close[t] <= results_daa["p_charge_daa"][t]
                for t in range(TIME_STEPS)
            ),
            name="close_discharge",
        )

        model_ida.addConstrs(
            (
                p_charge_ida[t] + results_daa["p_charge_daa"][t] <= self.power_capacity
                for t in range(TIME_STEPS)
            ),
            name="close_charge_limit",
        )
        model_ida.addConstrs(
            (
                p_discharge_ida[t] + results_daa["p_discharge_daa"][t]
                <= self.power_capacity
                for t in range(TIME_STEPS)
            ),
            name="close_discharge_limit",
        )

        if "Adaptive" in self.optimization_method:
            # Estimate potential profit from arbitrage (difference between min and max quantile forecasts)
            profit_opportunity = np.max(ida_price_vector_05) - np.min(
                ida_price_vector_05
            )

            # Normalize using a logistic function to keep risk factor in range [0,1]
            self.risk_factor = 1 / (
                1 + np.exp(-profit_opportunity / np.mean(ida_price_vector_05))
            )

        # print("Risk factor in IDA-Market: ", self.risk_factor)

        # Initialize objective as a list of terms
        objective_terms = []

        # Basic revenue maximization (Simple objective)
        objective_terms.append(
            gp.quicksum(
                self.dt
                * ida_price_vector_05[t]
                * (
                    p_discharge_ida[t]
                    + p_discharge_ida_close[t]
                    - p_charge_ida[t]
                    - p_charge_ida_close[t]
                )
                for t in range(TIME_STEPS)
            )
        )

        # Add risk-aware term if applicable
        if "risk-aware" in self.objective.lower():
            objective_terms.append(
                gp.quicksum(
                    self.dt
                    * (
                        -self.risk_factor
                        * (ida_price_vector_09[t] - ida_price_vector_05[t])
                        * p_charge_ida[t]
                        - self.risk_factor
                        * (ida_price_vector_05[t] - ida_price_vector_01[t])
                        * p_discharge_ida[t]
                    )
                    for t in range(TIME_STEPS)
                )
            )

        # Add more terms based on the selected objective strategy (e.g., smooth operation, penalty terms)
        if "smooth" in self.objective.lower():
            objective_terms.append(
                gp.quicksum(
                    -0.01 * (p_discharge_ida[t] - p_discharge_ida[t - 1]) ** 2
                    - 0.01 * (p_charge_ida[t] - p_charge_ida[t - 1]) ** 2
                    for t in range(1, TIME_STEPS)  # Start at t=1 to avoid index error
                )
            )

        if "piece-wise" in self.objective.lower():
            # Auxiliary variables for piecewise-linear transformation
            z_charge = {
                t: model_ida.addVar(lb=0, vtype=gp.GRB.CONTINUOUS, name=f"z_charge_{t}")
                for t in range(TIME_STEPS)
            }
            z_discharge = {
                t: model_ida.addVar(
                    lb=0, vtype=gp.GRB.CONTINUOUS, name=f"z_discharge_{t}"
                )
                for t in range(TIME_STEPS)
            }

            for t in range(TIME_STEPS):
                # PWL function for discharging
                model_ida.addGenConstrPWL(
                    p_discharge_ida[t],
                    z_discharge[t],
                    [
                        0,
                        0.2 * self.power_capacity,
                        0.4 * self.power_capacity,
                        0.6 * self.power_capacity,
                        0.8 * self.power_capacity,
                        1 * self.power_capacity,
                    ],
                    [
                        0,
                        self.risk_factor
                        * self.dt
                        * (ida_price_vector_05[t] - ida_price_vector_04[t]),
                        self.risk_factor
                        * self.dt
                        * (ida_price_vector_05[t] - ida_price_vector_03[t]),
                        self.risk_factor
                        * self.dt
                        * (ida_price_vector_05[t] - ida_price_vector_02[t]),
                        self.risk_factor
                        * self.dt
                        * (ida_price_vector_05[t] - ida_price_vector_01[t]),
                        self.risk_factor * self.dt * (ida_price_vector_05[t]),
                    ],
                    name=f"PWL_Discharge_{t}",
                )

                # PWL function for charging
                model_ida.addGenConstrPWL(
                    p_charge_ida[t],
                    z_charge[t],
                    [
                        0,
                        0.2 * self.power_capacity,
                        0.4 * self.power_capacity,
                        0.6 * self.power_capacity,
                        0.8 * self.power_capacity,
                        1 * self.power_capacity,
                    ],
                    [
                        0,
                        self.risk_factor
                        * self.dt
                        * (ida_price_vector_06[t] - ida_price_vector_05[t]),
                        self.risk_factor
                        * self.dt
                        * (ida_price_vector_07[t] - ida_price_vector_05[t]),
                        self.risk_factor
                        * self.dt
                        * (ida_price_vector_08[t] - ida_price_vector_05[t]),
                        self.risk_factor
                        * self.dt
                        * (ida_price_vector_09[t] - ida_price_vector_05[t]),
                        self.risk_factor * self.dt * (ida_price_vector_05[t]),
                    ],
                    name=f"PWL_Charge_{t}",
                )

            objective_terms.append(
                gp.quicksum(-(z_discharge[t] + z_charge[t]) for t in range(TIME_STEPS))
            )

        if self.objective == "perfect-foresight":
            model_ida.setObjective(
                gp.quicksum(
                    ida_price_vector_true[t]
                    * (
                        p_discharge_ida[t]
                        + p_discharge_ida_close[t]
                        - p_charge_ida[t]
                        - p_charge_ida_close[t]
                    )
                    for t in range(TIME_STEPS)
                ),
                sense=gp.GRB.MAXIMIZE,
            )
        else:
            # Combine all selected terms
            model_ida.setObjective(
                gp.quicksum(objective_terms),  # Summing all terms dynamically
                sense=gp.GRB.MAXIMIZE,
            )

        model_ida.optimize()

        self.results_ida = {
            "p_charge_ida": [p_charge_ida[t].x for t in range(TIME_STEPS)],
            "p_discharge_ida": [p_discharge_ida[t].x for t in range(TIME_STEPS)],
            "p_charge_ida_close": [p_charge_ida_close[t].x for t in range(TIME_STEPS)],
            "p_discharge_ida_close": [
                p_discharge_ida_close[t].x for t in range(TIME_STEPS)
            ],
            "p_batt_ida": [p_batt_ida[t].x for t in range(TIME_STEPS)],
            "soc_ida": [soc_ida[t].x for t in range(TIME_STEPS)],
            "x_bin_charge_ida": [x_bin_charge_ida[t].x for t in range(TIME_STEPS)],
            "x_bin_discharge_ida": [
                x_bin_discharge_ida[t].x for t in range(TIME_STEPS)
            ],
            "p_charge_daa_ida": [
                results_daa["p_charge_daa"][t]
                - p_discharge_ida_close[t].x
                + p_charge_ida[t].x
                for t in range(TIME_STEPS)
            ],
            "p_discharge_daa_ida": [
                results_daa["p_discharge_daa"][t]
                - p_charge_ida_close[t].x
                + p_discharge_ida[t].x
                for t in range(TIME_STEPS)
            ],
        }

        return self.results_ida

    def optimizeIDC(self, quantile_forecasts, results_ida, idc_price_vector_true):

        # Ensure forecasts are provided for all required quantiles
        if len(quantile_forecasts) != 9:
            raise ValueError("Expected 9 quantile forecasts (0.1 to 0.9).")

        # Unpack the quantile forecasts
        (
            idc_price_vector_01,
            idc_price_vector_02,
            idc_price_vector_03,
            idc_price_vector_04,
            idc_price_vector_05,
            idc_price_vector_06,
            idc_price_vector_07,
            idc_price_vector_08,
            idc_price_vector_09,
        ) = quantile_forecasts

        model_idc = gp.Model("IDC")

        # Set variables
        soc_idc = {
            t: model_idc.addVar(
                lb=0, ub=1, vtype=gp.GRB.CONTINUOUS, name=f"soc_idc_{t}"
            )
            for t in range(TIME_STEPS)
        }
        p_charge_idc = {
            t: model_idc.addVar(
                lb=0.1,
                ub=self.power_capacity,
                vtype=gp.GRB.SEMICONT,
                name=f"p_charge_idc_{t}",
            )
            for t in range(TIME_STEPS)
        }
        p_discharge_idc = {
            t: model_idc.addVar(
                lb=0.1,
                ub=self.power_capacity,
                vtype=gp.GRB.SEMICONT,
                name=f"p_charge_idc_{t}",
            )
            for t in range(TIME_STEPS)
        }
        p_charge_idc_close = {
            t: model_idc.addVar(
                lb=0.1,
                ub=self.power_capacity,
                vtype=gp.GRB.SEMICONT,
                name=f"p_charge_idc_close_{t}",
            )
            for t in range(TIME_STEPS)
        }
        p_discharge_idc_close = {
            t: model_idc.addVar(
                lb=0.1,
                ub=self.power_capacity,
                vtype=gp.GRB.SEMICONT,
                name=f"p_discharge_idc_close_{t}",
            )
            for t in range(TIME_STEPS)
        }

        p_batt_idc = {
            t: model_idc.addVar(
                lb=-self.power_capacity,
                ub=self.power_capacity,
                vtype=gp.GRB.CONTINUOUS,
                name=f"p_batt_idc_{t}",
            )
            for t in range(TIME_STEPS)
        }

        x_bin_charge_idc = {
            t: model_idc.addVar(vtype=gp.GRB.BINARY, name=f"x_bin_charge_idc_{t}")
            for t in range(TIME_STEPS)
        }
        x_bin_discharge_idc = {
            t: model_idc.addVar(vtype=gp.GRB.BINARY, name=f"x_bin_discharge_idc_{t}")
            for t in range(TIME_STEPS)
        }

        # Set constraints
        # Correcting SOC update equation
        model_idc.addConstrs(
            (
                soc_idc[t]
                == soc_idc[t - 1]
                + self.dt
                * (
                    (p_charge_idc[t] * self.efficiency)
                    + p_charge_idc_close[t]
                    + results_ida["p_charge_daa_ida"][t] * self.efficiency
                    - (p_discharge_idc[t] / self.efficiency)
                    - p_discharge_idc_close[t]
                    - results_ida["p_discharge_daa_ida"][t] / self.efficiency
                )
                / self.energy_capacity
                for t in range(1, TIME_STEPS)
            ),
            name="soc_update",
        )
        model_idc.addConstr((soc_idc[0] == 0.5), name="initial_soc")
        model_idc.addConstr((soc_idc[TIME_STEPS - 1] == 0.5), name="final_soc")

        model_idc.addConstrs(
            (
                p_charge_idc[t] <= self.power_capacity * x_bin_charge_idc[t]
                for t in range(TIME_STEPS)
            ),
            name="charge_limit",
        )
        model_idc.addConstrs(
            (
                p_discharge_idc[t] <= self.power_capacity * x_bin_discharge_idc[t]
                for t in range(TIME_STEPS)
            ),
            name="discharge_limit",
        )
        model_idc.addConstrs(
            x_bin_charge_idc[t] + x_bin_discharge_idc[t] <= 1 for t in range(TIME_STEPS)
        )

        model_idc.addConstrs(
            (
                p_batt_idc[t] == p_charge_idc[t] - p_discharge_idc[t]
                for t in range(TIME_STEPS)
            ),
            name="batt_power",
        )

        model_idc.addConstrs(
            (
                p_charge_idc[t] + p_discharge_idc[t] <= self.power_capacity
                for t in range(TIME_STEPS)
            ),
            name="power_limit",
        )

        if self.cyclic_constraint:
            WINDOW_SIZE = int(TIME_STEPS / 6)  # 6 hours

            model_idc.addConstrs(
                gp.quicksum(
                    p_charge_idc[t + k]
                    + results_ida["p_charge_daa_ida"][t + k]
                    - p_discharge_idc_close[t + k]
                    for k in range(WINDOW_SIZE)
                    if t + k < TIME_STEPS
                )
                <= (self.volume_limit / self.dt) / 6
                for t in range(0, TIME_STEPS - WINDOW_SIZE, 12)  # Apply every 3 hours
            )

            model_idc.addConstrs(
                gp.quicksum(
                    p_discharge_idc[t + k]
                    + results_ida["p_discharge_daa_ida"][t + k]
                    - p_charge_idc_close[t + k]
                    for k in range(WINDOW_SIZE)
                    if t + k < TIME_STEPS
                )
                <= (self.volume_limit / self.dt) / 6
                for t in range(0, TIME_STEPS - WINDOW_SIZE, 12)
            )

            model_idc.addConstr(
                (
                    gp.quicksum(
                        (
                            p_charge_idc[t]
                            + results_ida["p_charge_daa_ida"][t]
                            - p_discharge_idc_close[t]
                        )
                        for t in range(TIME_STEPS)
                    )
                    <= self.volume_limit / self.dt
                ),
                name="volume_limit",
            )
            model_idc.addConstr(
                (
                    gp.quicksum(
                        (
                            p_discharge_idc[t]
                            + results_ida["p_discharge_daa_ida"][t]
                            - p_charge_idc_close[t]
                        )
                        for t in range(TIME_STEPS)
                    )
                    <= self.volume_limit / self.dt
                ),
                name="volume_limit",
            )

        model_idc.addConstrs(
            (
                p_charge_idc_close[t] <= results_ida["p_discharge_daa_ida"][t]
                for t in range(TIME_STEPS)
            ),
            name="close_charge",
        )
        model_idc.addConstrs(
            (
                p_discharge_idc_close[t] <= results_ida["p_charge_daa_ida"][t]
                for t in range(TIME_STEPS)
            ),
            name="close_discharge",
        )

        model_idc.addConstrs(
            (
                p_charge_idc[t] + results_ida["p_charge_daa_ida"][t]
                <= self.power_capacity
                for t in range(TIME_STEPS)
            ),
            name="close_charge_limit",
        )
        model_idc.addConstrs(
            (
                p_discharge_idc[t] + results_ida["p_discharge_daa_ida"][t]
                <= self.power_capacity
                for t in range(TIME_STEPS)
            ),
            name="close_discharge_limit",
        )

        if "Adaptive" in self.optimization_method:
            # Estimate potential profit from arbitrage (difference between min and max quantile forecasts)
            profit_opportunity = np.max(idc_price_vector_05) - np.min(
                idc_price_vector_05
            )

            # Normalize using a logistic function to keep risk factor in range [0,1]
            self.risk_factor = 1 / (
                1 + np.exp(-profit_opportunity / np.mean(idc_price_vector_05))
            )

        # print("Risk factor in IDC-Market: ", self.risk_factor)

        # Initialize objective as a list of terms
        objective_terms = []

        # Basic revenue maximization (Simple objective)
        objective_terms.append(
            gp.quicksum(
                self.dt
                * idc_price_vector_05[t]
                * (
                    p_discharge_idc[t]
                    + p_discharge_idc_close[t]
                    - p_charge_idc[t]
                    - p_charge_idc_close[t]
                )
                for t in range(TIME_STEPS)
            )
        )

        # Add risk-aware term if applicable
        if "risk-aware" in self.objective.lower():
            objective_terms.append(
                gp.quicksum(
                    self.dt
                    * (
                        -self.risk_factor
                        * (idc_price_vector_09[t] - idc_price_vector_05[t])
                        * p_charge_idc[t]
                        - self.risk_factor
                        * (idc_price_vector_05[t] - idc_price_vector_01[t])
                        * p_discharge_idc[t]
                    )
                    for t in range(TIME_STEPS)
                )
            )

        # Add more terms based on the selected objective strategy (e.g., smooth operation, penalty terms)
        if "smooth" in self.objective.lower():
            objective_terms.append(
                gp.quicksum(
                    -0.01 * (p_discharge_idc[t] - p_discharge_idc[t - 1]) ** 2
                    - 0.01 * (p_charge_idc[t] - p_charge_idc[t - 1]) ** 2
                    for t in range(1, TIME_STEPS)  # Start at t=1 to avoid index error
                )
            )

        if "piece-wise" in self.objective.lower():
            # Auxiliary variables for piecewise-linear transformation
            z_charge = {
                t: model_idc.addVar(lb=0, vtype=gp.GRB.CONTINUOUS, name=f"z_charge_{t}")
                for t in range(TIME_STEPS)
            }
            z_discharge = {
                t: model_idc.addVar(
                    lb=0, vtype=gp.GRB.CONTINUOUS, name=f"z_discharge_{t}"
                )
                for t in range(TIME_STEPS)
            }

            for t in range(TIME_STEPS):
                # PWL function for discharging
                model_idc.addGenConstrPWL(
                    p_discharge_idc[t],
                    z_discharge[t],
                    [
                        0,
                        0.2 * self.power_capacity,
                        0.4 * self.power_capacity,
                        0.6 * self.power_capacity,
                        0.8 * self.power_capacity,
                        1 * self.power_capacity,
                    ],
                    [
                        0,
                        self.risk_factor
                        * self.dt
                        * (idc_price_vector_05[t] - idc_price_vector_04[t]),
                        self.risk_factor
                        * self.dt
                        * (idc_price_vector_05[t] - idc_price_vector_03[t]),
                        self.risk_factor
                        * self.dt
                        * (idc_price_vector_05[t] - idc_price_vector_02[t]),
                        self.risk_factor
                        * self.dt
                        * (idc_price_vector_05[t] - idc_price_vector_01[t]),
                        self.risk_factor * self.dt * (idc_price_vector_05[t]),
                    ],
                    name=f"PWL_Discharge_{t}",
                )

                # PWL function for charging
                model_idc.addGenConstrPWL(
                    p_charge_idc[t],
                    z_charge[t],
                    [
                        0,
                        0.2 * self.power_capacity,
                        0.4 * self.power_capacity,
                        0.6 * self.power_capacity,
                        0.8 * self.power_capacity,
                        1 * self.power_capacity,
                    ],
                    [
                        0,
                        self.risk_factor
                        * self.dt
                        * (idc_price_vector_06[t] - idc_price_vector_05[t]),
                        self.risk_factor
                        * self.dt
                        * (idc_price_vector_07[t] - idc_price_vector_05[t]),
                        self.risk_factor
                        * self.dt
                        * (idc_price_vector_08[t] - idc_price_vector_05[t]),
                        self.risk_factor
                        * self.dt
                        * (idc_price_vector_09[t] - idc_price_vector_05[t]),
                        self.risk_factor * self.dt * (idc_price_vector_05[t]),
                    ],
                    name=f"PWL_Charge_{t}",
                )

            objective_terms.append(
                gp.quicksum(-(z_discharge[t] + z_charge[t]) for t in range(TIME_STEPS))
            )

        if self.objective == "perfect-foresight":
            model_idc.setObjective(
                gp.quicksum(
                    idc_price_vector_true[t]
                    * (
                        p_discharge_idc[t]
                        + p_discharge_idc_close[t]
                        - p_charge_idc[t]
                        - p_charge_idc_close[t]
                    )
                    for t in range(TIME_STEPS)
                ),
                sense=gp.GRB.MAXIMIZE,
            )
        else:
            # Combine all selected terms
            model_idc.setObjective(
                gp.quicksum(objective_terms),  # Summing all terms dynamically
                sense=gp.GRB.MAXIMIZE,
            )

        model_idc.optimize()

        self.results_idc = {
            "p_charge_idc": [p_charge_idc[t].x for t in range(TIME_STEPS)],
            "p_discharge_idc": [p_discharge_idc[t].x for t in range(TIME_STEPS)],
            "p_charge_idc_close": [p_charge_idc_close[t].x for t in range(TIME_STEPS)],
            "p_discharge_idc_close": [
                p_discharge_idc_close[t].x for t in range(TIME_STEPS)
            ],
            "p_batt_idc": [p_batt_idc[t].x for t in range(TIME_STEPS)],
            "soc_idc": [soc_idc[t].x for t in range(TIME_STEPS)],
            "x_bin_charge_idc": [x_bin_charge_idc[t].x for t in range(TIME_STEPS)],
            "x_bin_discharge_idc": [
                x_bin_discharge_idc[t].x for t in range(TIME_STEPS)
            ],
            "p_charge_daa_ida_idc": [
                results_ida["p_charge_daa_ida"][t]
                - p_discharge_idc_close[t].x
                + p_charge_idc[t].x
                for t in range(TIME_STEPS)
            ],
            "p_discharge_daa_ida_idc": [
                results_ida["p_discharge_daa_ida"][t]
                - p_charge_idc_close[t].x
                + p_discharge_idc[t].x
                for t in range(TIME_STEPS)
            ],
        }

        return self.results_idc

    def algorithmic_idc(self, results_ida):
        pass


if __name__ == "__main__":

    gp.setParam("OutputFlag", 1)

    sea = SequentialEnergyArbitrage()
    # Create 9 x 96 dummy forecasts with random values
    dummy_forecasts = np.random.randn(9, 96) * 120
    results_daa = sea.optimizeDAA(dummy_forecasts)
    dummy_forecasts = np.random.randn(9, 96) * 240
    results_ida = sea.optimizeIDA(dummy_forecasts, results_daa)
    dummy_forecasts = np.random.randn(9, 96) * 360
    results_idc = sea.optimizeIDC(dummy_forecasts, results_ida)

    # Transform results into a pandas DataFrame and save as xlsx
    import pandas as pd

    results_df = pd.DataFrame(results_idc)
    results_df.to_excel("results.xlsx", index=False)

    # plot profile
    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots()

    # Plot charging and discharging power as bar charts
    ax1.bar(
        range(TIME_STEPS),
        results_idc["p_charge_daa_ida_idc"],
        label="Charge Power",
        color="tab:blue",
        alpha=0.6,
    )
    ax1.bar(
        range(TIME_STEPS),
        results_idc["p_discharge_daa_ida_idc"],
        label="Discharge Power",
        color="tab:orange",
        alpha=0.6,
    )
    ax1.set_xlabel("Time Steps")
    ax1.set_ylabel("Power ($MW$)")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(results_idc["soc_idc"], label="State of Charge (SOC)", color="tab:green")
    ax2.set_ylabel("State of Charge (SOC)")
    ax2.legend(loc="upper right")
    ax2.set_ylim(-0.1, 1.1)

    # Plot binary charging and discharging actions
    fig, ax3 = plt.subplots()
    ax3.bar(
        range(TIME_STEPS),
        results_idc["x_bin_charge_idc"],
        label="Binary Charge Action",
        color="tab:blue",
    )
    ax3.bar(
        range(TIME_STEPS),
        results_idc["x_bin_discharge_idc"],
        label="Binary Discharge Action",
        color="tab:orange",
    )
    ax3.set_xlabel("Time Steps")
    ax3.set_ylabel("Binary Action")
    ax3.legend(loc="upper left")
    ax3.set_ylim(-0.1, 1.1)

    plt.show()
