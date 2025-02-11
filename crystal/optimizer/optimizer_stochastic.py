import os
import numpy as np
import math
import pyomo.environ as pyo
from pyomo.environ import *

import pyomo.opt as po

#stochastische Optimierung: angepasste Zielfunktionen (Optimierung über den Erwartungswert)


class energy_arbitrage_stochastic_optimizer:
    def __init__(self, name, risk_factor):
        self.name: str = name
        self.risk_factor = risk_factor

    def optimize_daa(self, n_cycles: int, energy_cap: int, power_cap: int, quantile_forecasts):
        """
        Calculates the optimal charge/discharge schedule on the day-ahead auction (DAA)
        using dynamic quantile forecasts.

        Parameters:
        - n_cycles: Maximum number of allowed cycles
        - energy_cap: Energy capacity
        - power_cap: Power capacity
        - quantile_forecasts: Variable-length list of quantile forecasts (0.1 to 0.9)

        Returns:
        - step1_soc_daa: Resulting state of charge schedule
        - step1_cha_daa: Resulting charge schedule / Positions on DA Auction
        - step1_dis_daa: Resulting discharge schedule / Positions on DA Auction
        - step1_profit_daa: Profit from Day-ahead auction trades
        """

        # Ensure forecasts are provided for all required quantiles
        if len(quantile_forecasts) != 9:
            raise ValueError("Expected 9 quantile forecasts (0.1 to 0.9).")

        # Unpack the quantile forecasts
        (daa_price_vector_01, daa_price_vector_02, daa_price_vector_03,
         daa_price_vector_04, daa_price_vector_05, daa_price_vector_06,
         daa_price_vector_07, daa_price_vector_08, daa_price_vector_09) = quantile_forecasts

        # Initialize pyomo model:
        model = pyo.ConcreteModel()

        # Set parameters:

        # Number of hours, Liste beginnt bei 0
        model.H = pyo.RangeSet(0, len(daa_price_vector_01)/4-1)
        # Number of quarters
        model.Q = pyo.RangeSet(1, len(daa_price_vector_01))
        # Number of quarters plus 1
        model.Q_plus_1 = pyo.RangeSet(1, len(daa_price_vector_01)+1)
        # Daily discharged energy limit
        volume_limit = energy_cap * n_cycles

        # Initialize variables:

        # State of charge
        model.soc = pyo.Var(model.Q_plus_1, domain=pyo.Reals)
        # Charges on the Day-ahead auction
        model.cha_daa = pyo.Var(model.Q, domain=pyo.NonNegativeReals, bounds=(0, 1))
        # Discharges on the Day-ahead auction
        model.dis_daa = pyo.Var(model.Q, domain=pyo.NonNegativeReals, bounds=(0, 1))

        # Define Constraints:
        # Remark: In some of the constraints, you will notice that the indices [q] and [q-1] are used for the same quarter. This is due to Python lists counting from 0 and Pyomo Variable lists counting from 1.

        @model.Constraint(model.Q_plus_1)
        def set_maximum_soc(model, q):
            """ State of charge can never be higher than Energy Capacity. (Constraint 1.1) """
            return model.soc[q] <= energy_cap

        @model.Constraint(model.Q_plus_1)
        def set_minimum_soc(model, q):
            """State of charge can never be less than 0. (Constraint 1.2)"""
            return model.soc[q] >= 0

        @model.Constraint()
        def set_first_and_last_soc_to_0(model):
            """ State of charge at the first quarter must be 0. (Constraint 1.3) """
            return model.soc[1] == 0.5

        @model.Constraint()
        def set_last_soc_to_0(model):
            """ State of charge at quarter 97 (i.e., first quarter of next day) must be 0. (Constraint 1.4) """
            return model.soc[97] == 0.5

        @model.Constraint(model.Q)
        def soc_step_constraint(model, q):
            """ The state of charge of each quarter equals the state if charge of the previous quarter plus charges minus discharges. (Constraint 1.5) """
            return model.soc[q + 1] == model.soc[q] + power_cap / 4 * model.cha_daa[q] - power_cap / 4 * model.dis_daa[q]

        @model.Constraint()
        def charge_cycle_limit(model):
            """ Sum of all charges has to be below the daily limit. (Constraint 1.6) """
            #Die Viertelstunden werden über die 24h eines Tages iteriert und damit mit dem zugehörigen täglichen max. Wert verglichen, dass fortführend
            return sum(model.cha_daa[q] * power_cap / 4 for q in model.Q) <= volume_limit

        @model.Constraint()
        def discharge_cycle_limit(model):
            """ Sum of all discharges has to be below the daily limit. (Constraint 1.7) """
            return sum(model.dis_daa[q] * power_cap / 4 for q in model.Q) <= volume_limit

        @model.Constraint(model.H)
        def cha_daa_quarters_1_2_parity(model, q):
            """ Set daa positions of quarter 1 and 2 of each hour equal. (Constraint 1.8)
            On the DA Auction, positions in all 4 quarters of the hour have to be identical since trades are taken in hourly blocks. """
            return model.cha_daa[4 * q + 1] == model.cha_daa[4 * q + 2]

        @model.Constraint(model.H)
        def cha_daa_quarters_2_3_parity(model, q):
            """ Set daa positions of quarter 2 and 3 of each hour equal. (Constraint 1.8)"""
            return model.cha_daa[4 * q + 2] == model.cha_daa[4 * q + 3]

        @model.Constraint(model.H)
        def cha_daa_quarters_3_4_parity(model, q):
            """ Set daa positions of quarter 3 and 4 of each hour equal. (Constraint 1.8) """
            return model.cha_daa[4 * q + 3] == model.cha_daa[4 * q + 4]

        @model.Constraint(model.H)
        def dis_daa_quarters_1_2_parity(model, q):
            """ Set daa positions of quarter 1 and 2 of each hour equal. (Constraint 1.9)
            On the DA Auction, positions in all 4 quarters of the hour have to be identical since trades are taken in hourly blocks. """
            return model.dis_daa[4 * q + 1] == model.dis_daa[4 * q + 2]

        @model.Constraint(model.H)
        def dis_daa_quarters_2_3_parity(model, q):
            """ Set daa positions of quarter 2 and 3 of each hour equal. (Constraint 1.9) """
            return model.dis_daa[4 * q + 2] == model.dis_daa[4 * q + 3]

        @model.Constraint(model.H)
        def dis_daa_quarters_3_4_parity(model, q):
            """ Set daa positions of quarter 3 and 4 of each hour equal. (Constraint 1.9) """
            return model.dis_daa[4 * q + 3] == model.dis_daa[4 * q + 4]

        # Define objective function and solve the optimization problem.
        # The objective is to maximize revenue from DA Auction trades over all possible charge-discharge schedules.

        model.obj = pyo.Objective(expr=pyo.quicksum(power_cap / 4 * (
                                                                    daa_price_vector_05[q-1] * (model.dis_daa[q] - model.cha_daa[q]) -
                                                                    self.risk_factor * (daa_price_vector_05[q-1] - daa_price_vector_01[q-1]) * model.cha_daa[q] -
                                                                    self.risk_factor * (daa_price_vector_09[q-1] - daa_price_vector_05[q-1]) * model.dis_daa[q]
                                                                ) for q in model.Q), sense=pyo.maximize)

        solver = pyo.SolverFactory("gurobi")
        solver.solve(model, timelimit=200)

        # Extract optimal schedules
        step1_soc_daa = [model.soc[q].value for q in model.Q]
        step1_cha_daa = [model.cha_daa[q].value for q in model.Q]
        step1_dis_daa = [model.dis_daa[q].value for q in model.Q]

        # # Calculate profit
        # step1_profit_daa = sum(
        #     power_cap / 4 * (
        #             daa_price_vector_05[q - 1] * (model.dis_daa[q] - model.cha_daa[q]) -
        #             self.risk_factor * (daa_price_vector_05[q - 1] - daa_price_vector_01[q - 1]) * model.cha_daa[q] -
        #             self.risk_factor * (daa_price_vector_09[q - 1] - daa_price_vector_05[q - 1]) * model.dis_daa[q]
        #     ) for q in range(len(daa_price_vector_01))
        # )

        step1_profit_daa = 0

        return step1_soc_daa, step1_cha_daa, step1_dis_daa, step1_profit_daa

    def optimize_ida(self, n_cycles: int, energy_cap: int, power_cap: int, step1_cha_daa: list, step1_dis_daa: list, quantile_forecasts):
        """
        Calculates optimal charge/discharge schedule on the intraday auction (ida) for a given 96-d ida_price_vector.

        Parameters:
        - n_cycles: Maximum number of allowed cycles
        - energy_cap: Energy capacity
        - power_cap: Power capacity
        - ida_price_vector: 96-dimensional ida price vector
        - step1_cha_daa: Previous Buys on the Day-Ahead auction
        - step1_dis_daa: Previous Sells on the Day-Ahead auction

        Returns:
        - step2_soc_ida: Resulting state of charge schedule
        - step2_cha_ida: Resulting charges on ID Auction
        - step2_dis_ida: Resulting discharges on ID Auction
        - step2_cha_ida_close: Resulting charges on ID Auction to close previous DA Auction positions
        - step2_dis_ida_close: Resulting discharge on ID Auction to close previous DA Auction positions
        - step2_profit_ida: Profit from Day-ahead auction trades
        - step2_cha_daaida: Combined charges from DA Auction and ID Auction
        - step2_dis_daaida: Combined discharges from DA Auction and ID Auction
        """

        # Ensure forecasts are provided for all required quantiles
        if len(quantile_forecasts) != 9:
            raise ValueError("Expected 9 quantile forecasts (0.1 to 0.9).")

        # Unpack the quantile forecasts
        (ida_price_vector_01, ida_price_vector_02, ida_price_vector_03,
         ida_price_vector_04, ida_price_vector_05, ida_price_vector_06,
         ida_price_vector_07, ida_price_vector_08, ida_price_vector_09) = quantile_forecasts

        # Initialize pyomo model:
        model = pyo.ConcreteModel()

        # Set parameters:

        # Number of hours
        model.H = pyo.RangeSet(0, len(ida_price_vector_01)/4-1)
        # Number of quarters
        model.Q = pyo.RangeSet(1, len(ida_price_vector_01))
        # Number of quarters plus 1
        model.Q_plus_1 = pyo.RangeSet(1, len(ida_price_vector_01)+1)
        # Daily discharged energy limit
        volume_limit = energy_cap * n_cycles

        # Initialize variables:

        # State of charge
        model.soc = pyo.Var(model.Q_plus_1, domain=pyo.Reals)
        # Charges on the intraday auction
        model.cha_ida = pyo.Var(model.Q, domain=pyo.NonNegativeReals, bounds=(0, 1))
        # Discharges on the intraday auction
        model.dis_ida = pyo.Var(model.Q, domain=pyo.NonNegativeReals, bounds=(0, 1))
        # Charges on the intraday auction to close previous positions from the day-ahead auction
        model.cha_ida_close = pyo.Var(model.Q, domain=pyo.NonNegativeReals, bounds=(0, 1))
        # Charges on the intraday auction to close previous positions from the day-ahead auction
        model.dis_ida_close = pyo.Var(model.Q, domain=pyo.NonNegativeReals, bounds=(0, 1))

        # Define Constraints:

        @model.Constraint(model.Q_plus_1)
        def set_maximum_soc(model, q):
            """ State of charge can never be higher than Energy Capacity. (Constraint 2.1) """
            return model.soc[q] <= energy_cap

        @model.Constraint(model.Q_plus_1)
        def set_minimum_soc(model, q):
            """ State of charge can never be less than 0. (Constraint 2.2) """
            return model.soc[q] >= 0

        @model.Constraint()
        def set_first_soc_to_0(model):
            """ State of charge at the first quarter must be 0. (Constraint 2.3) """
            return model.soc[1] == 0.5

        @model.Constraint()
        def set_last_soc_to_0(model):
            """ State of charge at quarter 97 (i.e., first quarter of next day) must be 0. (Constraint 2.4) """
            return model.soc[97] == 0.5

        @model.Constraint(model.Q)
        def soc_step_constraint(model, q):
            """ The state of charge of each quarter equals the state if charge of the previous quarter plus charges minus discharges. (Constraint 2.5) """
            return model.soc[q+1] == model.soc[q] + power_cap/4 * (model.cha_ida[q] - model.dis_ida[q] + model.cha_ida_close[q] - model.dis_ida_close[q] + step1_cha_daa[q-1] - step1_dis_daa[q-1])

        @model.Constraint()
        def charge_cycle_limit(model):
            """ Sum of all charges has to be below the daily limit. (Constraint 2.6)"""
            # - Entladen auf IDA für DAA, da das die zugehörige DAA-Position schließt und somit das Laden nie stattgefunden hat
            return ((np.sum(step1_cha_daa) + sum(model.cha_ida[q] for q in model.Q) - sum(model.dis_ida_close[q] for q in model.Q)) * power_cap/4 <= volume_limit)

        @model.Constraint()
        def discharge_cycle_limit(model):
            """ Sum of all discharges has to be below the daily limit. (Constraint 2.7) """
            return ((np.sum(step1_dis_daa) + sum(model.dis_ida[q] for q in model.Q) - sum(model.cha_ida_close[q] for q in model.Q)) * power_cap/4 <= volume_limit)

        @model.Constraint(model.Q)
        def cha_close_logic(model, q):
            """ cha_ida_close can only close or reduce existing dis_daa positions. They can only be placed, where dis_daa positions exist. (Constraint 2.8) """
            return model.cha_ida_close[q] <= step1_dis_daa[q-1]

        @model.Constraint(model.Q)
        def dis_close_logic(model, q):
            """ dis_ida_close can only close or reduce existing cha_daa positions. They can only be placed, where cha_daa positions exist. (Constraint 2.9) """
            return model.dis_ida_close[q] <= step1_cha_daa[q-1]

        @model.Constraint(model.Q)
        def charge_rate_limit(model, q):
            """ Sum of cha_ida[q] and cha_daa[q] has to be less or equal to 1. (Constraint 2.10) """
            # in Summe kann in einer Viertelstunde nicht mehr in den Speicher geladen werden, als über die max. über Laderate(=1) möglich wäre
            return model.cha_ida[q] + step1_cha_daa[q-1] <= 1

        @model.Constraint(model.Q)
        def discharge_rate_limit(model, q):
            """ Sum of dis_ida[q] and dis_daa[q] has to be less or equal to 1. (Constraint 2.11) """
            return model.dis_ida[q] + step1_dis_daa[q-1] <= 1

        # Define objective function and solve the optimization problem
        # The objective is to maximize revenue from ID Auction trades over all possible charge-discharge schedules.
        model.obj = pyo.Objective(expr=sum(power_cap / 4 * (
                (ida_price_vector_05[q - 1]) *
                (model.dis_ida[q] + model.dis_ida_close[q] - model.cha_ida[q] - model.cha_ida_close[q]) -
                self.risk_factor * (ida_price_vector_05[q - 1] - ida_price_vector_01[q - 1]) * model.cha_ida[q] -
                self.risk_factor * (ida_price_vector_09[q - 1] - ida_price_vector_05[q - 1]) * model.dis_ida[q])
                                           for q in model.Q),
                                  sense=pyo.maximize)

        solver = pyo.SolverFactory("gurobi")
        solver.solve(model, timelimit=200)

        # Retrieve arrays of resulting optimal soc/charge/discharge schedules after the ID Auction:
        step2_soc_ida = [model.soc[q].value for q in range(1, len(ida_price_vector_01) + 1)]
        step2_cha_ida = [model.cha_ida[q].value for q in range(1, len(ida_price_vector_01) + 1)]
        step2_dis_ida = [model.dis_ida[q].value for q in range(1, len(ida_price_vector_01) + 1)]
        step2_cha_ida_close = [model.cha_ida_close[q].value for q in range(1, len(ida_price_vector_01) + 1)]
        step2_dis_ida_close = [model.dis_ida_close[q].value for q in range(1, len(ida_price_vector_01) + 1)]

        # step2_profit_ida = np.sum(((np.asarray(step2_dis_ida) + step2_dis_ida_close) - (
        #     np.asarray(step2_cha_ida) + step2_cha_ida_close)) * (0.05 * ida_price_vector_01 + 0.05 * ida_price_vector_02 + 0.1 * ida_price_vector_03 + 0.15 * ida_price_vector_04 + 0.2 * ida_price_vector_05 + 0.15 * ida_price_vector_06 + 0.1 * ida_price_vector_07 + 0.1 * ida_price_vector_08 + 0.1 * ida_price_vector_09)) * power_cap/4

        step2_profit_ida = 0

        # Calculate total physical charge discharge schedules of combined day-ahead and intraday auction trades:
        step2_cha_daaida = np.asarray(
            step1_cha_daa) - step2_dis_ida_close + step2_cha_ida

        assert type(step1_cha_daa) == type(step2_dis_ida_close)
        step2_dis_daaida = np.asarray(
            step1_dis_daa) - step2_cha_ida_close + step2_dis_ida

        return (step2_soc_ida, step2_cha_ida, step2_dis_ida, step2_cha_ida_close, step2_dis_ida_close, step2_profit_ida, step2_cha_daaida, step2_dis_daaida)

    def optimize_idc(self, n_cycles: int, energy_cap: int, power_cap: int, step2_cha_daaida: list, step2_dis_daaida: list, quantile_forecasts):
        """
        Calculates optimal charge/discharge schedule on the intraday continuous (idc) for a given 96-d idc_price_vector.

        Parameters:
        - n_cycles: Maximum number of allowed cycles
        - energy_cap: Energy capacity
        - power_cap: Power capacity
        - ida_price_vector: 96-dimensional ida price vector
        - step2_cha_daaida: Previous combined Buys on the DA Auction and ID Auction
        - step2_dis_daaida: Previous combined Sells on the DA Auction and ID Auction

        Returns:
        - step3_soc_idc: Resulting state of charge schedule
        - step3_cha_idc: Resulting charges on ID Continuous
        - step3_dis_idc: Resulting discharges on ID Continuous
        - step3_cha_idc_close: Resulting charges on ID Continuous to close previous DA or ID Auction positions
        - step3_dis_idc_close: Resulting discharge on ID Continuous to close previous DA or ID Auction positions
        - step3_profit_idc: Profit from Day-ahead auction trades
        - step3_cha_daaidaidc: Combined charges from DA Auction, ID Auction and ID Continuous
        - step3_dis_daaidaidc: Combined discharges from DA Auction, ID Auction and ID Continuous
        """

        # Ensure forecasts are provided for all required quantiles
        if len(quantile_forecasts) != 9:
            raise ValueError("Expected 9 quantile forecasts (0.1 to 0.9).")

        # Unpack the quantile forecasts
        (idc_price_vector_01, idc_price_vector_02, idc_price_vector_03,
         idc_price_vector_04, idc_price_vector_05, idc_price_vector_06,
         idc_price_vector_07, idc_price_vector_08, idc_price_vector_09) = quantile_forecasts

        # Initialize pyomo model:

        model = pyo.ConcreteModel()

        # Set parameters:
        # Number of hours
        model.H = pyo.RangeSet(0, len(idc_price_vector_01)/4-1)
        # Number of quarters
        model.Q = pyo.RangeSet(1, len(idc_price_vector_01))
        # Number of quarters plus 1
        model.Q_plus_1 = pyo.RangeSet(1, len(idc_price_vector_01)+1)
        # Daily discharged energy limit
        volume_limit = energy_cap * n_cycles

        # Initialize variables:
        # State of charge
        model.soc = pyo.Var(model.Q_plus_1, domain=pyo.Reals)
        # Charges on the intraday auction
        model.cha_idc = pyo.Var(model.Q, domain=pyo.NonNegativeReals, bounds=(0, 1))
        # Discharges on the intraday auction
        model.dis_idc = pyo.Var(model.Q, domain=pyo.NonNegativeReals, bounds=(0, 1))
        # Charges on the intraday auction to close previous positions from the day-ahead auction
        model.cha_idc_close = pyo.Var(model.Q, domain=pyo.NonNegativeReals, bounds=(0, 1))
        # Charges on the intraday auction to close previous positions from the day-ahead auction
        model.dis_idc_close = pyo.Var(model.Q, domain=pyo.NonNegativeReals, bounds=(0, 1))

        # Set Constraints:
        @model.Constraint(model.Q_plus_1)
        def set_maximum_soc(model, q):
            """ State of charge can never be higher than Energy Capacity. (Constraint 3.1) """
            return model.soc[q] <= energy_cap

        @model.Constraint(model.Q_plus_1)
        def set_minimum_soc(model, q):
            """ State of charge can never be less than 0. (Constraint 3.2) """
            return model.soc[q] >= 0

        @model.Constraint()
        def set_first_soc_to_0(model):
            """ State of charge at the first quarter must be 0. (Constraint 3.3) """
            return model.soc[1] == 0.5

        @model.Constraint()
        def set_last_soc_to_0(model):
            """ State of charge at quarter 97 (i.e., first quarter of next day) must be 0. (Constraint 3.4) """
            return model.soc[97] == 0.5

        @model.Constraint(model.Q)
        def soc_step_constraint(model, q):
            """ The state of charge of each quarter equals the state if charge of the previous quarter plus charges minus discharges. (Constraint 3.5) """
            return model.soc[q+1] == model.soc[q] + power_cap/4 * (model.cha_idc[q] - model.dis_idc[q] + model.cha_idc_close[q] - model.dis_idc_close[q] + step2_cha_daaida[q-1] - step2_dis_daaida[q-1])

        @model.Constraint()
        def charge_cycle_limit(model):
            """ Sum of all charges has to be below the daily limit. (Constraint 3.6) """
            return (np.sum(step2_dis_daaida) + sum(model.dis_idc[q] for q in model.Q) - sum(model.cha_idc_close[q] for q in model.Q)) * power_cap / 4 <= volume_limit

        @model.Constraint()
        def discharge_cycle_limit(model):
            """ Sum of all discharges has to be below the daily limit. (Constraint 3.7) """
            return (np.sum(step2_cha_daaida) + sum(model.cha_idc[q] for q in model.Q) - sum(model.dis_idc_close[q] for q in model.Q)) * power_cap / 4 <= volume_limit

        @model.Constraint(model.Q)
        def cha_close_logic(model, q):
            """ cha_idc_close can only close or reduce existing dis_daaida positions. They can only be placed, where dis_daaida positions exist. (Constraint 3.8) """
            return model.cha_idc_close[q] <= step2_dis_daaida[q-1]

        @model.Constraint(model.Q)
        def dis_close_logic(model, q):
            """ dis_idc_close can only close or reduce existing cha_daaida positions. They can only be placed, where cha_daaida positions exist. (Constraint 3.9) """
            return model.dis_idc_close[q] <= step2_cha_daaida[q-1]

        @model.Constraint(model.Q)
        def charge_rate_limit(model, q):
            """ Sum of cha_idc[q] and cha_daaida[q] has to be less or equal to 1. (Constraint 3.10) """
            return model.cha_idc[q] + step2_cha_daaida[q-1] <= 1

        @model.Constraint(model.Q)
        def discharge_rate_limit(model, q):
            """ Sum of dis_idc[q] and dis_daaida[q] has to be less or equal to 1. (Constraint 3.11)"""
            return model.dis_idc[q] + step2_dis_daaida[q-1] <= 1

        # Define objective function and solve the optimization problem
        # The objective is to maximize revenue from ID Continuous trades over all possible charge-discharge schedules.
        model.obj = pyo.Objective(expr=sum([power_cap / 4 * (
                (idc_price_vector_05[q - 1]) *
                (model.dis_idc[q] + model.dis_idc_close[q] - model.cha_idc[q] - model.cha_idc_close[q]) -
                self.risk_factor * (idc_price_vector_05[q - 1] - idc_price_vector_01[q - 1]) * model.cha_idc[q] -
                self.risk_factor * (idc_price_vector_09[q - 1] - idc_price_vector_05[q - 1]) * model.dis_idc[q]
        )
                                            for q in model.Q]),
                                  sense=pyo.maximize)

        solver = pyo.SolverFactory("gurobi")
        solver.solve(model, timelimit=200)

        # Retrieve arrays of resulting optimal soc/charge/discharge schedules after the ID Auction:
        step3_soc_idc = [model.soc[q].value for q in range(1, len(idc_price_vector_01) + 1)]
        step3_cha_idc = [model.cha_idc[q].value for q in range(1, len(idc_price_vector_01) + 1)]
        step3_dis_idc = [model.dis_idc[q].value for q in range(1, len(idc_price_vector_01) + 1)]
        step3_cha_idc_close = [model.cha_idc_close[q].value for q in range(1, len(idc_price_vector_01) + 1)]
        step3_dis_idc_close = [model.dis_idc_close[q].value for q in range(1, len(idc_price_vector_01) + 1)]

        # step3_profit_idc = np.sum(((np.asarray(step3_dis_idc) + step3_dis_idc_close) - (
        #     np.asarray(step3_cha_idc) + step3_cha_idc_close)) * (0.05 * idc_price_vector_01 + 0.1 * idc_price_vector_02 + 0.1 * idc_price_vector_03 + 0.15 * idc_price_vector_04 + 0.15 * idc_price_vector_05 + 0.15 * idc_price_vector_06 + 0.1 * idc_price_vector_07 + 0.1 * idc_price_vector_08 + 0.1 * idc_price_vector_09)) * power_cap/4

        step3_profit_idc = 0

        # Calculate total physical charge discharge schedules of combined day-ahead and intraday auction trades:

        step3_cha_daaidaidc = np.asarray(
            step2_cha_daaida) - step3_dis_idc_close + step3_cha_idc
        step3_dis_daaidaidc = np.asarray(
            step2_dis_daaida) - step3_cha_idc_close + step3_dis_idc

        return(step3_soc_idc, step3_cha_idc, step3_dis_idc, step3_cha_idc_close, step3_dis_idc_close, step3_profit_idc, step3_cha_daaidaidc, step3_dis_daaidaidc)
