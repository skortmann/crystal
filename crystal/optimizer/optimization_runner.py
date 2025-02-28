#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script was inspired by FlexPower https://github.com/FlexPwr/bess-optimizer

Copyright (c) by Institute for High Voltage Equipment and Grids,
Digitalization and Energy Economics (IAEW), RWTH Aachen University,
12.02.2025, s.kortmann. All rights reserved.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from crystal.optimizer import energy_arbitrage_stochastic_optimizer, SequentialEnergyArbitrage


def optimization_runner_pyomo(
    forecast_results, market_data,
    name, battery_capacity, battery_power, risk_factor, objective, **kwargs
):
    """
    Runs energy arbitrage stochastic optimization for a given scenario.
    """
    print(f"\n⚡ Running Optimization for Scenario: {name}\n")

    optimizer_stochastic_flex = energy_arbitrage_stochastic_optimizer(
        name=name,
        risk_factor=risk_factor,
        objective_function=objective,
    )

    optimization_results = []
    revenue_total = 0

    for day_start in tqdm(
        forecast_results["idc"]["timestamp"][::96], desc=f"Optimizing {name}"
    ):
        try:
            # Load day-ahead forecasts
            day_forecasts_daa_subset = forecast_results["daa"].loc[
                (forecast_results["daa"]["timestamp"] >= day_start)
                & (
                    forecast_results["daa"]["timestamp"]
                    < day_start + pd.Timedelta(hours=24)
                )
            ]
            daa_price_vector_true = market_data["daa"].loc[
                (market_data["daa"]["timestamp"] >= day_start)
                & (market_data["daa"]["timestamp"] < day_start + pd.Timedelta(hours=24))
            ]["value"]

            # Extend DAA price vector
            daa_price_vector_true = np.repeat(daa_price_vector_true.to_numpy(), 4)

            if len(day_forecasts_daa_subset) < 24 or len(daa_price_vector_true) < 96:
                print(
                    f"⚠️ Skipping optimization for {day_start} (Insufficient Forecast Data)"
                )
                continue

            # Extract quantile forecasts for DAA
            quantiles_daa = [
                day_forecasts_daa_subset[f"{q:.1f}"].values
                for q in np.arange(0.1, 1.0, 0.1)
            ]
            quantiles_daa = [
                np.repeat(q, 4) for q in quantiles_daa
            ]  # Extend DAA forecasts

            # Optimize DAA
            step1_soc_daa, step1_cha_daa, step1_dis_daa, step1_profit_daa = (
                optimizer_stochastic_flex.optimize_daa(
                    n_cycles=1,
                    energy_cap=battery_capacity,
                    power_cap=battery_power,
                    quantile_forecasts=quantiles_daa,
                )
            )

            # Load IDA forecasts
            day_forecasts_ida_subset = forecast_results["ida"].loc[
                (forecast_results["ida"]["timestamp"] >= day_start)
                & (
                    forecast_results["ida"]["timestamp"]
                    < day_start + pd.Timedelta(hours=24)
                )
            ]
            ida_price_vector_true = market_data["ida"].loc[
                (market_data["ida"]["timestamp"] >= day_start)
                & (market_data["ida"]["timestamp"] < day_start + pd.Timedelta(hours=24))
            ]["value"]

            quantiles_ida = [
                day_forecasts_ida_subset[f"{q:.1f}"].values
                for q in np.arange(0.1, 1.0, 0.1)
            ]

            (
                step2_soc_ida,
                step2_cha_ida,
                step2_dis_ida,
                step2_cha_ida_close,
                step2_dis_ida_close,
                step2_profit_ida,
                step2_cha_daaida,
                step2_dis_daaida,
            ) = optimizer_stochastic_flex.optimize_ida(
                n_cycles=1,
                energy_cap=battery_capacity,
                power_cap=battery_power,
                step1_cha_daa=step1_cha_daa,
                step1_dis_daa=step1_dis_daa,
                quantile_forecasts=quantiles_ida,
            )

            # Handle IDC forecasts
            day_forecasts_idc_subset = forecast_results["idc"].loc[
                (forecast_results["idc"]["timestamp"] >= day_start)
                & (
                    forecast_results["idc"]["timestamp"]
                    < day_start + pd.Timedelta(hours=24)
                )
            ]
            idc_price_vector_true = market_data["idc"].loc[
                (market_data["idc"]["timestamp"] >= day_start)
                & (market_data["idc"]["timestamp"] < day_start + pd.Timedelta(hours=24))
            ]["value"]

            quantiles_idc = [
                day_forecasts_idc_subset[f"{q:.1f}"].values
                for q in np.arange(0.1, 1.0, 0.1)
            ]

            (
                step3_soc_idc,
                step3_cha_idc,
                step3_dis_idc,
                step3_cha_idc_close,
                step3_dis_idc_close,
                step3_profit_idc,
                step3_cha_daaidaidc,
                step3_dis_daaidaidc,
            ) = optimizer_stochastic_flex.optimize_idc(
                n_cycles=1,
                energy_cap=battery_capacity,
                power_cap=battery_power,
                step2_cha_daaida=step2_cha_daaida,
                step2_dis_daaida=step2_dis_daaida,
                quantile_forecasts=quantiles_idc,
            )

            # Calculate daily profits
            dt = 1 / 4
            revenue_daa_today_stoc = (
                np.sum(
                    daa_price_vector_true
                    * (np.asarray(step1_dis_daa) - np.asarray(step1_cha_daa))
                )
                * dt
            )
            revenue_ida_today_stoc = (
                np.sum(
                    ida_price_vector_true
                    * (
                        np.asarray(step2_dis_ida)
                        + np.asarray(step2_dis_ida_close)
                        - np.asarray(step2_cha_ida)
                        - np.asarray(step2_cha_ida_close)
                    )
                )
                * dt
            )
            revenue_idc_today_stoc = (
                np.sum(
                    idc_price_vector_true
                    * (
                        np.asarray(step3_dis_idc)
                        + np.asarray(step3_dis_idc_close)
                        - np.asarray(step3_cha_idc)
                        - np.asarray(step3_cha_idc_close)
                    )
                )
                * dt
            )

            revenue_total += (
                revenue_daa_today_stoc + revenue_ida_today_stoc + revenue_idc_today_stoc
            )

            optimization_results.append(
                {
                    "day": day_start,
                    "revenue_daa": revenue_daa_today_stoc,
                    "revenue_ida": revenue_ida_today_stoc,
                    "revenue_idc": revenue_idc_today_stoc,
                    "daily_profit": revenue_daa_today_stoc
                    + revenue_ida_today_stoc
                    + revenue_idc_today_stoc,
                    "cumulative_profit": revenue_total,
                }
            )

        except Exception as e:
            print(f"⚠️ Error processing {day_start} in {name}: {e}")
            continue

    print(f"\n✅ Optimization Completed for: {name}")
    print(f"💰 Total Revenue: {revenue_total}")

    # Save results per scenario
    scenario_file = Path(kwargs["result_dir"]) / f"optimization_results_{name}.csv"
    pd.DataFrame(optimization_results).to_csv(scenario_file, index=False)

    print(f"📁 Results Saved: {scenario_file}")
    return revenue_total

def optimization_runner_gurobi(
    forecast_results, market_data,
    name,
    battery_capacity,
    battery_power,
    cyclic_constraint,
    risk_factor,
    objective,
    **kwargs,
):
    """
    Runs the SequentialEnergyArbitrage optimization framework for a given scenario.

    Parameters:
    - name: str -> Scenario name.
    - battery_capacity: int -> Energy capacity of the battery (MWh).
    - battery_power: int -> Power capacity of the battery (MW).
    - risk_factor: float -> Risk adjustment factor.
    - optimization_method: str -> Strategy name (e.g., "Risk Adaptive", "Risk High").
    - kwargs: dict -> Additional arguments (e.g., result directory).

    Returns:
    - revenue_total: float -> Total revenue for the scenario.
    """
    print(f"\n⚡ Running Optimization for Scenario: {name}\n")

    # Initialize the optimizer
    optimizer = SequentialEnergyArbitrage(
        energy_capacity=battery_capacity,
        power_capacity=battery_power,
        cyclic_constraint=cyclic_constraint,
        risk_factor=risk_factor,
        objective=objective,
    )

    optimization_results = []
    battery_schedule = []
    revenue_total = 0

    # Loop through each day in the forecast data
    for day_start in tqdm(
        forecast_results["idc"]["timestamp"][::96], desc=f"Optimizing {name}"
    ):
        try:
            # Load Day-Ahead Auction (DAA) forecasts
            day_forecasts_daa_subset = forecast_results["daa"].loc[
                (forecast_results["daa"]["timestamp"] >= day_start)
                & (
                    forecast_results["daa"]["timestamp"]
                    < day_start + pd.Timedelta(hours=24)
                )
            ]
            daa_price_vector_true = (
                market_data["daa"]
                .loc[
                    (market_data["daa"]["timestamp"] >= day_start)
                    & (
                        market_data["daa"]["timestamp"]
                        < day_start + pd.Timedelta(hours=24)
                    )
                ]["value"]
                .to_numpy()
            )

            # Extend DAA price vector to match time steps
            daa_price_vector_true = np.repeat(daa_price_vector_true, 4)

            if len(day_forecasts_daa_subset) < 24 or len(daa_price_vector_true) < 96:
                print(
                    f"⚠️ Skipping optimization for {day_start} (Insufficient Forecast Data)"
                )
                continue

            # Extract quantile forecasts for DAA
            quantiles_daa = [
                day_forecasts_daa_subset[f"{q:.1f}"].values
                for q in np.arange(0.1, 1.0, 0.1)
            ]
            quantiles_daa = [np.repeat(q, 4) for q in quantiles_daa]  # Extend forecasts

            # Optimize DAA
            results_daa = optimizer.optimizeDAA(quantiles_daa, daa_price_vector_true)

            # Load Intraday Auction (IDA) forecasts
            day_forecasts_ida_subset = forecast_results["ida"].loc[
                (forecast_results["ida"]["timestamp"] >= day_start)
                & (
                    forecast_results["ida"]["timestamp"]
                    < day_start + pd.Timedelta(hours=24)
                )
            ]
            ida_price_vector_true = (
                market_data["ida"]
                .loc[
                    (market_data["ida"]["timestamp"] >= day_start)
                    & (
                        market_data["ida"]["timestamp"]
                        < day_start + pd.Timedelta(hours=24)
                    )
                ]["value"]
                .to_numpy()
            )

            quantiles_ida = [
                day_forecasts_ida_subset[f"{q:.1f}"].values
                for q in np.arange(0.1, 1.0, 0.1)
            ]

            # Optimize IDA
            results_ida = optimizer.optimizeIDA(
                quantiles_ida, results_daa, ida_price_vector_true
            )

            # Load Intraday Continuous (IDC) forecasts
            day_forecasts_idc_subset = forecast_results["idc"].loc[
                (forecast_results["idc"]["timestamp"] >= day_start)
                & (
                    forecast_results["idc"]["timestamp"]
                    < day_start + pd.Timedelta(hours=24)
                )
            ]
            idc_price_vector_true = (
                market_data["idc"]
                .loc[
                    (market_data["idc"]["timestamp"] >= day_start)
                    & (
                        market_data["idc"]["timestamp"]
                        < day_start + pd.Timedelta(hours=24)
                    )
                ]["value"]
                .to_numpy()
            )

            quantiles_idc = [
                day_forecasts_idc_subset[f"{q:.1f}"].values
                for q in np.arange(0.1, 1.0, 0.1)
            ]

            # Optimize IDC
            results_idc = optimizer.optimizeIDC(
                quantiles_idc, results_ida, idc_price_vector_true
            )

            # Calculate daily profits
            dt = 1 / 4  # Since each time step is 15 minutes
            revenue_daa_today = (
                np.sum(
                    daa_price_vector_true
                    * (
                        np.asarray(results_daa["p_discharge_daa"])
                        - np.asarray(results_daa["p_charge_daa"])
                    )
                )
                * dt
            )
            revenue_ida_today = (
                np.sum(
                    ida_price_vector_true
                    * (
                        np.asarray(results_ida["p_discharge_ida"])
                        + np.asarray(results_ida["p_discharge_ida_close"])
                        - np.asarray(results_ida["p_charge_ida"])
                        - np.asarray(results_ida["p_charge_ida_close"])
                    )
                )
                * dt
            )
            revenue_idc_today = (
                np.sum(
                    idc_price_vector_true
                    * (
                        np.asarray(results_idc["p_discharge_idc"])
                        + np.asarray(results_idc["p_discharge_idc_close"])
                        - np.asarray(results_idc["p_charge_idc"])
                        - np.asarray(results_idc["p_charge_idc_close"])
                    )
                )
                * dt
            )

            revenue_total += revenue_daa_today + revenue_ida_today + revenue_idc_today

            # Store results for the day
            optimization_results.append(
                {
                    "timestamp": day_start,
                    "revenue_daa": revenue_daa_today,
                    "revenue_ida": revenue_ida_today,
                    "revenue_idc": revenue_idc_today,
                    "daily_profit": revenue_daa_today
                    + revenue_ida_today
                    + revenue_idc_today,
                    "cumulative_profit": revenue_total,
                }
            )

            # Store battery schedule for the day
            battery_schedule.append(
                {
                    "timestamp": day_start,
                    "p_charge_daa": results_daa["p_charge_daa"],
                    "p_discharge_daa": results_daa["p_discharge_daa"],
                    "soc_daa": results_daa["soc_daa"],
                    "p_charge_ida": results_ida["p_charge_ida"],
                    "p_charge_ida_close": results_ida["p_charge_ida_close"],
                    "p_discharge_ida": results_ida["p_discharge_ida"],
                    "p_discharge_ida_close": results_ida["p_discharge_ida_close"],
                    "soc_ida": results_ida["soc_ida"],
                    "p_charge_idc": results_idc["p_charge_idc"],
                    "p_charge_idc_close": results_idc["p_charge_idc_close"],
                    "p_discharge_idc": results_idc["p_discharge_idc"],
                    "p_discharge_idc_close": results_idc["p_discharge_idc_close"],
                    "p_charge_daa_ida_idc": results_idc["p_charge_daa_ida_idc"],
                    "p_discharge_daa_ida_idc": results_idc["p_discharge_daa_ida_idc"],
                    "soc_idc": results_idc["soc_idc"],
                    "daa_prices": daa_price_vector_true,
                    "ida_prices": ida_price_vector_true,
                    "idc_prices": idc_price_vector_true,
                    "daa_price_forecast": quantiles_daa,
                    "ida_price_forecast": quantiles_ida,
                    "idc_price_forecast": quantiles_idc,
                }
            )

        except Exception as e:
            print(f"⚠️ Error processing {day_start} in {name}: {e}")
            continue

    print(f"\n✅ Optimization Completed for: {name}")
    print(f"💰 Total Revenue: {revenue_total}")

    # Save results per scenario
    result_dir = kwargs.get("result_dir", "./results")  # Default directory
    Path(result_dir).mkdir(parents=True, exist_ok=True)
    scenario_file = Path(result_dir) / f"optimization_results_{name}.csv"
    scenario_schedule_file = Path(result_dir) / f"battery_schedule_{name}.json"

    pd.DataFrame(optimization_results).to_csv(scenario_file, index=False)
    pd.DataFrame(battery_schedule).to_json(
        scenario_schedule_file, orient="records", date_format="iso"
    )

    print(f"📁 Results Saved: {scenario_file}")
    return