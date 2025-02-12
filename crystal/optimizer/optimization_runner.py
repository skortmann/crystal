#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Small description of optimization_runner

Copyright (c) by Institute for High Voltage Equipment and Grids,
Digitalization and Energy Economics (IAEW), RWTH Aachen University,
12.02.2025, s.kortmann. All rights reserved.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from crystal.optimizer import energy_arbitrage_stochastic_optimizer

# get global variable forecast_results and market_data
global forecast_results, market_data


def optimization_runner(
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
