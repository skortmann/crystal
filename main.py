#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Small description of main

Copyright (c) by Institute for High Voltage Equipment and Grids,
Digitalization and Energy Economics (IAEW), RWTH Aachen University,
28.01.2025, s.kortmann. All rights reserved.
"""

import os
import time
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
from crystal import (
    Paths,
    Scenario,
    SequentialEnergyArbitrage,
    io,
    Forecaster,
    compute_metrics,
    plot_forecast_results,
)
import matplotlib.pyplot as plt

if "rwth-latex" in plt.style.available:
    plt.style.use(["rwth-latex", "blue"])
plt.rcParams["font.size"] = 12

# Define paths
paths = Paths()

# Define multiple optimization scenarios with different risk factors & objectives
optimization_scenarios = [
    Scenario(
        name="Perfect Foresight",
        battery_capacity=1,
        battery_power=1,
        cyclic_constraint=True,
        risk_factor=0.0,
        objective="perfect-foresight",
    ),
    Scenario(
        name="Risk Low",
        battery_capacity=1,
        battery_power=1,
        cyclic_constraint=True,
        risk_factor=0.05,
        objective="risk-aware",
    ),
    Scenario(
        name="Risk Medium",
        battery_capacity=1,
        battery_power=1,
        cyclic_constraint=True,
        risk_factor=0.125,
        objective="risk-aware",
    ),
    Scenario(
        name="Risk High",
        battery_capacity=1,
        battery_power=1,
        cyclic_constraint=True,
        risk_factor=0.25,
        objective="risk-aware",
    ),
    Scenario(
        name="No Risk Penalty",
        battery_capacity=1,
        battery_power=1,
        cyclic_constraint=True,
        risk_factor=0.0,
        objective="risk-aware",
    ),
    # Scenario(
    #     name="Risk Adaptive",
    #     battery_capacity=1,
    #     battery_power=1,
    #     cyclic_constraint=True,
    #     risk_factor=0.0,
    #     objective="risk-aware",
    # ),
    Scenario(
        name="Risk Low PWL",
        battery_capacity=1,
        battery_power=1,
        cyclic_constraint=True,
        risk_factor=0.05,
        objective="piece-wise",
    ),
    Scenario(
        name="Risk Medium PWL",
        battery_capacity=1,
        battery_power=1,
        cyclic_constraint=True,
        risk_factor=0.125,
        objective="piece-wise",
    ),
    Scenario(
        name="Risk High PWL",
        battery_capacity=1,
        battery_power=1,
        cyclic_constraint=True,
        risk_factor=0.25,
        objective="piece-wise",
    ),
    # Scenario(
    #     name="Risk Adaptive PWL",
    #     battery_capacity=1,
    #     battery_power=1,
    #     cyclic_constraint=True,
    #     risk_factor="adaptive",
    #     objective="piece-wise",
    # ),
]


def optimization_runner(
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


if __name__ == "__main__":
    data_preprocessing = False
    train_forecasting = False
    do_forecasting = False
    evaluate_forecasting = False
    do_optimization = False
    post_processing = True

    start_time = time.time()

    # Dictionary to store market data
    market_data = {}
    forecast_models = {}

    if data_preprocessing:
        for market in ["daa", "ida", "idc"]:
            df = io.read_csv_data(
                paths.data_dir / f"iaew_marktdaten_{market}_epex.csv", market
            )
            df = io.feature_engineering(df, market)
            df.to_csv(paths.data_dir / f"{market}_price_vector_full.csv", index=False)
            market_data[market] = df
    else:
        for market in ["daa", "ida", "idc"]:
            market_data[market] = pd.read_csv(
                paths.data_dir / f"{market}_price_vector_full.csv"
            )

    print(f"📁 Data loaded")

    if train_forecasting:
        print("\n🏋️ Train forecasting models\n")

        for market in ["daa", "ida", "idc"]:
            if market == "daa":
                prediction_length = 24
            else:
                prediction_length = 96

            df_market = market_data[market].copy()

            # # **Train-Test Split (Location-Based)**
            # train_size = int(0.8 * len(df_market))  # Use 80% for training, 20% for testing
            # df_train = df_market.iloc[:train_size]  # Training set
            # df_test = df_market.iloc[train_size:]  # Hold-out test set for later evaluation

            # **Train-Test Split (Time-Based)**
            day_start_time = pd.to_datetime("2021-10-01 00:00:00")
            day_end_time = pd.to_datetime("2022-01-10 23:45:00")
            df_train = df_market.loc[df_market["timestamp"] < day_start_time]
            df_test = df_market.loc[
                (df_market["timestamp"] >= day_start_time)
                & (df_market["timestamp"] < day_end_time)
            ]

            # Save test set separately for later evaluation
            test_file = paths.results_dir / f"{market}_test_set.csv"
            df_test.to_csv(test_file, index=False)

            # Initialize forecaster for each market
            forecast_models[market] = Forecaster(
                market=market,
                model_path=paths.model_dir / Path(f"{market}_model"),
                prediction_length=prediction_length,
                target_column="value",
                eval_metric="WQL",
            )

            # Train and save the model
            forecast_models[market].train(df_train)
            forecast_models[market].save()

    if do_forecasting:
        print("\n🔮 Do forecasting for time series\n")

        forecast_results = {}

        for market in ["daa", "ida", "idc"]:
            print(f"Generating forecasts iteratively for {market}...")
            if market == "daa":
                prediction_length = 24
            else:
                prediction_length = 96

            # Load test set (Ground Truth)
            df_test = pd.read_csv(
                paths.results_dir / f"{market}_test_set.csv", parse_dates=["timestamp"]
            )

            # Initialize and load the forecaster for each market
            forecast_models[market] = Forecaster(
                market=market,
                model_path=paths.model_dir / Path(f"{market}_model"),
                prediction_length=prediction_length,
                target_column="value",
                eval_metric="WQL",
            )
            forecast_models[market].load()  # Load the trained model

            all_forecasts = []  # Store all forecasts for this market

            # Set window size to the latest 7 days of data (7 * prediction_length)
            window_size = 7 * prediction_length

            # Calculate the range of valid indices for the sliding window
            num_iterations = (
                len(df_test) - window_size
            ) // prediction_length + 1  # Number of prediction-length iterations

            # Use tqdm for progress tracking
            for current_index in tqdm(
                range(
                    7 * prediction_length,
                    num_iterations * prediction_length,
                    prediction_length,
                ),
                desc=f"Forecasting {market}",
            ):
                # Select the last 7 days of data
                recent_data = df_test.iloc[0 : current_index + window_size]

                # Generate forecast for the next prediction_length time steps
                forecast = forecast_models[market].predict(recent_data)

                # Generate timestamps for the forecasted period
                day_start_time = pd.to_datetime(
                    recent_data["timestamp"].iloc[-1]
                ) + pd.Timedelta(minutes=15 if market != "daa" else 60)
                forecast_timestamps = pd.date_range(
                    start=day_start_time,
                    periods=prediction_length,
                    freq="15min" if market != "daa" else "1h",
                )

                # Combine forecasted values with timestamps
                forecast["timestamp"] = forecast_timestamps

                # Store the forecast
                all_forecasts.append(forecast)

            # Concatenate all forecasts into a single DataFrame for this market
            market_forecast_df = pd.concat(all_forecasts).reset_index(drop=True)
            forecast_results[market] = market_forecast_df

            # Save the forecast DataFrame to the results directory
            forecast_file = paths.results_dir / f"{market}_forecast_results.csv"
            market_forecast_df.to_csv(forecast_file, index=False)

            print(f"Forecasts for {market} saved to {forecast_file}.")

        print("Iterative forecasting completed.")

    if evaluate_forecasting:
        print("\n🔎 Evaluate forecasting results\n")

        forecast_results = {}  # Store forecast results for each market
        market_data = {}  # Store market data for each market

        for market in ["daa", "ida", "idc"]:
            print(f"Evaluating forecasts for {market}...")
            if market == "daa":
                prediction_length = 24
            else:
                prediction_length = 96

            # Load test set (Ground Truth)
            df_test = pd.read_csv(
                paths.results_dir / f"{market}_test_set.csv", parse_dates=["timestamp"]
            )

            # Load ground truth and forecasts
            market_data[market] = pd.read_csv(
                paths.data_dir / f"{market}_price_vector_full.csv",
                parse_dates=["timestamp"],
            )
            forecast_results[market] = pd.read_csv(
                paths.results_dir / f"{market}_forecast_results.csv"
            )

            # Align timestamps
            df_test["timestamp"] = pd.to_datetime(df_test["timestamp"])
            forecast_results[market]["timestamp"] = pd.to_datetime(
                forecast_results[market]["timestamp"]
            )

            # Store results per day
            metrics_results = []

            for day_start in range(
                7 * prediction_length, len(df_test), prediction_length
            ):
                # Select the test set window for 1 day
                day_test_set = df_test.iloc[day_start : day_start + prediction_length]
                day_start_time = day_test_set["timestamp"].min()

                # Align corresponding forecast for this day
                day_forecasts = forecast_results[market].loc[
                    (forecast_results[market]["timestamp"] >= day_start_time)
                    & (
                        forecast_results[market]["timestamp"]
                        < day_start_time
                        + pd.Timedelta(
                            hours=(
                                prediction_length
                                if market == "daa"
                                else prediction_length / 4
                            )
                        )
                    )
                ]

                if (
                    len(day_test_set) < prediction_length
                    or len(day_forecasts) < prediction_length
                ):
                    print(
                        f"⚠️ Skipping evaluation for {day_start_time} due to insufficient data."
                    )
                    continue

                # Compute metrics
                daily_metrics = compute_metrics(
                    day_test_set, day_forecasts, prediction_length
                )
                daily_metrics["day"] = day_start_time

                metrics_results.append(daily_metrics)

            # Combine all daily metrics
            metrics_df = pd.concat(metrics_results).reset_index(drop=True)

            # Save the daily metrics to the results directory
            metrics_file = paths.results_dir / f"{market}_daily_forecast_evaluation.csv"
            metrics_df.to_csv(metrics_file, index=False)

            print(f"📊 Daily evaluation metrics for {market} saved to {metrics_file}")
            print(metrics_df)

    if do_optimization:
        print("\n🚀 Running Optimization for Multiple Scenarios\n")

        forecast_results = {}  # Store forecast results for each market
        market_data = {}  # Store market data for each market

        # Load the true price vectors for each market
        for market in ["daa", "ida", "idc"]:
            market_data[market] = pd.read_csv(
                paths.data_dir / f"{market}_price_vector_full.csv",
                parse_dates=["timestamp"],
            )
            forecast_results[market] = pd.read_csv(
                paths.results_dir / f"{market}_forecast_results.csv",
                parse_dates=["timestamp"],
            )

        # Run all scenarios and save configurations
        for scenario in optimization_scenarios:
            scenario.parameters["result_dir"] = paths.get_scenario_results_path(
                scenario.parameters["name"]
            )
            scenario_yaml_path = os.path.join(
                scenario.parameters["result_dir"], "config.yaml"
            )
            scenario.to_yaml(scenario_yaml_path)

            scenario.run(optimization_runner)

    if post_processing:
        print("\n🚀 Do postprocesing for each market\n")

        forecast_results = {}  # Store forecast results for each market
        market_data = {}  # Store market data for each market

        # Load the true price vectors for each market
        for market in ["daa", "ida", "idc"]:
            market_data[market] = pd.read_csv(
                paths.data_dir / f"{market}_price_vector_full.csv",
                parse_dates=["timestamp"],
            )
            forecast_results[market] = pd.read_csv(
                paths.results_dir / f"{market}_forecast_results.csv",
                parse_dates=["timestamp"],
            )

            plot_forecast_results(
                market_data[market],
                forecast_results[market],
                market,
                save_path=paths.results_dir / f"{market}_forecast_results.png",
            )

        profits = {}

        # Run all scenarios and save configurations
        for scenario in optimization_scenarios:
            name = scenario.parameters["name"]
            if name == "Perfect Foresight":
                continue
            scenario.parameters["result_dir"] = paths.get_scenario_results_path(
                scenario.parameters["name"]
            )
            scenario_file = (
                Path(scenario.parameters["result_dir"])
                / f"optimization_results_{name}.csv"
            )
            scenario_schedule_file = (
                Path(scenario.parameters["result_dir"])
                / f"battery_schedule_{name}.json"
            )

            optimization_results = pd.read_csv(scenario_file)
            battery_schedule = pd.read_json(scenario_schedule_file)

            profits[name] = optimization_results["cumulative_profit"].iloc[-1]
            print(f"💰 Cumulative Profit for {name}: {profits[name]} €")

        # Plot bar chart for profits
        profits_df = pd.DataFrame(profits, index=[0])
        plt.figure(figsize=(6.6, 5))
        profits_df.T.plot(kind="bar", legend=False)
        plt.xlabel("Scenario")
        plt.ylabel("Cumulative Profit [€]")
        plt.title(f"Cumulative Profits for different Scenarios")
        plt.tight_layout()
        plt.grid(axis="y")
        plt.savefig(paths.results_dir / f"profits.pdf")
        # plt.show()

    print(f"\n⏳ Execution Time: {time.time() - start_time:.2f} seconds.")
