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
    io,
    Forecaster,
    compute_metrics,
    optimization_runner_gurobi,
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
        name="Perfect Foresight (Nominal)",
        battery_capacity=1,
        battery_power=1,
        cyclic_constraint=True,
        risk_factor=0.0,
        objective="perfect-foresight",
        efficiency=0.95,
    ),
    Scenario(
        name="Perfect Foresight (Low Capacity, Low Power)",
        battery_capacity=10,
        battery_power=10,
        cyclic_constraint=True,
        risk_factor=0.0,
        objective="perfect-foresight",
        cell_type = "LFP",
    ),
    Scenario(
        name="Perfect Foresight (High Capacity, Low Power)",
        battery_capacity=20,
        battery_power=10,
        cyclic_constraint=True,
        risk_factor=0.0,
        objective="perfect-foresight",
    ),
    Scenario(
        name="Perfect Foresight (High Power, Low Capacity)",
        battery_capacity=10,
        battery_power=20,
        cyclic_constraint=True,
        risk_factor=0.0,
        objective="perfect-foresight",
    ),
    Scenario(
        name="Perfect Foresight (High Power, High Capacity)",
        battery_capacity=20,
        battery_power=20,
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
    Scenario(
        name="Risk Adaptive",
        battery_capacity=1,
        battery_power=1,
        cyclic_constraint=True,
        risk_factor=0.0,
        objective="risk-aware",
    ),
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
    Scenario(
        name="Risk Adaptive PWL",
        battery_capacity=1,
        battery_power=1,
        cyclic_constraint=True,
        risk_factor="adaptive",
        objective="piece-wise",
    ),
]


if __name__ == "__main__":
    data_preprocessing = False
    train_forecasting = False
    do_forecasting = False
    evaluate_forecasting = True
    do_optimization = True
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

            scenario.parameters["market_data"] = market_data
            scenario.parameters["forecast_results"] = forecast_results

            scenario.run(optimization_runner_gurobi)

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
        optimization_results = {}
        battery_schedule = {}
        results2plot = []

        # Run all scenarios and save configurations
        for scenario in optimization_scenarios:
            name = scenario.parameters["name"]
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

            optimization_results[name] = pd.read_csv(scenario_file)
            battery_schedule[name] = pd.read_json(scenario_schedule_file)

            profits[name] = optimization_results[name]["cumulative_profit"].iloc[-1]
            print(f"💰 Cumulative Profit for {name}: {profits[name]} €")

            results2plot.append(name)

        # Plot the battery schedule for the last day
        fig, axes = plt.subplots(
            nrows=len(results2plot) + 1,
            figsize=(6.6, len(results2plot) * 5),
            sharex=True,
        )

        # First two subplots: Charging and Discharging Side by Side
        time_range = np.arange(96)
        for idx, idx_name in enumerate(results2plot):
            power = battery_schedule[idx_name].iloc[-1]
            axes[idx].bar(
                time_range,
                [-x for x in power["p_charge_daa_ida_idc"]],
                label=f"$P_{{ch}}$ for Scenario {idx_name}",
                color="#57ab27",
            )
            axes[idx].bar(
                time_range,
                power["p_discharge_daa_ida_idc"],
                label=f"$P_{{dch}}$ for Scenario {idx_name}",
                color="#e30066",
            )
            axes[idx].set_ylabel("Power [MW]")
            axes[idx].set_title(f"Battery Schedule - {idx_name}")
            axes[idx].legend(loc="upper left")
            axes[idx].set_ylim(-1, 1)
            # add grid
            axes[idx].grid(axis="y", alpha=0.3)

        # Third subplot: Forecast with Fill Between
        name = results2plot[0]
        last_day_schedule = battery_schedule[name].iloc[-1]
        axes[len(results2plot)].plot(
            last_day_schedule["daa_prices"],
            label="Actual Prices",
            color="black",
            linestyle="-",
        )
        axes[len(results2plot)].plot(
            last_day_schedule["daa_price_forecast"][4],
            label="Mean Forecast",
            color="#e30066",
            linestyle="--",
        )
        axes[len(results2plot)].fill_between(
            range(len(last_day_schedule["daa_price_forecast"][0])),
            last_day_schedule["daa_price_forecast"][0],
            last_day_schedule["daa_price_forecast"][-1],
            color="#00549f",
            alpha=0.3,
            label="90-10\% Quantile",
        )
        axes[len(results2plot)].plot(
            last_day_schedule["daa_price_forecast"][0],
            label="10\% Quantile",
            color="#00549f",
        )
        axes[len(results2plot)].plot(
            last_day_schedule["daa_price_forecast"][-1],
            label="90\% Quantile",
            color="#00549f",
        )
        axes[len(results2plot)].set_xlabel("Time")
        axes[len(results2plot)].set_ylabel("Price")
        axes[len(results2plot)].legend(loc="upper left")
        axes[len(results2plot)].grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plt.savefig(paths.results_dir / f"battery_schedule.pdf")
        # plt.show()

        # Plot the battery schedule for the last day
        fig, axes = plt.subplots(
            nrows=len(results2plot), figsize=(6.6, len(results2plot) * 3.5), sharex=True
        )

        # First two subplots: Charging and Discharging Side by Side
        time_range = np.arange(96)
        for idx, idx_name in enumerate(results2plot):
            power = battery_schedule[idx_name].iloc[-1]

            power["p_net_daa_ida_idc"] = np.subtract(
                power["p_discharge_daa_ida_idc"], power["p_charge_daa_ida_idc"]
            )

            axes[idx].plot(
                time_range,
                [x for x in power["p_net_daa_ida_idc"]],
                label=f"$P_{{net}}$ for Scenario {idx_name}",
                color="#00549f",
            )
            axes[idx].set_ylabel("Power [MW]")
            axes[idx].set_title(f"Battery Schedule - {idx_name}")
            axes[idx].legend(loc="upper left")
            # axes[idx].set_ylim(-1, 1)
            # add grid
            axes[idx].grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plt.savefig(paths.results_dir / f"battery_operational_schedule.pdf")
        # plt.show()

        # Plot the battery schedule for the last day
        fig, axes = plt.subplots(
            nrows=len(results2plot), figsize=(6.6, len(results2plot) * 3.5), sharex=True
        )

        # First two subplots: Charging and Discharging Side by Side
        time_range = np.arange(96)
        for idx, idx_name in enumerate(results2plot):
            power = battery_schedule[idx_name].iloc[-1]

            # Column to normalize
            column_to_normalize = ["p_discharge_daa_ida_idc", "p_charge_daa_ida_idc"]

            for column in column_to_normalize:

                # Min-Max Normalization for lists
                min_x = min(lst for lst in power[column])
                max_x = max(lst for lst in power[column])

                power[column + "_pu"] = [
                    (x - min_x) / (max_x - min_x) for x in power[column]
                ]

            power["p_net_daa_ida_idc_pu"] = np.subtract(
                power["p_discharge_daa_ida_idc_pu"], power["p_charge_daa_ida_idc_pu"]
            )

            axes[idx].plot(
                time_range,
                [x for x in power["p_net_daa_ida_idc_pu"]],
                label=f"$P_{{net}}$ for Scenario {idx_name}",
                color="#00549f",
            )
            axes[idx].set_ylabel("Power [MW (p.u.)]")
            axes[idx].set_title(f"Battery Schedule - {idx_name}")
            axes[idx].legend(loc="upper left")
            axes[idx].set_ylim(-1, 1)
            # add grid
            axes[idx].grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plt.savefig(paths.results_dir / f"battery_p_u_schedule.pdf")
        # plt.show()

        # Modify x-axis labels to enforce line breaks before "("
        formatted_labels = [label.replace(" (", "\n(") for label in profits.keys()]

        # Plot bar chart for profits
        profits_df = pd.DataFrame(profits, index=[0])
        plt.figure(figsize=(6.6, 5))
        profits_df.T.plot(kind="bar", legend=False)
        plt.xticks(
            ticks=range(len(formatted_labels)), labels=formatted_labels, rotation=90
        )  # Apply formatted labels
        plt.xlabel("Scenario")
        plt.ylabel("Cumulative Profit [€]")
        plt.title("Cumulative Profits for different Scenarios")
        plt.tight_layout()
        plt.grid(axis="y")
        plt.savefig(paths.results_dir / f"profits.pdf")
        # plt.show()

    print(f"\n⏳ Execution Time: {time.time() - start_time:.2f} seconds.")
