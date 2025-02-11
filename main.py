#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Small description of main

Copyright (c) by Institute for High Voltage Equipment and Grids, 
Digitalization and Energy Economics (IAEW), RWTH Aachen University, 
28.01.2025, s.kortmann. All rights reserved.
"""
import os
from pathlib import Path
import time
from tqdm import tqdm
import numpy as np
import pandas as pd
from crystal import Paths, Scenario, io, forecaster, compute_metrics, energy_arbitrage_stochastic_optimizer

# Define paths
paths = Paths()

# Define multiple scenarios
scenarios = [
    Scenario(name="Scenario 1", duration=24, battery_capacity=100, fcr_allocation=50),
    Scenario(name="Scenario 2", duration=48, battery_capacity=200, fcr_allocation=75, afrr_capacity=100),
    Scenario(name="Scenario 3", duration=72, battery_capacity=150, fcr_allocation=60, item_list=[1,2,3])
]


# Define a sample runner function
def sample_runner(name, duration, battery_capacity, fcr_allocation, **kwargs):
    print(f"Running scenario '{name}' with:")
    print(f"Duration: {duration} hours")
    print(f"Battery Capacity: {battery_capacity} kWh")
    print(f"FCR Allocation: {fcr_allocation} MW")
    for key, value in kwargs.items():
        print(f"{key}: {value}")
    return "Run completed."

def run_optimization(battery_capacity, battery_power, risk_factor, **kwargs):
    print(f"Running optimization with:")
    print(f"Battery Capacity: {battery_capacity} kWh")
    print(f"Battery Power: {battery_power} kW")
    print(f"Risk Factor: {risk_factor}")
    return "Run completed."

if __name__ == "__main__":

    # Run all scenarios and save their configurations
    for scenario in scenarios:
        scenario.parameters['result_dir'] = paths.get_scenario_results_path(scenario.parameters['name'])
        scenario_yaml_path = os.path.join(scenario.parameters['result_dir'], "config.yaml")
        scenario.to_yaml(scenario_yaml_path)
        result = scenario.run(sample_runner)
        print(result)

    data_preprocessing = False
    train_forecasting = False
    do_forecasting = False
    evaluate_forecasting = False
    run_optimization = True
    post_processing = False

    start_time = time.time()

    # Dictionary to store market data
    market_data = {}
    forecast_models = {}

    if data_preprocessing:
        for market in ['daa', 'ida', 'idc']:
            df = io.read_csv_data(paths.data_dir / f"{market}_price_vector.csv", market)
            df = io.feature_engineering(df, market)
            df.to_csv(paths.data_dir / f"{market}_price_vector_full.csv", index=False)
            market_data[market] = df
    else:
        for market in ['daa', 'ida', 'idc']:
            market_data[market] = pd.read_csv(paths.data_dir / f"{market}_price_vector_full.csv")

    print("Data loaded.")

    if train_forecasting:
        for market in ['daa', 'ida', 'idc']:
            if market == "daa":
                prediction_length = 24
            else:
                prediction_length = 96
            # Initialize forecaster for each market
            forecast_models[market] = forecaster.Forecaster(
                market=market,
                model_path=paths.model_dir / Path(f"{market}_model"),
                prediction_length=prediction_length,
                target_column="value",
                eval_metric="WQL",
            )

            # Train and save the model
            forecast_models[market].train(market_data[market])
            forecast_models[market].save()

    if do_forecasting:
        forecast_results = {}

        for market in ['daa', 'ida', 'idc']:
            print(f"Generating forecasts iteratively for {market}...")
            if market == "daa":
                prediction_length = 24
            else:
                prediction_length = 96

            # Initialize and load the forecaster for each market
            forecast_models[market] = forecaster.Forecaster(
                market=market,
                model_path=paths.model_dir / Path(f"{market}_model"),
                prediction_length=prediction_length,
                target_column="value",
                eval_metric="WQL",
            )
            forecast_models[market].load()  # Load the trained model

            df_market = market_data[market].copy()
            all_forecasts = []  # Store all forecasts for this market

            # Set window size to the latest 7 days of data (7 * prediction_length)
            window_size = 7 * prediction_length

            # Calculate the range of valid indices for the sliding window
            num_iterations = (len(df_market) - window_size) // prediction_length + 1  # Number of prediction-length iterations

            # Use tqdm for progress tracking
            for current_index in tqdm(range(0, num_iterations * prediction_length, prediction_length),
                                      desc=f"Forecasting {market}"):
                # Select the last 7 days of data
                recent_data = df_market.iloc[current_index:current_index + window_size]

                # Generate forecast for the next prediction_length time steps
                forecast = forecast_models[market].predict(recent_data)

                # Generate timestamps for the forecasted period
                day_start_time = pd.to_datetime(recent_data['timestamp'].iloc[-1]) + pd.Timedelta(minutes=15 if market != "daa" else 60)
                forecast_timestamps = pd.date_range(start=day_start_time, periods=prediction_length,
                                                    freq='15min' if market != "daa" else '1h')

                # Combine forecasted values with timestamps
                forecast['timestamp'] = forecast_timestamps

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
        for market in ['daa', 'ida', 'idc']:
            print(f"Evaluating forecasts for {market}...")
            if market == "daa":
                prediction_length = 24
            else:
                prediction_length = 96

            # Load ground truth and forecasts
            ground_truth = market_data[market]
            forecast_results[market] = pd.read_csv(paths.results_dir / f"{market}_forecast_results.csv")

            # Ensure timestamps are datetime
            ground_truth['timestamp'] = pd.to_datetime(ground_truth['timestamp'])
            forecast_results[market]['timestamp'] = pd.to_datetime(forecast_results[market]['timestamp'])

            # Align timestamps
            metrics_results = []

            # Iterate over each day of forecasted values
            for day_start in range(0, len(forecast_results[market]), prediction_length):
                # Select 1-day window
                day_forecasts = forecast_results[market].iloc[day_start:day_start + prediction_length]
                day_start_time = day_forecasts['timestamp'].min()

                # Align corresponding ground truth for this day
                day_ground_truth = ground_truth[
                    (ground_truth['timestamp'] >= day_start_time) &
                    (ground_truth['timestamp'] < day_start_time + pd.Timedelta(
                        hours=prediction_length if market == "daa" else prediction_length / 4))
                    ]

                if len(day_ground_truth) < prediction_length or len(day_forecasts) < prediction_length:
                    print(f"Skipping evaluation for {day_start_time} due to insufficient data.")
                    continue

                # Compute metrics for the day
                daily_metrics = compute_metrics(day_ground_truth, day_forecasts, prediction_length)
                daily_metrics['day'] = day_start_time  # Track the evaluation day

                metrics_results.append(daily_metrics)

            # Combine all daily metrics
            metrics_df = pd.concat(metrics_results).reset_index(drop=True)

            # Save the daily metrics to the results directory
            metrics_file = paths.results_dir / f"{market}_daily_forecast_evaluation.csv"
            metrics_df.to_csv(metrics_file, index=False)

            print(f"Daily evaluation metrics for {market} saved to {metrics_file}.")
            print(metrics_df)

    if run_optimization:
        print(f"Running optimization sequentially")
        optimizer_stochastic_flex = energy_arbitrage_stochastic_optimizer("sequential_decision_making", risk_factor=0.2)

        forecast_results = {} # Store forecast results for each market

        # Load the true price vectors for each market
        for market in ['daa', 'ida', 'idc']:
            market_data[market] = pd.read_csv(paths.data_dir / f"{market}_price_vector_full.csv",
                                              parse_dates=['timestamp'])
            forecast_results[market] = pd.read_csv(paths.results_dir / f"{market}_forecast_results.csv", parse_dates=['timestamp'])

        optimization_results = []  # Store optimization results for each day
        revenue_total = 0

        # Iterate over each day's forecasts
        for day_start in tqdm(forecast_results['idc']['timestamp'][::96]):
            # Handle DAA forecasts and optimization
            day_forecasts_daa_subset = forecast_results['daa'].loc[(forecast_results['daa']['timestamp'] >= day_start) &
                                                         (forecast_results['daa']['timestamp'] < day_start + pd.Timedelta(
                                                             hours=24))]
            daa_price_vector_true = market_data['daa'].loc[(market_data['daa']['timestamp'] >= day_start) &
                                                           (market_data['daa']['timestamp'] < day_start + pd.Timedelta(
                                                               hours=24))]['value']
            daa_price_vector_true = np.repeat(daa_price_vector_true.to_numpy(), 4)  # Extend DAA price vector

            if len(day_forecasts_daa_subset) < 24 or len(daa_price_vector_true) < 96:
                print(f"Skipping optimization for {day_start} due to insufficient forecast data.")
                continue

            # Extract quantile forecasts for DAA
            quantiles_daa = [day_forecasts_daa_subset[f'{q:.1f}'].values for q in np.arange(0.1, 1.0, 0.1)]
            quantiles_daa = [np.repeat(q, 4) for q in quantiles_daa]  # Extend DAA forecasts

            # Optimize DAA
            step1_soc_daa, step1_cha_daa, step1_dis_daa, step1_profit_daa = optimizer_stochastic_flex.optimize_daa(
                n_cycles=1,
                energy_cap=1,
                power_cap=1,
                quantile_forecasts=quantiles_daa
            )

            # Handle IDA forecasts and optimization
            day_forecasts_ida_subset = forecast_results['ida'].loc[(forecast_results['ida']['timestamp'] >= day_start) &
                                                         (forecast_results['ida']['timestamp'] < day_start + pd.Timedelta(
                                                             hours=24))]
            ida_price_vector_true = market_data['ida'].loc[(market_data['ida']['timestamp'] >= day_start) &
                                                           (market_data['ida']['timestamp'] < day_start + pd.Timedelta(
                                                               hours=24))]['value']
            quantiles_ida = [day_forecasts_ida_subset[f'{q:.1f}'].values for q in np.arange(0.1, 1.0, 0.1)]

            step2_soc_ida, step2_cha_ida, step2_dis_ida, step2_cha_ida_close, step2_dis_ida_close, step2_profit_ida, step2_cha_daaida, step2_dis_daaida = optimizer_stochastic_flex.optimize_ida(
                n_cycles=1,
                energy_cap=1,
                power_cap=1,
                step1_cha_daa=step1_cha_daa,
                step1_dis_daa=step1_dis_daa,
                quantile_forecasts=quantiles_ida
            )

            # Handle IDC forecasts and optimization
            day_forecasts_idc_subset = forecast_results['idc'].loc[(forecast_results['idc']['timestamp'] >= day_start) &
                                                         (forecast_results['idc']['timestamp'] < day_start + pd.Timedelta(
                                                             hours=24))]
            idc_price_vector_true = market_data['idc'].loc[(market_data['idc']['timestamp'] >= day_start) &
                                                           (market_data['idc']['timestamp'] < day_start + pd.Timedelta(
                                                               hours=24))]['value']
            quantiles_idc = [day_forecasts_idc_subset[f'{q:.1f}'].values for q in np.arange(0.1, 1.0, 0.1)]

            step3_soc_idc, step3_cha_idc, step3_dis_idc, step3_cha_idc_close, step3_dis_idc_close, step3_profit_idc, step3_cha_daaidaidc, step3_dis_daaidaidc = optimizer_stochastic_flex.optimize_idc(
                n_cycles=1,
                energy_cap=1,
                power_cap=1,
                step2_cha_daaida=step2_cha_daaida,
                step2_dis_daaida=step2_dis_daaida,
                quantile_forecasts=quantiles_idc
            )

            # Calculate daily profits
            dt = 1/4
            revenue_daa_today_stoc = np.sum(daa_price_vector_true * (np.asarray(step1_dis_daa) - np.asarray(step1_cha_daa))) * dt
            revenue_ida_today_stoc = np.sum(ida_price_vector_true * (
                        np.asarray(step2_dis_ida) + np.asarray(step2_dis_ida_close) - np.asarray(step2_cha_ida) - np.asarray(step2_cha_ida_close))) * dt
            revenue_idc_today_stoc = np.sum(idc_price_vector_true * (
                        np.asarray(step3_dis_idc) + np.asarray(step3_dis_idc_close) - np.asarray(step3_cha_idc) - np.asarray(step3_cha_idc_close))) * dt

            revenue_total += revenue_daa_today_stoc + revenue_ida_today_stoc + revenue_idc_today_stoc

            # Store the results for the current day
            optimization_results.append({
                'day': day_start,
                'soc': step3_soc_idc,
                'charge': step3_cha_idc,
                'discharge': step3_dis_idc,
                'revenue_daa': revenue_daa_today_stoc,
                'revenue_ida': revenue_ida_today_stoc,
                'revenue_idc': revenue_idc_today_stoc,
                'daily_profit': revenue_daa_today_stoc + revenue_ida_today_stoc + revenue_idc_today_stoc,
                'cummulated_profit': revenue_total,
            })

        print("Sequential optimization completed.")
        print("Total revenue: ", revenue_total)

        # Save the daily optimization results to a CSV
        optimization_results_df = pd.DataFrame(optimization_results)
        optimization_file = paths.results_dir / "daily_optimization_results.csv"
        optimization_results_df.to_csv(optimization_file, index=False)

        print(f"Daily optimization results saved to {optimization_file}.")

    if post_processing:
        pass

    print(f"Execution time: {time.time() - start_time:.2f} seconds.")
