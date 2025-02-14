#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Improved Time Series Analyzer with model saving, loading, and enhanced stability.

Copyright (c) by Institute for High Voltage Equipment and Grids,
Digitalization and Energy Economics (IAEW), RWTH Aachen University,
06.02.2025, s.kortmann. All rights reserved.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from darts import TimeSeries
from darts.models import ARIMA, Theta, TBATS, NBEATSModel, TSMixerModel
from darts.metrics import mape, rmse
from darts.utils.utils import SeasonalityMode
from darts.dataprocessing.transformers.scaler import Scaler
from darts.utils.statistics import plot_acf
from joblib import Parallel, delayed, dump, load
import warnings

warnings.filterwarnings("ignore")

MODEL_SAVE_DIR = "saved_models"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)


class TimeSeriesAnalyzer:
    def __init__(self):
        add_encoders = {
            "cyclic": {"future": ["hour", "dayofweek"]},
            "datetime_attribute": {"future": ["hour", "dayofweek"]},
            "position": {"past": ["relative"], "future": ["relative"]},
            "transformer": Scaler(),
            "tz": "CET",
        }
        self.models = {
            # 'ARIMA': ARIMA(p=2, d=1, q=2, seasonal_order=(1, 1, 1, 24)),
            # 'Theta': Theta(season_mode=SeasonalityMode.ADDITIVE, seasonality_period=24),
            # 'TBATS': TBATS(use_trend=True, use_box_cox=False, seasonal_periods=[24]),
            "NBEATS": NBEATSModel(
                input_chunk_length=24 * 7,
                output_chunk_length=24,
                add_encoders=add_encoders,
            ),
            "TSMIXER": TSMixerModel(
                input_chunk_length=24 * 7,
                output_chunk_length=24,
                add_encoders=add_encoders,
            ),
        }
        self.model_store = {}

    def _load_csv_data(self, filepath):
        """Load and prepare sunspot data."""
        try:
            df = pd.read_csv(filepath)
            df["Date"] = pd.to_datetime(df["von"])
            df = df[df.produkt == "Stundenkontrakt"]
            df = (
                df[["Date", "preis"]]
                .drop_duplicates(subset=["Date"])
                .sort_values("Date")
            )
            series = TimeSeries.from_dataframe(
                df, "Date", "preis", fill_missing_dates=True, freq="1h"
            )
            scaler = Scaler()  # default uses sklearn's MinMaxScaler
            series = scaler.fit_transform(series)

            return series
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return None

    def load_data(self, filepath):
        return self._load_csv_data(filepath)

    def save_model(self, model, model_name):
        try:
            save_path = os.path.join(MODEL_SAVE_DIR, f"{model_name}.pt")
            model.save(save_path)
            print(f"Model '{model_name}' saved to {save_path}")
        except Exception as e:
            print(f"Error saving model {model_name}: {str(e)}")

    def load_model(self, model_name):
        load_path = os.path.join(MODEL_SAVE_DIR, f"{model_name}.pt")
        if os.path.exists(load_path):
            try:
                print(f"Loading saved model '{model_name}' from {load_path}")
                return model_name.load(load_path)
            except Exception as e:
                print(f"Error loading model {model_name}: {str(e)}")
        return None

    def analyze(self, series, train_test_split=0.8, parallel=False, n_jobs=-1):
        if series is None:
            print("No data to analyze")
            return None, None, None

        train, test = series.split_before(train_test_split)
        results = {}

        def train_and_evaluate_model(name, model, train, test):
            saved_model = self.load_model(name)

            if saved_model:
                model = saved_model
            else:
                try:
                    model.fit(train)
                    self.save_model(model, name)
                except Exception as e:
                    print(f"Error training {name}: {str(e)}")
                    return name, None

            try:
                pred = model.predict(len(test))
                metrics = self._calculate_metrics(test, pred)
                self._print_metrics(name, metrics)
                return name, {"prediction": pred, **metrics, "model": model}
            except Exception as e:
                print(f"Error making predictions with {name}: {str(e)}")
                return name, None

        # Train models in parallel
        if parallel == True:
            parallel_results = Parallel(n_jobs=n_jobs)(
                delayed(train_and_evaluate_model)(name, model, train, test)
                for name, model in self.models.items()
            )
        else:
            parallel_results = [
                train_and_evaluate_model(name, model, train, test)
                for name, model in self.models.items()
            ]

        # Collect results and store trained models
        for name, result in parallel_results:
            if result:
                results[name] = result
                self.model_store[name] = result["model"]

        return results, train, test

    def _calculate_metrics(self, actual, predicted):
        try:
            return {
                "MAPE": mape(actual, predicted),
                "RMSE": rmse(actual, predicted),
            }
        except Exception as e:
            print(f"Error calculating metrics: {str(e)}")
            return {"MAPE": np.nan, "RMSE": np.nan}

    def _print_metrics(self, model_name, metrics):
        print(f"{model_name} Performance:")
        for metric, value in metrics.items():
            if not np.isnan(value):
                print(f"{metric}: {value:.2f}")

    def plot_results(self, series, results, train, test, save_path="forecast.png"):
        if series is None or results is None:
            print("No data to plot")
            return

        plt.figure(figsize=(15, 7))
        train.plot(label="Training", alpha=0.6)
        test.plot(label="Test", alpha=0.6)

        if results:
            colors = plt.cm.rainbow(np.linspace(0, 1, len(results)))
            for (name, result), color in zip(results.items(), colors):
                if "prediction" in result:
                    result["prediction"].plot(
                        label=f'{name} (MAPE: {result["MAPE"]:.1f}%)', color=color
                    )

        plt.title("Preis Number Forecasting Comparison")
        plt.xlabel("Time")
        plt.ylabel("Preis Number")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()

    def backtesting(self, model, series):
        try:
            backtest_results = model.backtest(
                series,
                start=0.8,  # Use 80% of the data for the first training
                forecast_horizon=24,
                stride=1,
                retrain=False,
                verbose=True,
                metric=[mape, rmse],
            )
            print("Backtest Results:")
            print(f"MAPE: {backtest_results[0]:.2f}%")
            print(f"RMSE: {backtest_results[1]:.2f}")
            return model.historical_forecasts(
                series,
                start=0.8,
                forecast_horizon=10,
                stride=1,
                retrain=False,
                verbose=True,
            )
        except Exception as e:
            print(f"Error during backtesting: {str(e)}")
            return None

    def plot_historical_forecasts(self, series, historical_forecasts, name):
        if historical_forecasts is None:
            print(f"No historical forecasts to plot for {name}")
            return

        plt.figure(figsize=(12, 6))
        series.plot(label="Actual")
        historical_forecasts.plot(label="Forecast")
        plt.title(f"{name} - Historical Forecasts")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{name}_Backtest.png")
        plt.show()

        plt.figure(figsize=(10, 6))
        plot_acf(series, max_lag=50)
        plt.title("Autocorrelation Function")
        plt.savefig(f"{name}_ACF_plot.png")
        plt.show()


def main():
    analyzer = TimeSeriesAnalyzer()

    real_series = analyzer.load_data("../daa_price_vector.csv")
    real_results, real_train, real_test = analyzer.analyze(real_series)
    analyzer.plot_results(
        real_series, real_results, real_train, real_test, "preis_forecast.png"
    )

    for name, model in analyzer.model_store.items():
        historical_forecasts = analyzer.backtesting(model, real_series)
        analyzer.plot_historical_forecasts(real_series, historical_forecasts, name)


if __name__ == "__main__":
    main()
