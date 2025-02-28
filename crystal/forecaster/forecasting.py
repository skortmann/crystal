import numpy as np
import pandas as pd
from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame
import os

WEEKEND_INDICES = [5, 6]


class Forecaster:
    def __init__(
        self,
        market: str,
        model_path: str,
        prediction_length: int,
        target_column: str = "value",
        eval_metric: str = "WQL",
    ):
        """
        Initializes the Forecaster class.

        Parameters:
        - model_path (str): Directory where the model will be saved/loaded.
        - prediction_length (int): Number of future time steps to predict.
        - target_column (str): Name of the column containing the target variable.
        - eval_metric (str): Evaluation metric for the predictor (default: 'WQL').
        """
        self.market = market
        self.model_path = model_path
        self.prediction_length = prediction_length
        self.target_column = target_column
        self.eval_metric = eval_metric
        self.predictor = None  # Store the trained model after loading or training

    def train(self, df):
        """
        Train a time series forecasting model using AutoGluon and save it to the object.

        Parameters:
        - df (pd.DataFrame): Time series data with 'timestamp' and target column.
        """
        df["item_id"] = 1  # Assuming a single item for simplicity
        df = df.drop(columns=["Date"])

        # Convert DataFrame to TimeSeriesDataFrame
        train_data = TimeSeriesDataFrame.from_data_frame(
            df, id_column="item_id", timestamp_column="timestamp"
        )

        # Initialize and train the predictor
        self.predictor = TimeSeriesPredictor(
            path=self.model_path,
            prediction_length=self.prediction_length,
            target=self.target_column,
            eval_metric=self.eval_metric,
            freq="15min" if self.market in ["ida", "idc"] else "1h",
            known_covariates_names=[
                "year",
                "month",
                "day",
                "dayofweek",
                "hour",
                "minute",
                "weekend",
                "sin_day_of_week",
                "cos_day_of_week",
                "sin_hour_of_day",
                "cos_hour_of_day",
            ],
        )

        self.predictor.fit(
            train_data,
            hyperparameters={
                "DeepAR": {},
                "DLinear": {},
                "PatchTST": {},
                "SimpleFeedForward": {},
                "TemporalFusionTransformer": {},
                "TiDE": {},
                "WaveNet": {},
            },
            num_val_windows=10,
            # refit_full=True,
            refit_every_n_windows=2,
            # time_limit=60 * 90,
            presets="best_quality",
            excluded_model_types=["Chronos"],
        )

        print(f"Model trained and saved at: {self.model_path}")

        # Calculate feature importance
        importance_df = self.predictor.feature_importance()

        # Display top features
        print(importance_df.head(10))

    def save(self):
        """
        Save the trained model if it has been initialized and trained.
        """
        if self.predictor:
            self.predictor.save()
            print(f"Model successfully saved at {self.model_path}.")
        else:
            raise ValueError(
                "No trained model available to save. Please train the model first."
            )

    def load(self):
        """
        Load the trained model from the specified path and store it in the object.
        """
        if os.path.exists(self.model_path):
            self.predictor = TimeSeriesPredictor.load(self.model_path)
            print(f"Model loaded from {self.model_path}.")
        else:
            raise ValueError(
                f"No model found at {self.model_path}. Please train and save the model first."
            )

    def predict(self, df):
        """
        Generate forecasts using the loaded or trained model.

        Parameters:
        - df (pd.DataFrame): Time series data to use for prediction, with 'timestamp' and target column.

        Returns:
        - pd.DataFrame: Forecasted values.
        """
        if self.predictor is None:
            raise ValueError("No model loaded. Please load or train the model first.")

        df = df.assign(item_id=1)  # Avoid SettingWithCopyWarning

        # Convert to TimeSeriesDataFrame
        prediction_data = TimeSeriesDataFrame.from_data_frame(
            df, id_column="item_id", timestamp_column="timestamp"
        )

        from autogluon.timeseries.utils.forecast import (
            get_forecast_horizon_index_ts_dataframe,
        )

        future_index = get_forecast_horizon_index_ts_dataframe(
            prediction_data, prediction_length=self.prediction_length
        )
        future_timestamps = future_index.get_level_values("timestamp")
        known_covariates = pd.DataFrame(index=future_index)
        known_covariates["weekend"] = future_timestamps.weekday.isin(
            WEEKEND_INDICES
        ).astype(float)
        known_covariates["year"] = future_timestamps.year
        known_covariates["month"] = future_timestamps.month
        known_covariates["day"] = future_timestamps.day
        known_covariates["dayofweek"] = future_timestamps.dayofweek
        known_covariates["hour"] = future_timestamps.hour
        known_covariates["minute"] = future_timestamps.minute

        # Cyclical encoding for time-based features
        known_covariates["sin_day_of_week"] = np.sin(
            2 * np.pi * known_covariates["dayofweek"] / 7
        )
        known_covariates["cos_day_of_week"] = np.cos(
            2 * np.pi * known_covariates["dayofweek"] / 7
        )

        if self.prediction_length == 96:  # 15-minute intervals
            known_covariates["sin_hour_of_day"] = np.sin(
                2 * np.pi * known_covariates["minute"] / (self.prediction_length * 15)
            )
            known_covariates["cos_hour_of_day"] = np.cos(
                2 * np.pi * known_covariates["minute"] / (self.prediction_length * 15)
            )
        else:  # Hourly intervals
            known_covariates["sin_hour_of_day"] = np.sin(
                2 * np.pi * known_covariates["hour"] / self.prediction_length
            )
            known_covariates["cos_hour_of_day"] = np.cos(
                2 * np.pi * known_covariates["hour"] / self.prediction_length
            )

        # known_covariates.drop(columns=['year', 'hour', 'minute'], inplace=True)

        # Make predictions
        forecasts = self.predictor.predict(
            prediction_data, known_covariates=known_covariates
        )
        return forecasts


# Example usage:
if __name__ == "__main__":

    df = pd.DataFrame()  # Load your time series data here

    # Initialize forecaster
    forecaster = Forecaster(model_path="models/forecast_model", prediction_length=96)

    # Train and save model
    forecaster.train(df)
    forecaster.save()

    # Load and predict
    forecaster.load()
    predictions = forecaster.predict(df)
    print(predictions)
