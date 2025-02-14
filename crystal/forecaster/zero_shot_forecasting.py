#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Small description of zero_shot_forecasting

Copyright (c) by Institute for High Voltage Equipment and Grids,
Digitalization and Energy Economics (IAEW), RWTH Aachen University,
16.01.2025, s.kortmann. All rights reserved.
"""

# load packages
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

plt.style.use(["rwth-latex"])
from sklearn.metrics import mean_squared_error, r2_score
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
from autogluon.features.generators import DatetimeFeatureGenerator
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# load dataset and create date features
df = pd.read_csv("daa_price_vector_full.csv")
df["Date"] = pd.to_datetime(df["timestamp"])
# Only use every fourth entry
df = df.iloc[::4]
df = df.rename(columns={"daa_price_vector": "cnt"})
# df['year'] = df.Date.dt.year
df["month"] = df.Date.dt.month
df["day"] = df.Date.dt.day
df["dayofweek"] = df.Date.dt.dayofweek
df["hour"] = df.Date.dt.hour
df = df.drop(columns="timestamp", axis=1)
df["item_id"] = 1
print(df.head())

# split train and test data
train_data = df[df["Date"] < "2021-12-01"]
test_data = df[df["Date"] >= "2021-12-01"].iloc[:96]

# prepare for modeling
df["item_id"] = 1
train_data = TimeSeriesDataFrame.from_data_frame(
    df=train_data, id_column="item_id", timestamp_column="Date"
)
print(train_data)

test_data = TimeSeriesDataFrame.from_data_frame(
    df=test_data, id_column="item_id", timestamp_column="Date"
)
print(test_data)

all_data = TimeSeriesDataFrame.from_data_frame(
    df=pd.concat([train_data, test_data]), id_column="item_id", timestamp_column="Date"
)
print(all_data)

# create known covariate list
cov = list(df.drop(columns=["cnt", "item_id", "Date"]))

# create and fit model
predictor = TimeSeriesPredictor(
    prediction_length=96, target="cnt", eval_metric="WQL", known_covariates_names=cov
)

predictor.fit(
    train_data, presets="best_quality", num_val_windows=2, time_limit=600, verbosity=2
)

leaderboard = predictor.leaderboard(train_data, silent=True)
print(leaderboard)

predictions = predictor.predict(train_data, known_covariates=test_data)
print(predictions.head())

# return evaluation metric
rmse = mean_squared_error(
    test_data.reset_index()["cnt"], predictions.reset_index()["mean"]
)
print("RSME before optimization: ", rmse)

fig, ax = plt.subplots()
quantiles = [f"{i/10:.1f}" for i in range(1, 10)]
predictions["probabilistic_forecast"] = (
    0.05 * predictions["0.1"]
    + 0.1 * predictions["0.2"]
    + 0.1 * predictions["0.3"]
    + 0.15 * predictions["0.4"]
    + 0.15 * predictions["0.5"]
    + 0.15 * predictions["0.6"]
    + 0.1 * predictions["0.7"]
    + 0.1 * predictions["0.8"]
    + 0.1 * predictions["0.9"]
)
for quantile in quantiles:
    ax.plot(
        predictions.reset_index().index,
        predictions.reset_index()[quantile],
        label=f"{int(float(quantile) * 100)}%-Quantil",
    )
ax.plot(
    predictions.reset_index().index,
    predictions.reset_index()["probabilistic_forecast"],
    label="Probabilistic Forecast",
)
ax.plot(
    predictions.reset_index().index,
    predictions.reset_index()["mean"],
    label="Forecasted DAA Price",
)
plt.xlabel("Time")
plt.ylabel("Price")
plt.title("Forecasted DAA Price")
plt.legend()
plt.show()

# create error table for evaluation
error_tbl = pd.DataFrame(
    data=predictions.reset_index()["mean"] - test_data.reset_index()["cnt"],
    columns=["autogluon_error"],
)
error_tbl["autogluon_error_pct"] = (
    error_tbl["autogluon_error"] / test_data.reset_index()["cnt"]
)
error_tbl["actual"] = test_data.reset_index()["cnt"]
error_tbl["pred"] = predictions.reset_index()["mean"]
error_tbl["prob"] = (
    0.05 * predictions.reset_index()["0.1"]
    + 0.1 * predictions.reset_index()["0.2"]
    + 0.1 * predictions.reset_index()["0.3"]
    + 0.15 * predictions.reset_index()["0.4"]
    + 0.15 * predictions.reset_index()["0.5"]
    + 0.15 * predictions.reset_index()["0.6"]
    + 0.1 * predictions.reset_index()["0.7"]
    + 0.1 * predictions.reset_index()["0.8"]
    + 0.1 * predictions.reset_index()["0.9"]
)
error_tbl["prob_error"] = error_tbl["prob"] - test_data.reset_index()["cnt"]
error_tbl = pd.concat(
    [test_data.drop(columns="cnt").reset_index(), error_tbl.reset_index()], axis=1
)
print(error_tbl)

# plots including absolute error
plt.clf()
plt.plot(error_tbl.index, error_tbl.actual)
plt.plot(error_tbl.index, error_tbl.pred)
plt.plot(error_tbl.index, error_tbl.autogluon_error)
plt.plot(error_tbl.index, error_tbl.prob)
plt.plot(error_tbl.index, error_tbl.prob_error)
plt.legend((["actual", "pred", "absolute error", "prob", "prob error"]))
plt.show()

# find optimal lags
# pacf chart
plot_pacf(df.cnt)
plt.show()

# add lags and resplit data
for i in range(1, 24):
    df["lag" + str(i)] = df.cnt.shift(i)
df["diff"] = df.cnt.diff()
df = df.tail(len(df) - 7)
print(df)

df_res = pd.read_csv("Renewable_Energy_DAA.csv")
df["Date"] = pd.to_datetime(df["timestamp"])
df_res = df_res.drop(columns=["Date", "hour"])
df = pd.merge(df, df_res, on="Date")

train_data = df[df["Date"] < "2021-12-01"]
test_data = df[df["Date"] >= "2021-12-01"].iloc[:96]

train_data = TimeSeriesDataFrame.from_data_frame(
    df=train_data, id_column="item_id", timestamp_column="Date"
)
print(train_data)

test_data = TimeSeriesDataFrame.from_data_frame(
    df=test_data, id_column="item_id", timestamp_column="Date"
)
print(test_data)

all_data = TimeSeriesDataFrame.from_data_frame(
    df=pd.concat([train_data, test_data]), id_column="item_id", timestamp_column="Date"
)
print(all_data)

# create known covariate list
cov = list(
    df.drop(
        columns=["cnt", "item_id", "Date", "diff"]
        + [col for col in df.columns if "lag" in col]
    )
)

# create and fit model
predictor = TimeSeriesPredictor(
    prediction_length=96, target="cnt", eval_metric="WQL", known_covariates_names=cov
)

predictor.fit(
    train_data, presets="best_quality", num_val_windows=2, time_limit=600, verbosity=2
)

leaderboard = predictor.leaderboard(train_data, silent=True)
print(leaderboard)

predictions = predictor.predict(train_data, known_covariates=test_data)
print(predictions.head())

# return evaluation metric
rmse = mean_squared_error(
    test_data.reset_index()["cnt"], predictions.reset_index()["mean"]
)
print("RSME after optimization: ", rmse)

fig, ax = plt.subplots()
quantiles = [f"{i/10:.1f}" for i in range(1, 10)]
predictions.reset_index()["probabilistic_forecast"] = (
    0.05 * predictions.reset_index()["0.1"]
    + 0.1 * predictions.reset_index()["0.2"]
    + 0.1 * predictions.reset_index()["0.3"]
    + 0.15 * predictions.reset_index()["0.4"]
    + 0.15 * predictions.reset_index()["0.5"]
    + 0.15 * predictions.reset_index()["0.6"]
    + 0.1 * predictions.reset_index()["0.7"]
    + 0.1 * predictions.reset_index()["0.8"]
    + 0.1 * predictions.reset_index()["0.9"]
)
for quantile in quantiles:
    ax.plot(
        predictions.reset_index().index,
        predictions.reset_index()[quantile],
        label=f"{int(float(quantile) * 100)}%-Quantil",
    )
ax.plot(
    predictions.reset_index().index,
    predictions.reset_index()["probabilistic_forecast"],
    label="Probabilistic Forecast",
)
ax.plot(
    predictions.reset_index().index,
    predictions.reset_index()["mean"],
    label="Forecasted DAA Price",
)
plt.xlabel("Time")
plt.ylabel("Price")
plt.title("Forecasted DAA Price")
plt.legend()
plt.show()

# #return feature importance
# fimportance = TimeSeriesPredictor.feature_importance(predictor)
# fimportance = fimportance.sort_values('importance')
#
# plt.figure(figsize=(12,5))
# plt.barh(fimportance.index, fimportance['importance'])
# plt.title('Importance')
# plt.show()

# create error table for evaluation
error_tbl = pd.DataFrame(
    data=predictions.reset_index()["mean"] - test_data.reset_index()["cnt"],
    columns=["autogluon_error"],
)
error_tbl["autogluon_error_pct"] = (
    error_tbl["autogluon_error"] / test_data.reset_index()["cnt"]
)
error_tbl["actual"] = test_data.reset_index()["cnt"]
error_tbl["pred"] = predictions.reset_index()["mean"]
error_tbl["prob"] = (
    0.05 * predictions.reset_index()["0.1"]
    + 0.1 * predictions.reset_index()["0.2"]
    + 0.1 * predictions.reset_index()["0.3"]
    + 0.15 * predictions.reset_index()["0.4"]
    + 0.15 * predictions.reset_index()["0.5"]
    + 0.15 * predictions.reset_index()["0.6"]
    + 0.1 * predictions.reset_index()["0.7"]
    + 0.1 * predictions.reset_index()["0.8"]
    + 0.1 * predictions.reset_index()["0.9"]
)
error_tbl["prob_error"] = error_tbl["prob"] - test_data.reset_index()["cnt"]
error_tbl = pd.concat(
    [test_data.drop(columns="cnt").reset_index(), error_tbl.reset_index()], axis=1
)
print(error_tbl)
