import os
from entsoe import EntsoePandasClient
import pandas as pd
import numpy as np

# zusätzliche Parameter einlesen
api_key = os.environ["ENTSOE_API_KEY"]
client = EntsoePandasClient(api_key=api_key)

start = pd.Timestamp("20210101", tz="Europe/Brussels")
end = pd.Timestamp("20220101", tz="Europe/Brussels")
country_code = "DE_LU"

# # methods that return Pandas Series
# client.query_day_ahead_prices(country_code, start=start, end=end)
# client.query_load(country_code, start=start, end=end)
# client.query_load_forecast(country_code, start=start, end=end)
# client.query_load_and_forecast(country_code, start=start, end=end)
# client.query_generation_forecast(country_code, start=start, end=end)
# client.query_wind_and_solar_forecast(country_code, start=start, end=end, psr_type=None)
# client.query_intraday_wind_and_solar_forecast(country_code, start=start, end=end, psr_type=None)


def get_res_day_ahead_forecast_data():
    df_res_daa = client.query_wind_and_solar_forecast(
        country_code, start=start, end=end, psr_type=None
    )

    df_res_daa["hour"] = df_res_daa.index.hour

    df_res_daa.reset_index(inplace=True)
    df_res_daa.rename(columns={"index": "timestamp"}, inplace=True)

    return df_res_daa


def get_res_intraday_forecast_data():
    df_res_ida = client.query_intraday_wind_and_solar_forecast(
        country_code, start=start, end=end, psr_type=None
    )

    df_res_ida["hour"] = df_res_ida.index.hour

    df_res_ida.reset_index(inplace=True)
    df_res_ida.rename(columns={"index": "timestamp"}, inplace=True)

    return df_res_ida


if __name__ == "__main__":
    df_res_1 = get_res_day_ahead_forecast_data()
    df_res_2 = get_res_intraday_forecast_data()

    if df_res_1["timestamp"].dtype != "datetime64[ns]":
        df_res_1["timestamp"] = pd.to_datetime(df_res_1["timestamp"])

    if df_res_2["timestamp"].dtype != "datetime64[ns]":
        df_res_2["timestamp"] = pd.to_datetime(df_res_2["timestamp"])

    df_res_1.to_csv("Renewable_Energy_DAA.csv", index=False)
    df_res_2.to_csv("Renewable_Energy_IDA.csv", index=False)

    print(df_res_1)
    print(df_res_2)
