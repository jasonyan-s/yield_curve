from fredapi import Fred
import pandas as pd

def get_fred_yield_data(api_key, series_ids):
    fred = Fred(api_key=api_key)
    data = {}
    for label, series_id in series_ids.items():
        try:
            df = fred.get_series(series_id).dropna()
            data[label] = df.iloc[-1]  # Get latest value
        except Exception as e:
            data[label] = None
    return data
