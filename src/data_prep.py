import pandas as pd
from pathlib import Path

dtype_map = {
  "op_unique_carrier": "category",
  "origin": "category",
  "origin_city_name": "category",
  "origin_state_nm": "category",
  "dest": "category",
  "dest_city_name": "category",
  "dest_state_nm": "category",
  "cancelled": "Int8",
  "diverted": "Int8",
  "distance": "float32",
}

parse_dates = ["fl_date"]

def load_df():
  DATA_PATH = Path("flight_data_2024.csv")
  return pd.read_csv(
    DATA_PATH,
    dtype = dtype_map,
    parse_dates = parse_dates,
    low_memory = False,
  )
