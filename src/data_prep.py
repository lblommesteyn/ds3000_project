import pandas as pd
import numpy as np
from pathlib import Path
import kagglehub

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
  DATA_FILE_NAME = "flight_data_2024.csv"
  DATA_PATH = Path(DATA_FILE_NAME)

  if not DATA_PATH.exists():
    print("Downloading dataset...")
    download_root = kagglehub.dataset_download("hrishitpatil/flight-data-2024")
    downloaded_file = Path(download_root) / DATA_FILE_NAME
    downloaded_file.rename(DATA_PATH)
    print("Dataset downloaded to:", DATA_PATH)

  return pd.read_csv(
    DATA_PATH,
    dtype = dtype_map,
    parse_dates = parse_dates,
    low_memory = False,
  )

def hhmm_to_minutes(x):
  if pd.isna(x):
    return np.nan
  x = int(x)
  return (x // 100) * 60 + (x % 100)

def add_dep_hour(df: pd.DataFrame) -> pd.DataFrame:
  if "dep_hour" in df.columns:
    return df
  df = df.copy()
  df["crs_dep_time_min"] = df["crs_dep_time"].map(hhmm_to_minutes).astype("float32")
  df["dep_hour"] = (df["crs_dep_time_min"] // 60).astype("Int8")
  return df

def add_delay_label(df: pd.DataFrame, threshold: int = 15) -> pd.DataFrame:
  if "is_delayed_15" in df.columns:
    return df
  df = df.copy()
  df["arr_delay"] = pd.to_numeric(df["arr_delay"], errors = "coerce").astype("float32")
  df["is_delayed_15"] = (df["arr_delay"] > threshold).astype("Int8")
  return df

def filter_operated_flights(df: pd.DataFrame) -> pd.DataFrame:
  return df[(df["cancelled"] != 1) & (df["diverted"] != 1)].copy()

def add_predeparture_features(df: pd.DataFrame) -> pd.DataFrame:
  df = df.copy()
  df["is_weekend"] = (df["day_of_week"] >= 5).astype("int8")
  df["is_peak_summer"] = df["month"].isin([6, 7, 8]).astype("int8")
  df["distance_bucket"] = pd.cut(
    df["distance"],
    bins = [0, 300, 800, 1500, 3000, 6000],
    labels = [0, 1, 2, 3, 4],
    include_lowest = True
  ).astype("int8")
  return df

def sample_df(df: pd.DataFrame, n: int = 400_000, random_state: int = 42) -> pd.DataFrame:
  n_sample = min(n, len(df))
  return df.sample(n = n_sample, random_state = random_state)
