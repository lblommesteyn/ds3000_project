# XGBoost on sampled subset
# this has data leakage which leads to inflated scores
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, classification_report

from xgboost import XGBClassifier

DATA_PATH = Path("flight_data_2024.csv")

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

df = pd.read_csv(
    DATA_PATH,
    dtype=dtype_map,
    parse_dates=parse_dates,
    low_memory=False,
)

# convert HHMM to minutes from midnight
def hhmm_to_minutes(x):
    if pd.isna(x):
        return np.nan
    x = int(x)
    return (x // 100) * 60 + (x % 100)

# basic features on df
if "dep_hour" not in df.columns:
    time_cols = ["crs_dep_time", "dep_time", "crs_arr_time", "arr_time"]
    for col in time_cols:
        df[col + "_min"] = df[col].map(hhmm_to_minutes).astype("float32")
    df["dep_hour"] = (df["crs_dep_time_min"] // 60).astype("Int8")

# convert to binary (arrival delay > 15 minutes)
if "is_delayed_15" not in df.columns:
    df["arr_delay"] = pd.to_numeric(df["arr_delay"], errors="coerce").astype("float32")
    df["is_delayed_15"] = (df["arr_delay"] > 15).astype("Int8")

# drop cancelled/diverted
df_model_base = df[(df["cancelled"] != 1) & (df["diverted"] != 1)].copy()

# extra indicators
df_model_base["is_weekend"] = (df_model_base["day_of_week"] >= 5).astype("int8")
df_model_base["is_peak_summer"] = df_model_base["month"].isin([6, 7, 8]).astype("int8")

# distance buckets
df_model_base["distance_bucket"] = pd.cut(
    df_model_base["distance"],
    bins=[0, 300, 800, 1500, 3000, 6000],
    labels=[0, 1, 2, 3, 4],
    include_lowest=True
).astype("int8")

# convert to binary
df_model_base["late_aircraft_binary"] = (df_model_base["late_aircraft_delay"] > 0).astype("int8")
df_model_base["nas_binary"] = (df_model_base["nas_delay"] > 0).astype("int8")

# select features
feature_cols = [
    "dep_hour",
    "month",
    "day_of_week",
    "distance_bucket",
    "is_weekend",
    "is_peak_summer",
    "late_aircraft_binary",
    "nas_binary",
    "origin",
    "dest",
    "op_unique_carrier",
]

target_col = "is_delayed_15"

model_df_fe = df_model_base[feature_cols + [target_col]].dropna()

# sample for speed (400k rows)
n_sample = min(400_000, len(model_df_fe))
model_df_fe = model_df_fe.sample(n=n_sample, random_state=42)

X = model_df_fe[feature_cols]
y = model_df_fe[target_col]

print(f"Using {len(X):,} rows for XGBoost")

# train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42,
)

# preprocessing and XGBoost pipeline
categorical_features = ["origin", "dest", "op_unique_carrier"]
numeric_features = [
    "dep_hour",
    "month",
    "day_of_week",
    "distance_bucket",
    "is_weekend",
    "is_peak_summer",
    "late_aircraft_binary",
    "nas_binary",
]

preprocess_fe = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", "passthrough", numeric_features),
    ]
)

neg, pos = np.bincount(y_train)
scale_pos_weight = neg / pos

xgb_pipeline = Pipeline(
    steps=[
        ("prep", preprocess_fe),
        ("model", XGBClassifier(
            tree_method="hist",
            n_estimators=400,
            max_depth=7,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.8,
            eval_metric="auc",
            n_jobs=-1,
            random_state=42,
            scale_pos_weight=scale_pos_weight,
        ))
    ]
)

# fit and evaluate
xgb_pipeline.fit(X_train, y_train)

y_proba = xgb_pipeline.predict_proba(X_test)[:, 1]
y_pred = (y_proba > 0.5).astype(int)

auc = roc_auc_score(y_test, y_proba)
print(f"\nXGBoost with engineered features – ROC-AUC: {auc:.4f}\n")
print(classification_report(y_test, y_pred))
