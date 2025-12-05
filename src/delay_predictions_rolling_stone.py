import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv("../flight_data_2024.csv")

df['fl_date'] = pd.to_datetime(df['fl_date'], errors='coerce')

# Ensure datetime
df["fl_date"] = pd.to_datetime(df["fl_date"], errors="coerce")

# Remove rows with unusable delay data
delay_df = df.dropna(subset=["arr_delay"])

# Aggregate DAILY features
daily = (
    delay_df.groupby("fl_date")
    .agg(
        mean_delay=("arr_delay", "mean"),
        std_delay=("arr_delay", "std"),
        flight_count=("arr_delay", "size"),
        cancel_rate=("cancelled", "mean"),
    )
    .reset_index()
)

# Fill NaN (std with small days)
daily = daily.fillna(0)

# Feature Engineering: Calendar
daily["day_of_week"] = daily["fl_date"].dt.weekday
daily["day_of_year"] = daily["fl_date"].dt.dayofyear

# Rolling historical lag features
daily["delay_3d_avg"] = daily["mean_delay"].rolling(3).mean()
daily["delay_7d_avg"] = daily["mean_delay"].rolling(7).mean()

daily = daily.dropna()   # drop first few rolling rows

FEATURES = [
    "std_delay",
    "flight_count",
    "cancel_rate",
    "day_of_week",
    "day_of_year",
    "delay_3d_avg",
    "delay_7d_avg"
]

TARGET = "mean_delay"

# Train
train_size = int(len(daily) * 0.85)
train_df = daily.iloc[:train_size]
test_df  = daily.iloc[train_size:]

X_train = train_df[FEATURES]
y_train = train_df[TARGET]

X_test  = test_df[FEATURES]
y_test  = test_df[TARGET]


# Training
model = RandomForestRegressor(
    n_estimators=300,
    max_depth=None,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)


# Model evaluation
pred_test = model.predict(X_test)

mae = mean_absolute_error(y_test, pred_test)
r2  = r2_score(y_test, pred_test)

print("\n===== DAILY DELAY MODEL PERFORMANCE =====")
print(f"MAE : {mae:.2f} minutes")
print(f"R2  : {r2:.3f}")


# Training Graph
plt.figure(figsize=(12,6))
plt.plot(daily["fl_date"], daily["mean_delay"], label="Actual")
plt.plot(test_df["fl_date"], pred_test, label="Predicted (Test)", linestyle="dashed")

plt.title("Daily Mean Arrival Delay — Actual vs Predicted")
plt.xlabel("Date")
plt.ylabel("Average Delay (minutes)")
plt.legend()
plt.tight_layout()
plt.show()

# Feature importance
importances = pd.Series(model.feature_importances_, index=FEATURES)
importances = importances.sort_values(ascending=False)

print("\n===== MOST IMPORTANT FEATURES =====")
print(importances)

importances.plot(kind="barh", figsize=(8,6))
plt.title("Daily Delay Forecast Feature Importance")
plt.tight_layout()
plt.show()

# Delay Forcast
future_days = 120

# Last rolling data
last_window = daily.iloc[-7:].copy()

future_records = []

for i in range(1, future_days+1):

    new_date = daily["fl_date"].max() + pd.Timedelta(days=i)

    record = {
        "fl_date": new_date,
        "std_delay": last_window["std_delay"].mean(),
        "flight_count": last_window["flight_count"].mean(),
        "cancel_rate": last_window["cancel_rate"].mean(),
        "day_of_week": new_date.weekday(),
        "day_of_year": new_date.dayofyear,
        "delay_3d_avg": last_window["mean_delay"].tail(3).mean(),
        "delay_7d_avg": last_window["mean_delay"].tail(7).mean()
    }

    pred_delay = model.predict(pd.DataFrame([record])[FEATURES])[0]
    record["prediction"] = pred_delay

    # Add to results
    future_records.append(record)

    # Feed prediction back into rolling window
    last_window = pd.concat([
        last_window.iloc[1:],
        pd.DataFrame([{
            "std_delay": record["std_delay"],
            "flight_count": record["flight_count"],
            "cancel_rate": record["cancel_rate"],
            "mean_delay": pred_delay
        }])
    ], ignore_index=True)

future_df = pd.DataFrame(future_records)

# Graph
plt.figure(figsize=(12,6))

plt.plot(daily["fl_date"], daily["mean_delay"], label="Historical")
plt.plot(future_df["fl_date"], future_df["prediction"], linestyle="dashed", color="red", label="Forecast")

plt.title("30-Day Daily Arrival Delay Forecast")
plt.xlabel("Date")
plt.ylabel("Predicted Average Delay (minutes)")
plt.legend()
plt.tight_layout()
plt.show()

print("\n===== NEXT 30 DAYS: PREDICTED DAILY ARRIVAL DELAYS =====")
print(future_df[["fl_date", "prediction"]])
