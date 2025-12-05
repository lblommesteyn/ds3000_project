import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from data_prep import load_df

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_curve, roc_auc_score

df = load_df()

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

daily["lag1_delay"] = daily["mean_delay"].shift(1)
daily["lag2_delay"] = daily["mean_delay"].shift(2)
daily["lag3_delay"] = daily["mean_delay"].shift(3)
daily["lag7_delay"] = daily["mean_delay"].shift(7)
daily = daily.dropna()

FEATURES = [
    "std_delay",
    "flight_count",
    "cancel_rate",
    "day_of_week",
    "day_of_year",
    "lag1_delay",
    "lag2_delay",
    "lag3_delay",
    "lag7_delay"
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
    n_jobs=-1,
    verbose=3
)

model.fit(X_train, y_train)

# Model evaluation
pred_test = model.predict(X_test)

mae = mean_absolute_error(y_test, pred_test)
r2  = r2_score(y_test, pred_test)

print("\nDAILY DELAY MODEL PERFORMANCE")
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
os.makedirs("plots/lag", exist_ok=True)
plt.savefig("plots/lag/daily_delay.png")

# Feature importance
importances = pd.Series(model.feature_importances_, index=FEATURES)
importances = importances.sort_values(ascending=False)

print("\nMOST IMPORTANT FEATURES")
print(importances)

importances.plot(kind="barh", figsize=(8,6))
plt.title("Daily Delay Forecast Feature Importance")
plt.tight_layout()
plt.savefig("plots/lag/daily_delay_importance.png")

future_days = 30

last_window = daily.iloc[-7:].copy()

future_records = []

for i in range(1, future_days + 1):

    new_date = daily["fl_date"].max() + pd.Timedelta(days=i)

    record_features = {
        "std_delay": np.random.normal(
            last_window["std_delay"].mean(),
            last_window["std_delay"].std()
        ),

        "flight_count": int(
            max(1, np.random.normal(
                last_window["flight_count"].mean(),
                last_window["flight_count"].std()
            ))
        ),

        "cancel_rate": np.clip(
            np.random.normal(
                last_window["cancel_rate"].mean(),
                last_window["cancel_rate"].std()
            ),
            0,
            1
        ),

        "day_of_week": new_date.weekday(),
        "day_of_year": new_date.dayofyear,

        "lag1_delay": last_window["mean_delay"].iloc[-1],
        "lag2_delay": last_window["mean_delay"].iloc[-2],
        "lag3_delay": last_window["mean_delay"].iloc[-3],
        "lag7_delay": last_window["mean_delay"].iloc[0]
    }

    # Predict
    pred_delay = model.predict(
        pd.DataFrame([record_features])[FEATURES]
    )[0]

    record_features["prediction"] = pred_delay
    record_features["fl_date"] = new_date

    future_records.append(record_features)

    # Update lags with predicted value
    new_row = {
        "mean_delay": pred_delay,
        "std_delay": record_features["std_delay"],
        "flight_count": record_features["flight_count"],
        "cancel_rate": record_features["cancel_rate"]
    }

    last_window = pd.concat(
        [
            last_window.iloc[1:],
            pd.DataFrame([new_row])
        ],
        ignore_index=True
    )

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
plt.savefig("plots/lag/daily_delay_forecast.png")

print("\n===== NEXT 30 DAYS: PREDICTED DAILY ARRIVAL DELAYS =====")
print(future_df[["fl_date", "prediction"]])

DELAY_THRESHOLD = 1

# Convert regression outputs → classification labels
y_test_class = (y_test > DELAY_THRESHOLD).astype(int)
pred_test_class = (pred_test > DELAY_THRESHOLD).astype(int)

# Compute metrics
precision = precision_score(y_test_class, pred_test_class)
recall    = recall_score(y_test_class, pred_test_class)
f1        = f1_score(y_test_class, pred_test_class)

print("\n===== CLASSIFICATION METRICS (Delay > 1 min) =====")
print(f"Precision : {precision:.3f}")
print(f"Recall    : {recall:.3f}")
print(f"F1 Score  : {f1:.3f}")

# Confusion Matrix
cm = confusion_matrix(y_test_class, pred_test_class)

print("\n===== CONFUSION MATRIX =====")
print("Rows: Actual   | Columns: Predicted")
print("[ TN   FP ]")
print("[ FN   TP ]")
print(cm)

roc_auc = roc_auc_score(y_test_class, pred_test)

print("\n===== ROC-AUC =====")
print(f"AUC Score: {roc_auc:.3f}")

# Plot ROC Curve
fpr, tpr, thresholds = roc_curve(y_test_class, pred_test)

plt.figure(figsize=(7,6))
plt.plot(fpr, tpr)

plt.plot([0,1],[1,0], linestyle="--", color="gray", label="Random")

plt.xlabel("True Positive Rate (Recall)")
plt.ylabel("True Negative Rate")
plt.title("ROC Curve – Daily Delay Prediction")
plt.legend()
plt.tight_layout()
plt.savefig("plots/lag/daily_delay_roc.png")
