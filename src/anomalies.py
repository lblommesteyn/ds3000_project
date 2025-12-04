import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

sns.set(style="whitegrid")

df = pd.read_csv("../flight_data_2024.csv")

df['fl_date'] = pd.to_datetime(df['fl_date'], errors='coerce')

# Extract month for time analysis
df['month'] = df['fl_date'].dt.month

print("Loaded columns:")
print(df.columns.tolist())
print(df.head())

features = ['dep_delay', 'arr_delay', 'distance', 'air_time']

df_ml = df[features].dropna()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_ml)

model = IsolationForest(
    n_estimators=200,
    contamination=0.015,
    random_state=42
)

df_ml['anomaly_flag'] = model.fit_predict(X_scaled)

df_ml_indexed = df_ml.copy()
df_ml_indexed["row_id"] = df_ml_indexed.index

df_main = df.reset_index().merge(
    df_ml_indexed[["row_id", "anomaly_flag"]],
    left_on="index",
    right_on="row_id",
    how="left"
)

df_main["is_anomaly"] = df_main["anomaly_flag"] == -1

# Extract month
df_main["month"] = pd.to_datetime(df_main["fl_date"]).dt.month

# Filter anomaly rows
df_anom = df_main[df_main["is_anomaly"]].copy()

print(f"\n✅ Total Anomalies Detected: {len(df_anom)}")


# # AIRLINE ANOMALIES PER MONTH
airline_monthly = (
    df_anom.groupby(["month", "op_unique_carrier"])
    .size()
    .reset_index(name="anomaly_count")
)

plt.figure(figsize=(12,7))

for airline in airline_monthly["op_unique_carrier"].unique():
    subset = airline_monthly[airline_monthly["op_unique_carrier"] == airline]
    plt.plot(subset["month"], subset["anomaly_count"], label=airline)

plt.title("Monthly Flight Anomalies by Airline")
plt.xlabel("Month")
plt.ylabel("Anomaly Count")
plt.xticks(range(1,13))
plt.legend(ncol=4, fontsize=7)
plt.tight_layout()
plt.show()


# ROOT CAUSE CLASSIFICATION OF ANOMALIES

# Calculate speed for categorization
df_anom["speed_mph"] = 60 * df_anom["distance"] / df_anom["air_time"]

# Precompute quantile thresholds ONCE
upper_air = df["air_time"].quantile(0.99)
lower_air = df["air_time"].quantile(0.01)

# Create default category
df_anom["anomaly_cause"] = "Other / Data Quality"

# Apply rules vectorially (fast)

df_anom.loc[df_anom["speed_mph"] > 650, "anomaly_cause"] = "Unrealistically Fast"
df_anom.loc[df_anom["speed_mph"] < 150, "anomaly_cause"] = "Unusually Slow"

df_anom.loc[df_anom["arr_delay"] > 90, "anomaly_cause"] = "Extreme Arrival Delay"
df_anom.loc[df_anom["dep_delay"] > 90, "anomaly_cause"] = "Extreme Departure Delay"

df_anom.loc[df_anom["cancelled"] == 1, "anomaly_cause"] = "Cancellation"
df_anom.loc[df_anom["diverted"] == 1, "anomaly_cause"] = "Diversion"

df_anom.loc[df_anom["air_time"] > upper_air, "anomaly_cause"] = "Abnormally Long Air Time"
df_anom.loc[df_anom["air_time"] < lower_air, "anomaly_cause"] = "Abnormally Short Air Time"

cause_counts = (
    df_anom["anomaly_cause"]
    .value_counts()
    .reset_index()
)

cause_counts.columns = ["anomaly_cause", "count"]

print("\n=== ANOMALY CAUSES ===")
print(cause_counts)


# Count anomalies per departure location per month
origin_monthly = (
    df_anom
    .groupby(["month", "origin"])
    .size()
    .reset_index(name="anomaly_count")
)

# Keep top N busiest origins for legibility
TOP_ORIGINS = 10

top_origins = (
    origin_monthly
    .groupby("origin")["anomaly_count"]
    .sum()
    .sort_values(ascending=False)
    .head(TOP_ORIGINS)
    .index
)

origin_monthly = origin_monthly[origin_monthly["origin"].isin(top_origins)]

plt.figure(figsize=(12,7))

for origin in origin_monthly["origin"].unique():
    subset = origin_monthly[origin_monthly["origin"] == origin]
    plt.plot(subset["month"], subset["anomaly_count"], label=origin)

plt.title(f"Monthly Anomalies by Departure Airport (Top {TOP_ORIGINS})")
plt.xlabel("Month")
plt.ylabel("Anomaly Count")
plt.xticks(range(1,13))
plt.legend(fontsize=7, ncol=2)
plt.tight_layout()
plt.show()


df_valid = df[
    (df["distance"] > 0) &
    (df["air_time"] > 0)
].copy()

# REALISTIC LONGEST TRAVEL PATH - ONE ITINERARY

# Ensure valid distance/time rows
df_valid = df[
    (df["distance"] > 0) &
    (df["air_time"] > 0)
].copy()

df_valid["fl_date"] = pd.to_datetime(df_valid["fl_date"])


# LONGEST REALISTIC ITINERARY — SAME AIRLINE Flights exist on same day & carrier

airline_daily_paths = (
    df_valid
        .groupby(["op_unique_carrier", "fl_date"])["distance"]
        .sum()
        .reset_index()
)

max_airline_itinerary = (
    airline_daily_paths
        .groupby("op_unique_carrier")["distance"]
        .max()
        .sort_values(ascending=False)
        .reset_index()
)

top_airlines = max_airline_itinerary.head(10)
top_airlines["distance_kmiles"] = top_airlines["distance"] / 1000


print("\n LONGEST REALISTIC TRAVEL PATHS – SAME AIRLINE")
print(top_airlines)

# Plot
plt.figure(figsize=(10,6))
sns.barplot(
    data=top_airlines,
    x="distance_kmiles",
    y="op_unique_carrier"
)

plt.title("Top 10 Longest Single-Day Travel Paths — Same Airline")
plt.xlabel("Distance Traveled (Thousands of Miles)")
plt.ylabel("Airline")
plt.tight_layout()
plt.show()
