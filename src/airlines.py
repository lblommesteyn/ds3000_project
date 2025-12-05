import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import calendar
import os

from data_prep import load_df

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

sns.set(style="whitegrid")

df = load_df()

df['fl_date'] = pd.to_datetime(df['fl_date'], errors='coerce')

# Extract month for time analysis
df['month'] = df['fl_date'].dt.month

print("Loaded columns:")
print(df.columns.tolist())
print(df.head())


# SEASONALITY — FLIGHT VOLUME BY MONTH × AIRLINE
df_valid = df[
    (df["distance"] > 0) &
    (df["air_time"] > 0)
].copy()


monthly_airline = (
    df_valid
    .groupby(["month", "op_unique_carrier"])
    .size()
    .reset_index(name="flight_count")
)
monthly_airline["Airline"] = monthly_airline["op_unique_carrier"]

plt.figure(figsize=(10,6))
sns.lineplot(
    data=monthly_airline,
    x="month",
    y="flight_count",
    hue="Airline"
)

plt.title("Monthly Flight Volume Per Airline")
plt.xlabel("Month")
plt.ylabel("Number of Flights")
plt.xticks(
    ticks=range(1,13),
    labels=[calendar.month_abbr[m] for m in range(1,13)]
)
plt.tight_layout()
os.makedirs("plots/airlines", exist_ok=True)
plt.savefig("plots/airlines/monthly_airline.png")

# Delay probability heatmap (By Airline)
df_valid = df[
    (df["distance"] > 0) &
    (df["air_time"] > 0)
].copy()

df_valid["month"] = pd.to_datetime(df_valid["fl_date"]).dt.month
df_valid["speed_mph"] = 60 * df_valid["distance"] / df_valid["air_time"]

df_valid["delayed"] = df_valid["arr_delay"] > 15

route_delay = (
    df_valid
    .groupby(["origin", "dest"])
    .agg(
        flights=("delayed", "size"),
        delay_rate=("delayed", "mean")
    )
    .reset_index()
)

# Keep routes where traffic is significant
busy_routes = route_delay[route_delay["flights"] >= 50]

# Pivot heatmap grid
heatmap_data = (
    busy_routes
        .pivot(index="origin", columns="dest", values="delay_rate")
)

plt.figure(figsize=(14,10))
sns.heatmap(
    heatmap_data,
    cmap="Reds",
    vmin=0,
    vmax=1
)

plt.title("Probability of Delay by Route (Busy Routes Only)")
plt.xlabel("Destination")
plt.ylabel("Origin")
plt.tight_layout()
plt.savefig("plots/airlines/route_delay.png")

busy_routes = route_delay.query("flights >= 100")

# Only consider top 20 routes
# Get top airports by volume
TOP_AIRPORTS = 20

top_origins = (
    busy_routes.groupby("origin")["flights"]
    .sum()
    .sort_values(ascending=False)
    .head(TOP_AIRPORTS)
    .index
)

top_dests = (
    busy_routes.groupby("dest")["flights"]
    .sum()
    .sort_values(ascending=False)
    .head(TOP_AIRPORTS)
    .index
)

# Keep only busiest airports
filtered_routes = busy_routes[
    busy_routes["origin"].isin(top_origins) &
    busy_routes["dest"].isin(top_dests)
]

# Pivot matrix
heatmap_data = (
    filtered_routes
    .pivot(index="origin", columns="dest", values="delay_rate")
)

# Drop rows/columns that are still empty
heatmap_data = heatmap_data.dropna(how="all", axis=0)
heatmap_data = heatmap_data.dropna(how="all", axis=1)

plt.figure(figsize=(12,10))

sns.heatmap(
    heatmap_data,
    cmap="Reds",
    vmin=0, vmax=1,
    linewidths=0.5,
    square=True
)

plt.title("Probability of Delay by Route (Top Airports Only)")
plt.xlabel("Destination")
plt.ylabel("Origin")
plt.tight_layout()
plt.savefig("plots/airlines/route_delay_top_airports.png")


# Taxi time by Airline
# Build total taxi time = taxi_out + taxi_in (if present)
taxi_cols = [c for c in ['taxi_out', 'taxi_in'] if c in df.columns]
if taxi_cols:
    df['taxi_time'] = 0
    if 'taxi_out' in df.columns:
        df['taxi_time'] = df['taxi_time'] + df['taxi_out'].fillna(0)
    if 'taxi_in' in df.columns:
        df['taxi_time'] = df['taxi_time'] + df['taxi_in'].fillna(0)

    taxi_by_airline = (
        df.groupby('op_unique_carrier')['taxi_time']
          .mean()
          .reset_index()
          .sort_values('taxi_time', ascending=False)
    )

    print("\n=== AVERAGE TAXI TIME BY AIRLINE (minutes) ===")
    print(taxi_by_airline)

    plt.figure(figsize=(10,6))
    sns.barplot(
        data=taxi_by_airline,
        x='taxi_time',
        y='op_unique_carrier'
    )
    plt.title("Average Total Taxi Time by Airline")
    plt.xlabel("Average Taxi Time (min)")
    plt.ylabel("Airline")
    plt.tight_layout()
    plt.savefig("plots/airlines/taxi_time.png")


# Cancellation rate by Airline (Month and year)

# Year total per airline
cancel_year = (
    df.groupby('op_unique_carrier')
        .agg(
            flights=('cancelled', 'size'),
            cancelled=('cancelled', 'sum')
        )
        .reset_index()
)
cancel_year['cancel_rate'] = cancel_year['cancelled'] / cancel_year['flights']

print("\n=== YEAR TOTAL CANCELLATIONS BY AIRLINE ===")
print(cancel_year.sort_values('cancel_rate', ascending=False))

# Plot year total cancellation rate
plt.figure(figsize=(10,6))
sns.barplot(
    data=cancel_year.sort_values('cancel_rate', ascending=False),
    x='cancel_rate',
    y='op_unique_carrier'
)
plt.title("Cancellation Rate by Airline (Year Total)")
plt.xlabel("Cancellation Rate")
plt.ylabel("Airline")
plt.tight_layout()
plt.savefig("plots/airlines/cancel_year.png")

# Per month per airline
cancel_month = (
    df.groupby(['op_unique_carrier', 'month'])
        .agg(
            flights=('cancelled', 'size'),
            cancelled=('cancelled', 'sum')
        )
        .reset_index()
)
cancel_month['cancel_rate'] = cancel_month['cancelled'] / cancel_month['flights']
cancel_month["Airlines"] = cancel_month["op_unique_carrier"]

print("\n=== MONTHLY CANCELLATION STATS (FIRST FEW ROWS) ===")
print(cancel_month.head())

# Optional graph: monthly cancellation rate by airline (might be busy)
plt.figure(figsize=(12,6))
sns.lineplot(
    data=cancel_month,
    x='month',
    y='cancel_rate',
    hue='Airlines',
    marker='o'
)
plt.title("Monthly Cancellation Rate by Airline")
plt.xlabel("Month")
plt.ylabel("Cancellation Rate")
plt.xticks(
    ticks=range(1,13),
    labels=[calendar.month_abbr[m] for m in range(1,13)]
)
plt.tight_layout()
plt.savefig("plots/airlines/cancel_month.png")


# Flights delayed for security concerns (By month)
# DOT data usually has a 'security_delay' column (minutes).
security_col_candidates = [c for c in df.columns if 'security' in c.lower() and 'delay' in c.lower()]
security_col = security_col_candidates[0] if security_col_candidates else None

# A flight is "delayed for security" if security_delay > 0
df['security_delayed'] = df[security_col].fillna(0) > 0

security_month = (
    df.groupby('month')['security_delayed']
        .sum()
        .reset_index(name='num_security_delayed_flights')
)

print("\n=== FLIGHTS DELAYED FOR SECURITY REASONS BY MONTH ===")
print(security_month)

plt.figure(figsize=(10,6))
sns.barplot(
    data=security_month,
    x='month',
    y='num_security_delayed_flights'
)
plt.title("Flights Delayed for Security Concerns by Month")
plt.xlabel("Month")
plt.ylabel("Number of Flights")
plt.xticks(
    ticks=range(1,13),
    labels=[calendar.month_abbr[m] for m in range(1,13)]
)
plt.savefig("plots/airlines/security_delay.png")
plt.tight_layout()
plt.show()

# Most reliable airlines
df["on_time"] = (
    (df["arr_delay"] <= 15) &
    (df["cancelled"] == 0) &
    (df["diverted"] == 0)
)

reliability = (
    df.groupby("op_unique_carrier")
    .agg(
        total_flights=("on_time","count"),
        on_time_rate=("on_time","mean"),
        cancel_rate=("cancelled","mean"),
        divert_rate=("diverted","mean")
    )
    .sort_values("on_time_rate", ascending=False)
    .reset_index()
)

print("\n=== MOST RELIABLE AIRLINES ===")
print(reliability.head(10))

plt.figure(figsize=(12,6))
sns.barplot(x="op_unique_carrier", y="on_time_rate", data=reliability)
plt.xticks(rotation=45)
plt.title("On-Time Performance by Airline")
plt.ylim(0,1)
plt.xlabel("Airline")
plt.ylabel("On Time %")
plt.show()

monthly_volume = (
    df.groupby("month")
      .size()
      .reset_index(name="flight_count")
)

print("\n=== MONTHLY FLIGHT COUNTS ===")
print(monthly_volume)

plt.figure(figsize=(10,6))
sns.lineplot(data=monthly_volume, x="month", y="flight_count", marker='o')
plt.title("Monthly Air Traffic Volume (Single-Year Dataset)")
plt.xlabel("Month")
plt.ylabel("Flights")
plt.xticks(
    ticks=range(1,13),
    labels=[calendar.month_abbr[m] for m in range(1,13)]
)
plt.show()

