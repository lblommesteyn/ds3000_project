import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import calendar

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

# Longest possible travel path same day.

# Ensure valid distance/time rows

df_valid = df[
    (df["distance"] > 0) &
    (df["air_time"] > 0)
].copy()

df_valid["fl_date"] = pd.to_datetime(df_valid["fl_date"])

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


# Distance vs Delay (By Airline)

plt.figure(figsize=(10,6))

sns.scatterplot(
    data=df_valid.sample(min(10000, len(df_valid))),
    x="distance",
    y="arr_delay",
    hue="op_unique_carrier",
    legend=False,
    alpha=0.3
)

plt.title("Arrival Delay vs Flight Distance")
plt.xlabel("Distance (miles)")
plt.ylabel("Arrival Delay (minutes)")
plt.tight_layout()
plt.show()

# Speed destribution by Airline
df_valid["speed_mph"] = 60 * df_valid["distance"] / df_valid["air_time"]

plt.figure(figsize=(10,6))

sns.boxplot(
    data=df_valid,
    x="op_unique_carrier",
    y="speed_mph"
)

plt.title("Flight Speed Distribution by Airline")
plt.xlabel("Airline")
plt.ylabel("Speed (mph)")
plt.ylim(200, 650)
plt.tight_layout()
plt.show()


# Most flown routes
# Create a route label like "JFK-LAX"
df['route'] = df['origin'].astype(str) + '-' + df['dest'].astype(str)

# ---- Year total: top 20 routes 
TOP_N_ROUTES = 20

route_year = (
    df.groupby('route')
      .size()
      .reset_index(name='flight_count')
      .sort_values('flight_count', ascending=False)
)

top_routes = route_year.head(TOP_N_ROUTES)

print("\n=== TOP ROUTES (YEAR TOTAL) ===")
print(top_routes)

plt.figure(figsize=(10,6))
sns.barplot(data=top_routes, x='flight_count', y='route')
plt.title(f"Top {TOP_N_ROUTES} Most-Flown Routes (Year Total)")
plt.xlabel("Number of Flights")
plt.ylabel("Route (Origin–Destination)")
plt.tight_layout()
plt.show()

# Per month: same top routes only
route_month = (
    df[df['route'].isin(top_routes['route'])]
      .groupby(['month', 'route'])
      .size()
      .reset_index(name='flight_count')
)

plt.figure(figsize=(12,6))
sns.lineplot(
    data=route_month,
    x='month',
    y='flight_count',
    hue='route',
    marker='o'
)
plt.title(f"Monthly Flight Counts for Top {TOP_N_ROUTES} Routes")
plt.xlabel("Month")
plt.ylabel("Number of Flights")
plt.xticks(
    ticks=range(1,13),
    labels=[calendar.month_abbr[m] for m in range(1,13)]
)
plt.tight_layout()
plt.show()
