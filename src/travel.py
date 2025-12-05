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


# Ensure valid distance/time rows
df_valid = df[
    (df["distance"] > 0) &
    (df["air_time"] > 0)
].copy()

df_valid["fl_date"] = pd.to_datetime(df_valid["fl_date"])

# Most flown routes
df["route"] = df["origin"].astype(str) + "-" + df["dest"].astype(str)

# Count route usage
route_usage = (
    df.groupby(["origin", "dest"])
      .size()
      .reset_index(name="flight_count")
)

# Keep busy routes only
MIN_FLIGHTS = 100  
busy_routes = route_usage[route_usage["flight_count"] >= MIN_FLIGHTS]

# Top airports for manageable size
TOP_AIRPORTS = 20

top_origins = (
    busy_routes.groupby("origin")["flight_count"]
        .sum()
        .sort_values(ascending=False)
        .head(TOP_AIRPORTS)
        .index
)

top_dests = (
    busy_routes.groupby("dest")["flight_count"]
        .sum()
        .sort_values(ascending=False)
        .head(TOP_AIRPORTS)
        .index
)

filtered_routes = busy_routes[
    busy_routes["origin"].isin(top_origins) &
    busy_routes["dest"].isin(top_dests)
]

# Pivot matrix for heatmap
route_heatmap = (
    filtered_routes
        .pivot(index="origin", columns="dest", values="flight_count")
        .fillna(0)
)

plt.figure(figsize=(12,10))

# Most travelled to cities 
origin_counts = (
    df.groupby("origin")
      .size()
)

dest_counts = (
    df.groupby("dest")
      .size()
)

airport_activity = (
    pd.concat([origin_counts, dest_counts], axis=1)
      .fillna(0)
)

airport_activity.columns = ["departures", "arrivals"]

airport_activity["total_activity"] = (
    airport_activity["departures"] + airport_activity["arrivals"]
)

# Take top airports
TOP_CITIES = 25

airport_activity = (
    airport_activity
        .sort_values("total_activity", ascending=False)
        .head(TOP_CITIES)
)

# Prepare heatmap grid
city_heatmap = airport_activity[["departures", "arrivals"]]

plt.figure(figsize=(8,10))
sns.heatmap(
    route_heatmap,
    cmap="Blues",
    linewidths=0.5,
    square=True
)

plt.title("Most-Used Flight Routes (Top Airports Only)")
plt.xlabel("Destination Airport")
plt.ylabel("Origin Airport")
plt.tight_layout()
os.makedirs("plots/travel", exist_ok=True)
plt.savefig("plots/travel/route_heatmap.png")

sns.heatmap(
    city_heatmap,
    annot=True,
    fmt=".0f",
    cmap="Greens"
)

plt.title("Most Active Cities by Flight Traffic")
plt.xlabel("Traffic Type")
plt.ylabel("Airport Code")
plt.tight_layout()
plt.savefig("plots/travel/city_heatmap.png")

print("\n=== TOP BUSIEST ROUTES ===")
print(
    route_usage
        .sort_values("flight_count", ascending=False)
        .head(15)
)

print("\n=== MOST ACTIVE CITIES ===")
print(
    airport_activity[["total_activity"]]
)


# Longest travel path using same airline same day
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
plt.savefig("plots/travel/longest_paths.png")


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
plt.savefig("plots/travel/delay_vs_distance.png")

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
plt.savefig("plots/travel/speed_distribution.png")


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
plt.savefig("plots/travel/top_routes_year.png")

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
plt.savefig("plots/travel/top_routes_month.png")
