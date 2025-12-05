import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

from data_prep import load_df, hhmm_to_minutes

# Ensure plots directory exists
os.makedirs("plots", exist_ok=True)
sns.set_theme(style="whitegrid", palette="muted")

def run_ultra_eda():
    print("Loading data for Ultra EDA...")
    df = load_df()
    
    # Add dep_hour and delay label
    from data_prep import add_dep_hour, add_delay_label
    df = add_dep_hour(df)
    df = add_delay_label(df)
    
    # Convert times for plotting
    df['dep_time_min'] = df['dep_time'].map(hhmm_to_minutes)
    df['arr_time_min'] = df['arr_time'].map(hhmm_to_minutes)
    df['crs_dep_time_min'] = df['crs_dep_time'].map(hhmm_to_minutes)
    df['crs_arr_time_min'] = df['crs_arr_time'].map(hhmm_to_minutes)
    
    print("Generating Ultra EDA Visualizations...")
    
    # 1. Hub Congestion (Heatmap)
    plot_hub_congestion(df)
    
    # 2. Airport Efficiency (Taxi Times)
    plot_airport_efficiency(df)
    
    # 3. Holiday Impact
    plot_holiday_impact(df)

def plot_hub_congestion(df):
    print("Plotting Hub Congestion...")
    # Top 5 Hubs
    top_hubs = df['origin'].value_counts().head(5).index
    
    # Filter
    hubs = df[df['origin'].isin(top_hubs)].copy()
    
    # Group by Hub and Hour
    congestion = hubs.groupby(['origin', 'dep_hour']).size().unstack()
    
    plt.figure(figsize=(12, 6))
    sns.heatmap(congestion, cmap='inferno', annot=False)
    plt.title('Hub Congestion: Flights per Hour at Top 5 Airports', fontsize=14)
    plt.xlabel('Hour of Day')
    plt.ylabel('Airport')
    plt.tight_layout()
    plt.savefig('plots/hub_congestion.png', dpi=300)
    plt.close()

def plot_airport_efficiency(df):
    print("Plotting Airport Efficiency...")
    # Top 20 Airports by volume
    top_airports = df['origin'].value_counts().head(20).index
    
    # Taxi Out analysis
    taxi_out = df[df['origin'].isin(top_airports)].groupby('origin')['taxi_out'].mean().sort_values()
    
    plt.figure(figsize=(12, 8))
    taxi_out.plot(kind='barh', color='orange')
    plt.title('Average Taxi-Out Time by Airport (Top 20)', fontsize=14)
    plt.xlabel('Minutes')
    plt.ylabel('Airport')
    plt.tight_layout()
    plt.savefig('plots/taxi_efficiency.png', dpi=300)
    plt.close()

def plot_holiday_impact(df):
    print("Plotting Holiday Impact...")
    # Define holiday windows
    df['fl_date'] = pd.to_datetime(df['fl_date'])
    
    # Thanksgiving 2024 (Nov 28) window
    thanksgiving_window = (df['fl_date'] >= '2024-11-20') & (df['fl_date'] <= '2024-12-03')
    
    if thanksgiving_window.sum() == 0:
        print("No Thanksgiving data available (dataset might be partial year).")
        return

    tg_data = df[thanksgiving_window].groupby('fl_date')['is_delayed_15'].mean()
    
    plt.figure(figsize=(12, 6))
    tg_data.plot(marker='o', color='purple')
    plt.title('Thanksgiving 2024 Delay Spike', fontsize=14)
    plt.ylabel('Delay Probability')
    plt.axvline(pd.Timestamp('2024-11-28'), color='red', linestyle='--', label='Thanksgiving Day')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('plots/holiday_impact.png', dpi=300)
    plt.close()

if __name__ == "__main__":
    run_ultra_eda()
