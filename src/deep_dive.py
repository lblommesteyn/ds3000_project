import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from pathlib import Path
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from sklearn.calibration import calibration_curve
from xgboost import XGBClassifier

from data_prep import load_df, add_dep_hour, add_delay_label, filter_operated_flights, add_predeparture_features, sample_df

# Ensure plots directory exists
os.makedirs("plots", exist_ok=True)

# Set style for "awesome" plots
sns.set_theme(style="whitegrid", palette="viridis")

def run_deep_dive():
    print("Loading data...")
    df = load_df()
    
    # Basic Prep
    df = add_dep_hour(df)
    df = add_delay_label(df)
    df = filter_operated_flights(df)
    df = add_predeparture_features(df)
    
    # Sample for speed (but large enough for good graphs)
    df_vis = sample_df(df, n=500000)
    
    print("Generating Deep Dive Visualizations...")
    
    # 1. "The Wall of Delay" Heatmap (Day vs Hour)
    plot_wall_of_delay(df_vis)
    
    # 2. Route Network Graph
    plot_route_network(df_vis)
    
    # 3. Delay Cause Breakdown
    plot_delay_causes(df_vis)
    
    # 4. Model Comparison (Re-training on sample for curves)
    compare_models(df_vis)

def plot_wall_of_delay(df):
    print("Plotting Wall of Delay...")
    # Pivot table: Day of Week vs Hour
    pivot = df.groupby(['day_of_week', 'dep_hour'])['is_delayed_15'].mean().unstack()
    pivot = pivot.astype(float) # Ensure float for heatmap
    
    plt.figure(figsize=(14, 8))
    sns.heatmap(pivot, cmap='magma', annot=False, cbar_kws={'label': 'Probability of Delay > 15m'})
    plt.title('The Wall of Delay: When does the system melt down?', fontsize=16)
    plt.xlabel('Hour of Day', fontsize=12)
    plt.ylabel('Day of Week (1=Mon, 7=Sun)', fontsize=12)
    plt.tight_layout()
    plt.savefig('plots/wall_of_delay.png', dpi=300)
    plt.close()

def plot_route_network(df):
    print("Plotting Route Network...")
    # Top 30 routes by volume (reduced from 50 for clarity)
    routes = df.groupby(['origin', 'dest']).agg(
        volume=('fl_date', 'count'),
        avg_delay=('arr_delay', 'mean')
    ).reset_index()
    
    top_routes = routes.sort_values('volume', ascending=False).head(30)
    
    G = nx.from_pandas_edgelist(top_routes, 'origin', 'dest', ['volume', 'avg_delay'])
    
    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(G, k=0.3, iterations=50) # Increased k for more spacing
    
    # Node size by degree (hub importance)
    d = dict(G.degree)
    node_sizes = [v * 100 for v in d.values()]
    
    # Edge color by delay (Red = Bad, Green = Good)
    edges, weights = zip(*nx.get_edge_attributes(G, 'avg_delay').items())
    
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='skyblue', alpha=0.8)
    nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=weights, edge_cmap=plt.cm.RdYlGn_r, width=2, alpha=0.7)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
    
    plt.title('Top 50 US Air Routes: Colored by Average Delay', fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('plots/route_network.png', dpi=300)
    plt.close()

def plot_delay_causes(df):
    print("Plotting Delay Causes...")
    # Filter for delayed flights only
    delayed = df[df['arr_delay'] > 15].copy()
    
    # Columns for causes
    causes = ['carrier_delay', 'weather_delay', 'nas_delay', 'security_delay', 'late_aircraft_delay']
    
    # Normalize to get % contribution per flight
    delayed[causes] = delayed[causes].fillna(0)
    delayed['total_cause'] = delayed[causes].sum(axis=1)
    # Avoid division by zero
    delayed = delayed[delayed['total_cause'] > 0]
    
    # Top 10 Carriers
    top_carriers = delayed['op_unique_carrier'].value_counts().head(10).index
    delayed_top = delayed[delayed['op_unique_carrier'].isin(top_carriers)]
    
    # Aggregate
    cause_breakdown = delayed_top.groupby('op_unique_carrier')[causes].mean()
    # Normalize to 100% for stacked bar
    cause_breakdown = cause_breakdown.div(cause_breakdown.sum(axis=1), axis=0) * 100
    
    cause_breakdown.plot(kind='bar', stacked=True, figsize=(12, 8), colormap='viridis')
    plt.title('Anatomy of a Delay: What causes delays for major carriers?', fontsize=16)
    plt.ylabel('Percentage of Delay Minutes', fontsize=12)
    plt.xlabel('Carrier', fontsize=12)
    plt.legend(title='Delay Cause', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('plots/delay_causes_stacked.png', dpi=300)
    plt.close()

def compare_models(df):
    print("Comparing Models...")
    
    # Feature Engineering (Simplified for comparison speed)
    df['dep_hour_sin'] = np.sin(2 * np.pi * df['dep_hour'] / 24)
    df['dep_hour_cos'] = np.cos(2 * np.pi * df['dep_hour'] / 24)
    
    feature_cols = [
        "dep_hour_sin", "dep_hour_cos", "distance_bucket", "is_weekend",
        "origin", "dest", "op_unique_carrier"
    ]
    target_col = "is_delayed_15"
    
    model_df = df[feature_cols + [target_col]].dropna()
    X = model_df[feature_cols]
    y = model_df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    
    # Preprocessor
    preprocessor = ColumnTransformer([
        ('num', 'passthrough', ["dep_hour_sin", "dep_hour_cos", "distance_bucket", "is_weekend"]),
        ('cat', TargetEncoder(target_type='binary', smooth=10.0), ["origin", "dest", "op_unique_carrier"])
    ])
    
    models = {
        "Logistic Regression": Pipeline([('prep', preprocessor), ('clf', LogisticRegression(max_iter=500))]),
        "Random Forest": Pipeline([('prep', preprocessor), ('clf', RandomForestClassifier(n_estimators=50, max_depth=10, n_jobs=-1))]),
        "XGBoost": Pipeline([('prep', preprocessor), ('clf', XGBClassifier(n_estimators=100, max_depth=6, n_jobs=-1, eval_metric='auc'))])
    }
    
    plt.figure(figsize=(10, 8))
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
        
    plt.plot([0, 1], [0, 1], 'k--', label='Random Chance')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison', fontsize=16)
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig('plots/model_comparison_roc.png', dpi=300)
    plt.close()
    
    # Calibration Plot
    plt.figure(figsize=(10, 8))
    for name, model in models.items():
        y_proba = model.predict_proba(X_test)[:, 1]
        prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=10)
        plt.plot(prob_pred, prob_true, marker='o', label=name)
        
    plt.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Calibration Plot (Reliability Diagram)', fontsize=16)
    plt.legend()
    plt.grid(True)
    plt.savefig('plots/model_calibration.png', dpi=300)
    plt.close()

if __name__ == "__main__":
    run_deep_dive()
