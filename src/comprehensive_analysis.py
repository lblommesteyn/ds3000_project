import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, f1_score
from xgboost import XGBClassifier

# Import from existing data_prep
from data_prep import load_df, add_dep_hour, add_delay_label, filter_operated_flights, add_predeparture_features, sample_df

# Ensure plots directory exists
os.makedirs("plots", exist_ok=True)

def run_analysis():
    print("Loading data...")
    df = load_df()
    
    print("Initial data shape:", df.shape)
    
    # Basic Cleaning & Prep
    df = add_dep_hour(df)
    df = add_delay_label(df)
    df = filter_operated_flights(df)
    df = add_predeparture_features(df)
    
    print("Data shape after prep:", df.shape)
    
    perform_eda(df)
    
    return df

def perform_eda(df):
    print("Performing EDA...")
    
    # 1. Delay Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df['arr_delay'], bins=100, kde=True)
    plt.title('Distribution of Arrival Delays')
    plt.xlim(-60, 180) # Focus on reasonable range
    plt.xlabel('Arrival Delay (minutes)')
    plt.savefig('plots/delay_distribution.png')
    plt.close()
    
    # 2. Delay Rate by Hour
    delay_by_hour = df.groupby('dep_hour')['is_delayed_15'].mean()
    plt.figure(figsize=(10, 6))
    delay_by_hour.plot(marker='o')
    plt.title('Proportion of Flights Delayed > 15 min by Hour')
    plt.ylabel('Delay Rate')
    plt.xlabel('Hour of Day')
    plt.grid(True)
    plt.savefig('plots/delay_by_hour.png')
    plt.close()
    
    # 3. Delay Rate by Month
    delay_by_month = df.groupby('month')['is_delayed_15'].mean()
    plt.figure(figsize=(10, 6))
    delay_by_month.plot(kind='bar')
    plt.title('Proportion of Flights Delayed > 15 min by Month')
    plt.ylabel('Delay Rate')
    plt.xlabel('Month')
    plt.savefig('plots/delay_by_month.png')
    plt.close()
    
    # 4. Delay Rate by Carrier (Top 10)
    top_carriers = df['op_unique_carrier'].value_counts().head(10).index
    delay_by_carrier = df[df['op_unique_carrier'].isin(top_carriers)].groupby('op_unique_carrier')['is_delayed_15'].mean().sort_values()
    
    plt.figure(figsize=(12, 6))
    delay_by_carrier.plot(kind='barh')
    plt.title('Delay Rate by Carrier (Top 10 by Volume)')
    plt.xlabel('Delay Rate')
    plt.savefig('plots/delay_by_carrier.png')
    plt.close()

    # 5. Correlation Heatmap (Numeric features)
    numeric_cols = ['arr_delay', 'dep_hour', 'distance', 'is_delayed_15']
    corr = df[numeric_cols].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap')
    plt.savefig('plots/correlation_heatmap.png')
    plt.close()

def train_models(df):
    print("Training models...")
    
    # Feature Selection
    feature_cols = [
        "dep_hour", "month", "day_of_week", "distance_bucket", 
        "is_weekend", "is_peak_summer", 
        "origin", "dest", "op_unique_carrier"
    ]
    target_col = "is_delayed_15"
    
    # Prepare Data
    model_df = df[feature_cols + [target_col]].dropna()
    
    # Sampling for speed if dataset is huge (optional, but good for dev)
    model_df = sample_df(model_df, n=500000) 
    
    X = model_df[feature_cols]
    y = model_df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    
    # Preprocessing
    categorical_features = ["origin", "dest", "op_unique_carrier"]
    numeric_features = ["dep_hour", "month", "day_of_week", "distance_bucket", "is_weekend", "is_peak_summer"]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    # Models
    models = {
        "Logistic Regression": Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(max_iter=1000, random_state=42))
        ]),
        "Random Forest": Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(n_estimators=100, max_depth=10, n_jobs=-1, random_state=42))
        ]),
        "XGBoost": Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', XGBClassifier(
                n_estimators=200, 
                max_depth=7, 
                learning_rate=0.05, 
                eval_metric='auc',
                n_jobs=-1,
                random_state=42
            ))
        ])
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        auc = roc_auc_score(y_test, y_proba)
        f1 = f1_score(y_test, y_pred)
        
        print(f"\n--- {name} Results ---")
        print(f"ROC-AUC: {auc:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(classification_report(y_test, y_pred))
        
        with open("metrics.txt", "a") as f:
            f.write(f"\n--- {name} Results ---\n")
            f.write(f"ROC-AUC: {auc:.4f}\n")
            f.write(f"F1 Score: {f1:.4f}\n")
            f.write(classification_report(y_test, y_pred))
            f.write("\n")
        
        results[name] = {
            "model": model,
            "auc": auc,
            "f1": f1,
            "y_test": y_test,
            "y_proba": y_proba
        }
        
    return results

if __name__ == "__main__":
    # Clear metrics file
    with open("metrics.txt", "w") as f:
        f.write("Model Metrics\n")
        
    df = run_analysis()
    results = train_models(df)
