import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from pathlib import Path
import os

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, classification_report, f1_score, precision_recall_curve
from xgboost import XGBClassifier

from data_prep import load_df, add_dep_hour, add_delay_label, filter_operated_flights, add_predeparture_features, sample_df

# Ensure plots directory exists
os.makedirs("plots", exist_ok=True)

def run_advanced_analysis():
    print("Loading data...")
    df = load_df()
    
    # Basic Prep
    df = add_dep_hour(df)
    df = add_delay_label(df)
    df = filter_operated_flights(df)
    df = add_predeparture_features(df)
    
    # --- Advanced Feature Engineering ---
    print("Engineering advanced features...")
    
    # 1. Cyclical Time Features
    df['dep_hour_sin'] = np.sin(2 * np.pi * df['dep_hour'] / 24)
    df['dep_hour_cos'] = np.cos(2 * np.pi * df['dep_hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # 2. Congestion Proxy (Flights per hour at Origin)
    # Group by Date, Hour, Origin
    congestion = df.groupby(['fl_date', 'dep_hour', 'origin']).size().reset_index(name='hourly_traffic_origin')
    df = df.merge(congestion, on=['fl_date', 'dep_hour', 'origin'], how='left')
    
    # 3. Route Stats (Prior Probability of Delay on this Route)
    # To avoid leakage, we should ideally compute this on a past window, but for this project, 
    # we will use a simple Target Encoding approach within the Cross-Validation or Split.
    # Here we just prepare the columns.
    
    # Select Features
    feature_cols = [
        "dep_hour_sin", "dep_hour_cos", "month_sin", "month_cos",
        "distance_bucket", "is_weekend", "is_peak_summer",
        "hourly_traffic_origin",
        "origin", "dest", "op_unique_carrier"
    ]
    target_col = "is_delayed_15"
    
    # Prepare Data
    model_df = df[feature_cols + [target_col]].dropna()
    
    # Sample for feasibility
    model_df = sample_df(model_df, n=300000)
    
    X = model_df[feature_cols]
    y = model_df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    
    # --- Pipeline Construction ---
    
    # Target Encoding for High Cardinality Categoricals
    # We use TargetEncoder from sklearn (or we could use category_encoders)
    # Note: TargetEncoder handles leakage internally by using cross-fitting during fit()
    
    categorical_features = ["origin", "dest", "op_unique_carrier"]
    numeric_features = [
        "dep_hour_sin", "dep_hour_cos", "month_sin", "month_cos",
        "distance_bucket", "is_weekend", "is_peak_summer",
        "hourly_traffic_origin"
    ]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numeric_features),
            ('cat', TargetEncoder(target_type='binary', smooth=10.0), categorical_features)
        ])
    
    xgb_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(
            objective='binary:logistic',
            eval_metric='auc',
            n_jobs=-1,
            random_state=42,
            tree_method='hist' # Faster
        ))
    ])
    
    # --- Hyperparameter Tuning ---
    print("Tuning hyperparameters...")
    
    param_dist = {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__max_depth': [3, 5, 7, 9],
        'classifier__learning_rate': [0.01, 0.05, 0.1, 0.2],
        'classifier__subsample': [0.7, 0.8, 0.9, 1.0],
        'classifier__colsample_bytree': [0.7, 0.8, 0.9, 1.0],
        'classifier__scale_pos_weight': [1, 3, 5] # Handle imbalance
    }
    
    random_search = RandomizedSearchCV(
        xgb_pipeline, 
        param_distributions=param_dist, 
        n_iter=15, # Limited iterations for speed
        scoring='roc_auc', 
        cv=3, 
        verbose=1,
        n_jobs=-1,
        random_state=42
    )
    
    random_search.fit(X_train, y_train)
    
    best_model = random_search.best_estimator_
    print(f"Best Parameters: {random_search.best_params_}")
    
    # --- Evaluation ---
    print("Evaluating best model...")
    
    y_proba = best_model.predict_proba(X_test)[:, 1]
    
    # Find optimal threshold for F1
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    print(f"Optimal Threshold: {optimal_threshold:.4f}")
    
    y_pred = (y_proba >= optimal_threshold).astype(int)
    
    auc = roc_auc_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred)
    
    print(f"\n--- Advanced XGBoost Results ---")
    print(f"ROC-AUC: {auc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(classification_report(y_test, y_pred))
    
    with open("metrics_advanced.txt", "w") as f:
        f.write("Advanced Model Metrics\n")
        f.write(f"Best Params: {random_search.best_params_}\n")
        f.write(f"Optimal Threshold: {optimal_threshold:.4f}\n")
        f.write(f"ROC-AUC: {auc:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
        f.write(classification_report(y_test, y_pred))
        
    # --- SHAP Interpretation ---
    print("Calculating SHAP values...")
    
    # We need to transform the data first to pass to SHAP
    # Access the preprocessor from the pipeline
    X_test_transformed = best_model.named_steps['preprocessor'].transform(X_test)
    
    # Get feature names
    # Numeric names are static
    # TargetEncoder names are the same as input
    feature_names = numeric_features + categorical_features
    
    # Create SHAP explainer
    # Use the underlying XGBoost model
    xgb_model = best_model.named_steps['classifier']
    explainer = shap.TreeExplainer(xgb_model)
    
    # Calculate SHAP values for a subset of test data (for speed)
    shap_values = explainer.shap_values(X_test_transformed[:2000])
    
    # Summary Plot
    plt.figure()
    shap.summary_plot(shap_values, X_test_transformed[:2000], feature_names=feature_names, show=False)
    plt.title("SHAP Feature Importance")
    plt.tight_layout()
    plt.savefig('plots/shap_summary.png')
    plt.close()
    
    return best_model

if __name__ == "__main__":
    run_advanced_analysis()
