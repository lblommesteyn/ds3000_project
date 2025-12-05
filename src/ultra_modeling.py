import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, classification_report, f1_score, confusion_matrix
from xgboost import XGBClassifier

from data_prep import load_df, add_dep_hour, add_delay_label, filter_operated_flights, add_predeparture_features, sample_df

# Ensure plots directory exists
os.makedirs("plots", exist_ok=True)
sns.set_theme(style="whitegrid", palette="viridis")

def run_ultra_modeling():
    print("Loading data for Ultra Modeling...")
    df = load_df()
    
    # Basic Prep
    df = add_dep_hour(df)
    df = add_delay_label(df)
    df = filter_operated_flights(df)
    df = add_predeparture_features(df)
    
    # Advanced Features
    df['dep_hour_sin'] = np.sin(2 * np.pi * df['dep_hour'] / 24)
    df['dep_hour_cos'] = np.cos(2 * np.pi * df['dep_hour'] / 24)
    
    # Sample for feasibility
    model_df = sample_df(df, n=200000)
    
    feature_cols = [
        "dep_hour_sin", "dep_hour_cos", "distance_bucket", "is_weekend",
        "origin", "dest", "op_unique_carrier"
    ]
    target_col = "is_delayed_15"
    
    model_df = model_df[feature_cols + [target_col]].dropna()
    X = model_df[feature_cols]
    y = model_df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    
    print("Optimizing HistGradientBoosting (LightGBM equivalent)...")
    best_hgb_params = optimize_hgb(X_train, y_train)
    
    print("Training Final Stacked Model...")
    train_stacking_ensemble(X_train, y_train, X_test, y_test, best_hgb_params)

def optimize_hgb(X, y):
    # Preprocessing
    categorical_features = ["origin", "dest", "op_unique_carrier"]
    numeric_features = ["dep_hour_sin", "dep_hour_cos", "distance_bucket", "is_weekend"]
    
    preprocessor = ColumnTransformer([
        ('num', 'passthrough', numeric_features),
        ('cat', TargetEncoder(target_type='binary', smooth=10.0), categorical_features)
    ])
    
    pipeline = Pipeline([
        ('prep', preprocessor),
        ('clf', HistGradientBoostingClassifier(random_state=42, scoring='roc_auc'))
    ])
    
    param_dist = {
        'clf__learning_rate': [0.01, 0.05, 0.1, 0.2],
        'clf__max_iter': [100, 200, 300],
        'clf__max_leaf_nodes': [31, 63, 127],
        'clf__l2_regularization': [0, 1, 10]
    }
    
    search = RandomizedSearchCV(
        pipeline, 
        param_distributions=param_dist, 
        n_iter=10, 
        scoring='roc_auc', 
        cv=3, 
        verbose=1, 
        n_jobs=-1,
        random_state=42
    )
    
    search.fit(X, y)
    print(f"Best HGB Params: {search.best_params_}")
    return search.best_params_

def train_stacking_ensemble(X_train, y_train, X_test, y_test, hgb_params):
    # Common Preprocessor
    categorical_features = ["origin", "dest", "op_unique_carrier"]
    numeric_features = ["dep_hour_sin", "dep_hour_cos", "distance_bucket", "is_weekend"]
    
    preprocessor = ColumnTransformer([
        ('num', 'passthrough', numeric_features),
        ('cat', TargetEncoder(target_type='binary', smooth=10.0), categorical_features)
    ])
    
    # Extract clf params from pipeline params
    clean_hgb_params = {k.replace('clf__', ''): v for k, v in hgb_params.items()}
    
    estimators = [
        ('hgb', Pipeline([
            ('prep', preprocessor),
            ('clf', HistGradientBoostingClassifier(**clean_hgb_params, random_state=42))
        ])),
        ('xgb', Pipeline([
            ('prep', preprocessor),
            ('clf', XGBClassifier(n_estimators=200, max_depth=6, eval_metric='auc', n_jobs=-1))
        ])),
        ('rf', Pipeline([
            ('prep', preprocessor),
            ('clf', RandomForestClassifier(n_estimators=100, max_depth=10, n_jobs=-1))
        ]))
    ]
    
    # Stacking
    stacking_clf = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(),
        cv=3,
        n_jobs=-1
    )
    
    print("Training Stacking Classifier...")
    stacking_clf.fit(X_train, y_train)
    
    # Evaluation
    y_proba = stacking_clf.predict_proba(X_test)[:, 1]
    y_pred = (y_proba > 0.5).astype(int)
    
    auc = roc_auc_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred)
    
    print("\n--- Ultra Stacking Results ---")
    print(f"ROC-AUC: {auc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(classification_report(y_test, y_pred))
    
    # Cost Analysis
    # Cost = FN * 1000 + FP * 100
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    total_cost = (fn * 1000) + (fp * 100)
    print(f"\nEstimated Business Cost: ${total_cost:,}")
    
    with open("metrics_ultra.txt", "w") as f:
        f.write("Ultra Model Metrics (Stacking)\n")
        f.write(f"ROC-AUC: {auc:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
        f.write(f"Business Cost: ${total_cost:,}\n")
        f.write(classification_report(y_test, y_pred))

if __name__ == "__main__":
    run_ultra_modeling()
