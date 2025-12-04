import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, classification_report
from xgboost import XGBClassifier

from data_prep import load_df, add_dep_hour, add_delay_label, filter_operated_flights, add_predeparture_features, sample_df

df = load_df()
df = add_dep_hour(df)
df = add_delay_label(df)
df = filter_operated_flights(df)
df = add_predeparture_features(df)

df_model = df.copy()

# only use valid features
feature_cols = [
    "dep_hour",
    "month",
    "day_of_week",
    "distance_bucket",
    "is_weekend",
    "is_peak_summer",
    "origin",
    "dest",
    "op_unique_carrier",
]

target_col = "is_delayed_15"

model_df_fe_clean = df_model[feature_cols + [target_col]].dropna()
model_df_fe_clean = sample_df(model_df_fe_clean)

X = model_df_fe_clean[feature_cols]
y = model_df_fe_clean[target_col]

print(f"Using {len(X):,} rows for *clean* XGBoost")

# train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size = 0.2,
    stratify = y,
    random_state = 42
)

categorical_features = ["origin", "dest", "op_unique_carrier"]
numeric_features = [
    "dep_hour",
    "month",
    "day_of_week",
    "distance_bucket",
    "is_weekend",
    "is_peak_summer",
]

preprocess_clean = ColumnTransformer(
    transformers = [
        ("cat", OneHotEncoder(handle_unknown = "ignore"), categorical_features),
        ("num", "passthrough", numeric_features),
    ]
)

neg, pos = np.bincount(y_train)
scale_pos_weight = neg / pos

xgb_clean = Pipeline(
    steps = [
        ("prep", preprocess_clean),
        ("model", XGBClassifier(
            tree_method = "hist",
            n_estimators = 350,
            max_depth = 7,
            learning_rate = 0.06,
            subsample = 0.9,
            colsample_bytree = 0.8,
            eval_metric = "auc",
            n_jobs = -1,
            random_state = 42,
            scale_pos_weight = scale_pos_weight,
        ))
    ]
)

xgb_clean.fit(X_train, y_train)

y_proba = xgb_clean.predict_proba(X_test)[:, 1]
y_pred = (y_proba > 0.5).astype(int)

auc = roc_auc_score(y_test, y_proba)
print(f"\nXGBoost (no data leakage) – ROC-AUC: {auc:.4f}\n")
print(classification_report(y_test, y_pred))
