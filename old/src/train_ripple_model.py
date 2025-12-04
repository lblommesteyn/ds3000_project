from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
    roc_auc_score,
)

DATA_PATH = Path('data/processed/ripple_incidents_2025.parquet')
MODEL_DIR = Path('models')
REPORT_DIR = Path('reports')
FIG_DIR = REPORT_DIR / 'figures'

MODEL_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError('Processed dataset missing; run build_ripple_dataset.py first.')
    return pd.read_parquet(path).sort_values('subway_timestamp').reset_index(drop=True)


def train_test_split(df: pd.DataFrame, frac: float = 0.8):
    split_idx = int(len(df) * frac)
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def prepare_features(train: pd.DataFrame, test: pd.DataFrame):
    exclude = {
        'target_surface_delay',
        'target_surface_events',
        'target_linger_minutes',
        'ripple_label',
        'severe_label',
        'subway_timestamp',
        'incident_id',
    }
    feature_cols = [col for col in train.columns if col not in exclude]
    categorical_cols = [col for col in ['station_key', 'subway_code', 'subway_line', 'day_name'] if col in feature_cols]
    for df in (train, test):
        for col in categorical_cols:
            df[col] = df[col].astype(str)
    return feature_cols, categorical_cols


def _catboost_classifier(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    categorical_cols: list[str],
    label: str,
    class_weights: tuple[float, float],
    verbose: int = 200,
):
    train_pool = Pool(train_df[feature_cols], label=train_df[label], cat_features=[feature_cols.index(col) for col in categorical_cols])
    test_pool = Pool(test_df[feature_cols], label=test_df[label], cat_features=[feature_cols.index(col) for col in categorical_cols])

    model = CatBoostClassifier(
        loss_function='Logloss',
        depth=7,
        learning_rate=0.05,
        iterations=1500,
        random_seed=42,
        l2_leaf_reg=4.0,
        class_weights=list(class_weights),
        early_stopping_rounds=100,
        eval_metric='AUC',
        verbose=verbose,
    )
    model.fit(train_pool, eval_set=test_pool, use_best_model=True)

    train_prob = model.predict_proba(train_pool)[:, 1]
    test_prob = model.predict_proba(test_pool)[:, 1]

    thresholds = np.linspace(0.1, 0.8, 36)
    train_f1 = [f1_score(train_df[label], (train_prob >= t).astype(int), zero_division=0) for t in thresholds]
    threshold = float(thresholds[int(np.argmax(train_f1))])

    train_pred = (train_prob >= threshold).astype(int)
    test_pred = (test_prob >= threshold).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(test_df[label], test_pred, average='binary', zero_division=0)
    tn, fp, fn, tp = confusion_matrix(test_df[label], test_pred).ravel()

    metrics = {
        'threshold': threshold,
        'train': {
            'auc': float(roc_auc_score(train_df[label], train_prob)),
            'average_precision': float(average_precision_score(train_df[label], train_prob)),
            'f1': float(f1_score(train_df[label], train_pred, zero_division=0)),
        },
        'test': {
            'auc': float(roc_auc_score(test_df[label], test_prob)),
            'average_precision': float(average_precision_score(test_df[label], test_prob)),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'tp': int(tp),
        },
    }

    return model, metrics, train_prob, test_prob, threshold


def train_ripple_classifier(train_df, test_df, feature_cols, categorical_cols):
    positives = train_df['ripple_label'].mean()
    weight_neg = 1.0
    weight_pos = max(weight_neg, weight_neg * (1 - positives) / max(positives, 1e-6))
    return _catboost_classifier(train_df, test_df, feature_cols, categorical_cols, 'ripple_label', (weight_neg, weight_pos))


def train_severe_classifier(train_df, test_df, feature_cols, categorical_cols):
    positives = train_df['severe_label'].mean()
    weight_neg = 1.0
    weight_pos = max(weight_neg, weight_neg * (1 - positives) / max(positives, 1e-6))
    return _catboost_classifier(train_df, test_df, feature_cols, categorical_cols, 'severe_label', (weight_neg, weight_pos))


def train_positive_regressor(train_df, test_df, feature_cols, categorical_cols):
    train_pos = train_df[train_df['target_surface_delay'] > 0]
    if train_pos.empty:
        raise RuntimeError('No positive-delay incidents in training set.')

    train_pool = Pool(
        train_pos[feature_cols],
        label=np.log1p(train_pos['target_surface_delay']),
        cat_features=[feature_cols.index(col) for col in categorical_cols],
    )
    model = CatBoostRegressor(
        loss_function='RMSE',
        depth=8,
        learning_rate=0.05,
        iterations=2000,
        random_seed=42,
        l2_leaf_reg=3.0,
        early_stopping_rounds=100,
        eval_metric='RMSE',
        verbose=200,
    )
    model.fit(train_pool, use_best_model=True)

    test_pool_all = Pool(
        test_df[feature_cols],
        cat_features=[feature_cols.index(col) for col in categorical_cols],
    )
    train_pool_all = Pool(
        train_df[feature_cols],
        cat_features=[feature_cols.index(col) for col in categorical_cols],
    )

    train_pred_cond = np.expm1(model.predict(train_pool_all))
    test_pred_cond = np.expm1(model.predict(test_pool_all))
    train_pred_cond = np.clip(train_pred_cond, 0, None)
    test_pred_cond = np.clip(test_pred_cond, 0, None)

    train_pos_pred = train_pred_cond[: len(train_df)][train_df['target_surface_delay'] > 0]
    train_pos_actual = train_df.loc[train_df['target_surface_delay'] > 0, 'target_surface_delay']
    test_pos_pred = test_pred_cond[test_df['target_surface_delay'] > 0]
    test_pos_actual = test_df.loc[test_df['target_surface_delay'] > 0, 'target_surface_delay']

    pos_metrics = {
        'train': {
            'mae': float(mean_absolute_error(train_pos_actual, train_pos_pred)),
            'rmse': float(np.sqrt(mean_squared_error(train_pos_actual, train_pos_pred))),
            'r2': float(r2_score(train_pos_actual, train_pos_pred)),
        },
        'test': {
            'mae': float(mean_absolute_error(test_pos_actual, test_pos_pred)),
            'rmse': float(np.sqrt(mean_squared_error(test_pos_actual, test_pos_pred))),
            'r2': float(r2_score(test_pos_actual, test_pos_pred)),
        },
    }

    return model, pos_metrics, train_pred_cond, test_pred_cond


def plot_importance(model, feature_cols, title: str, output_path: Path):
    importances = model.get_feature_importance()
    order = np.argsort(importances)[::-1][:20]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh([feature_cols[i] for i in order][::-1], importances[order][::-1], color='#1f77b4')
    ax.set_title(title)
    ax.set_xlabel('Importance')
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return {feature_cols[i]: float(importances[i]) for i in np.argsort(importances)[::-1]}


def plot_shap(model, test_df, feature_cols, categorical_cols, output_path: Path):
    sample = test_df.sample(n=min(2000, len(test_df)), random_state=42)
    pool = Pool(sample[feature_cols], cat_features=[feature_cols.index(col) for col in categorical_cols])
    shap_values = model.get_feature_importance(pool, type='ShapValues')
    shap_vals = shap_values[:, :-1]
    shap_mean = np.mean(np.abs(shap_vals), axis=0)
    order = np.argsort(shap_mean)[::-1][:20]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh([feature_cols[i] for i in order][::-1], shap_mean[order][::-1], color='#ff7f0e')
    ax.set_title('SHAP Mean |Impact| (Positive-delay Regressor)')
    ax.set_xlabel('Mean |SHAP value|')
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return {feature_cols[i]: float(shap_mean[i]) for i in np.argsort(shap_mean)[::-1]}


def main() -> None:
    df = load_dataset(DATA_PATH)
    train_df, test_df = train_test_split(df)
    feature_cols, categorical_cols = prepare_features(train_df, test_df)

    ripple_model, ripple_metrics, ripple_train_prob, ripple_test_prob, ripple_threshold = train_ripple_classifier(train_df, test_df, feature_cols, categorical_cols)
    severe_model, severe_metrics, severe_train_prob, severe_test_prob, severe_threshold = train_severe_classifier(train_df, test_df, feature_cols, categorical_cols)
    reg_model, reg_pos_metrics, reg_train_cond, reg_test_cond = train_positive_regressor(train_df, test_df, feature_cols, categorical_cols)

    ripple_train_pred = (ripple_train_prob >= ripple_threshold).astype(int)
    ripple_test_pred = (ripple_test_prob >= ripple_threshold).astype(int)

    threshold_train_pred = ripple_train_pred * reg_train_cond
    threshold_test_pred = ripple_test_pred * reg_test_cond

    expected_train_pred = ripple_train_prob * reg_train_cond
    expected_test_pred = ripple_test_prob * reg_test_cond

    threshold_metrics = {
        'train': {
            'mae': float(mean_absolute_error(train_df['target_surface_delay'], threshold_train_pred)),
            'rmse': float(np.sqrt(mean_squared_error(train_df['target_surface_delay'], threshold_train_pred))),
            'r2': float(r2_score(train_df['target_surface_delay'], threshold_train_pred)),
        },
        'test': {
            'mae': float(mean_absolute_error(test_df['target_surface_delay'], threshold_test_pred)),
            'rmse': float(np.sqrt(mean_squared_error(test_df['target_surface_delay'], threshold_test_pred))),
            'r2': float(r2_score(test_df['target_surface_delay'], threshold_test_pred)),
        },
    }

    expected_metrics = {
        'train': {
            'mae': float(mean_absolute_error(train_df['target_surface_delay'], expected_train_pred)),
            'rmse': float(np.sqrt(mean_squared_error(train_df['target_surface_delay'], expected_train_pred))),
            'r2': float(r2_score(train_df['target_surface_delay'], expected_train_pred)),
        },
        'test': {
            'mae': float(mean_absolute_error(test_df['target_surface_delay'], expected_test_pred)),
            'rmse': float(np.sqrt(mean_squared_error(test_df['target_surface_delay'], expected_test_pred))),
            'r2': float(r2_score(test_df['target_surface_delay'], expected_test_pred)),
        },
    }

    ripple_model.save_model(MODEL_DIR / 'ripple_binary_classifier.cbm')
    severe_model.save_model(MODEL_DIR / 'ripple_severe_classifier.cbm')
    reg_model.save_model(MODEL_DIR / 'ripple_delay_regressor.cbm')

    ripple_fi = plot_importance(ripple_model, feature_cols, 'Feature Importance: Ripple Classifier', FIG_DIR / 'ripple_classifier_feature_importance.png')
    severe_fi = plot_importance(severe_model, feature_cols, 'Feature Importance: Extreme Ripple Classifier', FIG_DIR / 'severe_classifier_feature_importance.png')
    reg_fi = plot_importance(reg_model, feature_cols, 'Feature Importance: Positive-delay Regressor', FIG_DIR / 'ripple_regressor_feature_importance.png')
    shap_summary = plot_shap(reg_model, test_df, feature_cols, categorical_cols, FIG_DIR / 'ripple_regressor_shap.png')

    metrics_report = {
        'ripple_classifier': ripple_metrics,
        'severe_classifier': severe_metrics,
        'positive_delay_regression': reg_pos_metrics,
        'delay_prediction_threshold': threshold_metrics,
        'delay_prediction_expected': expected_metrics,
        'feature_importance': {
            'ripple_classifier': ripple_fi,
            'severe_classifier': severe_fi,
            'positive_delay_regressor': reg_fi,
        },
        'shap_summary_positive_regressor': shap_summary,
    }

    REPORT_DIR.joinpath('ripple_model_metrics.json').write_text(json.dumps(metrics_report, indent=2), encoding='utf-8')
    print('Training complete. Metrics stored in reports/ripple_model_metrics.json')
if __name__ == '__main__':
    main()









