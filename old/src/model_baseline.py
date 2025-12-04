from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_pinball_loss,
    mean_squared_error,
    r2_score,
)

try:
    from .data_utils import list_travel_time_files, load_travel_time
except ImportError:  # Fallback when executed as a script
    import sys
    from pathlib import Path as _Path

    sys.path.append(str(_Path(__file__).resolve().parent.parent))
    from src.data_utils import list_travel_time_files, load_travel_time

REPORT_PATH = Path('reports/baseline_model_metrics.json')


def prepare_datasets(cutoff: str = '2017-01-01', verbose: bool = True):
    if verbose:
        print('Preparing datasets...')
        print('  Loading travel-time CSV files')
    files = list_travel_time_files()
    if verbose:
        print(f'  Found {len(files)} files')
    df = load_travel_time(files)
    if verbose:
        print(f'  Loaded dataframe with {len(df):,} rows across {df.resultId.nunique()} segments')

    df['log_count'] = np.log1p(df['count'])
    cutoff_ts = pd.Timestamp(cutoff, tz='America/Toronto')
    train = df[df['updated_local'] < cutoff_ts].copy()
    test = df[df['updated_local'] >= cutoff_ts].copy()

    if verbose:
        print(f'  Train window: < {cutoff_ts.date()} | rows={len(train):,}')
        print(f'  Test window:  >= {cutoff_ts.date()} | rows={len(test):,}')

    result_categories = sorted(train['resultId'].unique())
    result_map = {name: idx for idx, name in enumerate(result_categories)}
    train['result_code'] = train['resultId'].map(result_map).astype('int16')
    test['result_code'] = test['resultId'].map(result_map).fillna(-1).astype('int16')

    dow_categories = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    dow_type = pd.CategoricalDtype(categories=dow_categories, ordered=True)
    train['dow_code'] = train['dow'].astype(dow_type).cat.codes.astype('int16')
    test['dow_code'] = test['dow'].astype(dow_type).cat.codes.astype('int16')

    month_categories = sorted(df['month'].unique())
    month_type = pd.CategoricalDtype(categories=month_categories, ordered=True)
    train['month_code'] = train['month'].astype(month_type).cat.codes.astype('int16')
    test['month_code'] = test['month'].astype(month_type).cat.codes.astype('int16')

    if verbose:
        print('  Computing segment-level priors')
    segment_mean = train.groupby('resultId')['timeInSeconds'].mean()
    segment_p90 = train.groupby('resultId')['timeInSeconds'].quantile(0.9)
    segment_records = train.groupby('resultId').size()

    train['segment_mean'] = train['resultId'].map(segment_mean)
    test['segment_mean'] = test['resultId'].map(segment_mean).fillna(segment_mean.mean())

    train['segment_p90'] = train['resultId'].map(segment_p90)
    test['segment_p90'] = test['resultId'].map(segment_p90).fillna(segment_p90.mean())

    train['segment_records_log'] = np.log1p(train['resultId'].map(segment_records))
    test['segment_records_log'] = np.log1p(test['resultId'].map(segment_records)).fillna(0)

    feature_cols = [
        'result_code',
        'hour',
        'dow_code',
        'month_code',
        'log_count',
        'year',
        'segment_mean',
        'segment_p90',
        'segment_records_log',
    ]

    if verbose:
        print(f'  Feature columns: {feature_cols}')

    X_train = train[feature_cols]
    y_train = train['timeInSeconds']
    X_test = test[feature_cols]
    y_test = test['timeInSeconds']

    if verbose:
        print('  Dataset preparation complete')

    return X_train, y_train, X_test, y_test, test


def fit_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    verbose: bool = True,
):
    if verbose:
        print('Fitting models...')
        print(f'  Training rows available: {len(X_train):,}')

    sample_target = y_train
    if len(X_train) > 2_000_000:
        if verbose:
            print('  Sampling 2,000,000 rows for faster training')
        sampled = X_train.sample(n=2_000_000, random_state=42)
        sample_target = y_train.loc[sampled.index]
    else:
        sampled = X_train

    if verbose:
        print('  Training mean forecaster (HistGradientBoostingRegressor)')
    reg_mean = HistGradientBoostingRegressor(
        learning_rate=0.1,
        max_depth=8,
        max_iter=300,
        l2_regularization=0.1,
        random_state=42,
    )
    reg_mean.fit(sampled, sample_target)
    if verbose:
        print('  Mean forecaster trained')

    if verbose:
        print('  Training 90th percentile forecaster')
    reg_p90 = HistGradientBoostingRegressor(
        loss='quantile',
        quantile=0.9,
        learning_rate=0.05,
        max_depth=8,
        max_iter=400,
        l2_regularization=0.1,
        random_state=42,
    )
    reg_p90.fit(sampled, sample_target)
    if verbose:
        print('  90th percentile forecaster trained')

    return reg_mean, reg_p90


def evaluate_models(
    reg_mean,
    reg_p90,
    X_train,
    y_train,
    X_test,
    y_test,
    test_df: pd.DataFrame,
    *,
    verbose: bool = True,
):
    if verbose:
        print('Evaluating models...')
    train_pred = reg_mean.predict(X_train)
    test_pred = reg_mean.predict(X_test)
    test_p90 = reg_p90.predict(X_test)

    metrics = {
        'train': {
            'mae': float(mean_absolute_error(y_train, train_pred)),
            'rmse': float(np.sqrt(mean_squared_error(y_train, train_pred))),
            'mape': float(mean_absolute_percentage_error(y_train, train_pred)),
            'r2': float(r2_score(y_train, train_pred)),
        },
        'test': {
            'mae': float(mean_absolute_error(y_test, test_pred)),
            'rmse': float(np.sqrt(mean_squared_error(y_test, test_pred))),
            'mape': float(mean_absolute_percentage_error(y_test, test_pred)),
            'r2': float(r2_score(y_test, test_pred)),
            'pinball_loss_p90': float(mean_pinball_loss(y_test, test_p90, alpha=0.9)),
            'coverage_p90': float((y_test <= test_p90).mean()),
        },
    }

    if verbose:
        print(
            '  Test metrics -> '
            f"MAE: {metrics['test']['mae']:.2f}s | RMSE: {metrics['test']['rmse']:.2f}s | "
            f"MAPE: {metrics['test']['mape']:.3f} | R2: {metrics['test']['r2']:.3f}"
        )
        print(
            '  Quantile metrics -> '
            f"pinball loss (p90): {metrics['test']['pinball_loss_p90']:.2f} | "
            f"coverage(p90): {metrics['test']['coverage_p90']:.3f}"
        )

    test_with_preds = test_df.copy()
    test_with_preds['pred_mean'] = test_pred
    test_with_preds['pred_p90'] = test_p90
    segment_perf = (
        test_with_preds.groupby('resultId')
        .agg(
            records=('timeInSeconds', 'size'),
            mae=(
                'timeInSeconds',
                lambda y: float(
                    np.mean(np.abs(y - test_with_preds.loc[y.index, 'pred_mean']))
                ),
            ),
            rmse=(
                'timeInSeconds',
                lambda y: float(
                    np.sqrt(
                        np.mean(
                            (y - test_with_preds.loc[y.index, 'pred_mean']) ** 2
                        )
                    )
                ),
            ),
            coverage_p90=(
                'timeInSeconds',
                lambda y: float(np.mean(y <= test_with_preds.loc[y.index, 'pred_p90'])),
            ),
        )
        .sort_values('mae', ascending=False)
        .reset_index()
    )

    if verbose:
        print('  Evaluation complete')

    return metrics, segment_perf, test_with_preds


def save_report(metrics: dict, segment_perf: pd.DataFrame, *, verbose: bool = True):
    if verbose:
        print(f'Saving report to {REPORT_PATH}')
    segment_top = segment_perf.head(15).to_dict(orient='records')
    REPORT_PATH.write_text(
        json.dumps({'metrics': metrics, 'segment_perf_top15': segment_top}, indent=2)
    )
    if verbose:
        print('  Report saved')


def main() -> None:
    print('Baseline model pipeline start')
    X_train, y_train, X_test, y_test, test_df = prepare_datasets(verbose=True)
    reg_mean, reg_p90 = fit_models(X_train, y_train, verbose=True)
    metrics, segment_perf, _ = evaluate_models(
        reg_mean,
        reg_p90,
        X_train,
        y_train,
        X_test,
        y_test,
        test_df,
        verbose=True,
    )
    save_report(metrics, segment_perf, verbose=True)
    print('Baseline model pipeline complete')


if __name__ == '__main__':
    main()
