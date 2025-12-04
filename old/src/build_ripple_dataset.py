from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

try:
    from src import cross_mode_analysis as cma
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    import src.cross_mode_analysis as cma  # type: ignore

OUTPUT_PARQUET = Path('data/processed/ripple_incidents_2025.parquet')
OUTPUT_CSV = Path('data/processed/ripple_incidents_2025.csv')


def _expanding_mean(series: pd.Series) -> pd.Series:
    shifted = series.shift()
    return shifted.expanding().mean()


def _expanding_sum(series: pd.Series) -> pd.Series:
    shifted = series.shift().fillna(0)
    return shifted.cumsum()


def build_dataset() -> pd.DataFrame:
    cma.load_codebook(cma.CODE_FILES)
    subway = cma.load_mode_file(cma.SUBWAY_FILE, mode='subway')
    bus = cma.load_mode_file(cma.BUS_FILE, mode='bus')
    streetcar = cma.load_mode_file(cma.STREETCAR_FILE, mode='streetcar')
    surface = pd.concat([bus, streetcar], ignore_index=True)

    matches, incidents = cma.match_surface_events(subway, surface)
    if incidents.empty:
        raise RuntimeError('No incidents found after matching surface events.')

    incidents = (
        incidents.sort_values('subway_timestamp')
        .rename(columns={'subway_id': 'incident_id'})
        .reset_index(drop=True)
    )
    incidents['station_key'] = incidents['station_key'].fillna('UNKNOWN_STATION')
    incidents['subway_line'] = incidents['subway_line'].fillna('UNKNOWN_LINE')
    incidents['subway_code'] = incidents['subway_code'].fillna('UNKNOWN_CODE')

    incidents['hour'] = incidents['subway_timestamp'].dt.hour
    incidents['day_name'] = incidents['subway_timestamp'].dt.day_name()
    incidents['is_weekend'] = incidents['day_name'].isin(['Saturday', 'Sunday']).astype('int8')
    incidents['month'] = incidents['subway_timestamp'].dt.month.astype('int8')
    incidents['quarter'] = incidents['subway_timestamp'].dt.quarter.astype('int8')
    incidents['weekofyear'] = incidents['subway_timestamp'].dt.isocalendar().week.astype('int16')
    incidents['minute_of_day'] = (
        incidents['subway_timestamp'].dt.hour * 60
        + incidents['subway_timestamp'].dt.minute
    ).astype('int16')

    incidents['target_surface_delay'] = incidents['matched_surface_delay'].astype('float32')
    incidents['target_surface_events'] = incidents['matched_surface_events'].astype('float32')
    incidents['target_linger_minutes'] = incidents['max_offset_minutes'].astype('float32')
    incidents['ripple_label'] = (incidents['target_surface_delay'] > 0).astype('int8')
    incidents['severe_label'] = (
        (incidents['target_surface_delay'] >= 60)
        | (incidents['target_surface_events'] >= 3)
        | (incidents['target_linger_minutes'] >= 30)
    ).astype('int8')

    incidents['station_prev_delay_mean'] = (
        incidents.groupby('station_key')['target_surface_delay']
        .transform(_expanding_mean)
        .fillna(incidents['target_surface_delay'].mean())
        .astype('float32')
    )
    incidents['station_prev_events_mean'] = (
        incidents.groupby('station_key')['target_surface_events']
        .transform(_expanding_mean)
        .fillna(incidents['target_surface_events'].mean())
        .astype('float32')
    )
    incidents['station_prev_linger_mean'] = (
        incidents.groupby('station_key')['target_linger_minutes']
        .transform(_expanding_mean)
        .fillna(incidents['target_linger_minutes'].mean())
        .astype('float32')
    )
    incidents['station_prev_delay_total'] = (
        incidents.groupby('station_key')['target_surface_delay']
        .transform(_expanding_sum)
        .astype('float32')
    )
    incidents['station_prev_incidents'] = incidents.groupby('station_key').cumcount().astype('int16')

    incidents['code_prev_delay_mean'] = (
        incidents.groupby('subway_code')['target_surface_delay']
        .transform(_expanding_mean)
        .fillna(incidents['target_surface_delay'].mean())
        .astype('float32')
    )
    incidents['code_prev_events_mean'] = (
        incidents.groupby('subway_code')['target_surface_events']
        .transform(_expanding_mean)
        .fillna(incidents['target_surface_events'].mean())
        .astype('float32')
    )
    incidents['code_prev_incidents'] = incidents.groupby('subway_code').cumcount().astype('int16')

    incidents['line_prev_delay_mean'] = (
        incidents.groupby('subway_line')['target_surface_delay']
        .transform(_expanding_mean)
        .fillna(incidents['target_surface_delay'].mean())
        .astype('float32')
    )
    incidents['line_prev_incidents'] = incidents.groupby('subway_line').cumcount().astype('int16')

    incidents['global_prev_delay'] = (
        incidents['target_surface_delay'].shift().expanding().mean().fillna(0)
    ).astype('float32')

    incidents['recent_station_delay_3'] = (
        incidents.groupby('station_key')['target_surface_delay']
        .transform(lambda s: s.shift().rolling(window=3, min_periods=1).mean())
        .fillna(incidents['station_prev_delay_mean'])
        .astype('float32')
    )
    incidents['recent_station_events_3'] = (
        incidents.groupby('station_key')['target_surface_events']
        .transform(lambda s: s.shift().rolling(window=3, min_periods=1).mean())
        .fillna(incidents['station_prev_events_mean'])
        .astype('float32')
    )

    incidents['subway_delay'] = incidents['subway_delay'].fillna(0).astype('float32')

    keep_cols = [
        'incident_id',
        'station_key',
        'subway_timestamp',
        'subway_code',
        'subway_line',
        'subway_delay',
        'hour',
        'day_name',
        'is_weekend',
        'month',
        'quarter',
        'weekofyear',
        'minute_of_day',
        'station_prev_delay_mean',
        'station_prev_events_mean',
        'station_prev_linger_mean',
        'station_prev_delay_total',
        'station_prev_incidents',
        'recent_station_delay_3',
        'recent_station_events_3',
        'code_prev_delay_mean',
        'code_prev_events_mean',
        'code_prev_incidents',
        'line_prev_delay_mean',
        'line_prev_incidents',
        'global_prev_delay',
        'target_surface_delay',
        'target_surface_events',
        'target_linger_minutes',
        'ripple_label',
        'severe_label',
    ]

    dataset = incidents[keep_cols].copy()
    return dataset


def main() -> None:
    dataset = build_dataset()
    OUTPUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(OUTPUT_PARQUET, index=False)
    dataset.to_csv(OUTPUT_CSV, index=False)
    print(f'Saved dataset with {len(dataset):,} incidents to {OUTPUT_PARQUET} and {OUTPUT_CSV}')


if __name__ == '__main__':
    main()
