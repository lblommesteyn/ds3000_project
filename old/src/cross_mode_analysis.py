from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Iterable, Tuple

import pandas as pd
from dateutil import tz

TORONTO_TZ = tz.gettz('America/Toronto')
DATA_DIR = Path('.')
REPORT_PATH = Path('reports/cross_mode_propagation.md')
WINDOW_MINUTES = 60

SUBWAY_FILE = DATA_DIR / 'TTC Subway Delay Data since 2025.csv'
BUS_FILE = DATA_DIR / 'TTC Bus Delay Data since 2025.csv'
STREETCAR_FILE = DATA_DIR / 'TTC Streetcar Delay Data since 2025.csv'
CODE_FILES = [
    DATA_DIR / 'Code Descriptions.csv',
    DATA_DIR / 'Code Descriptions (1).csv',
    DATA_DIR / 'Code Descriptions (2).csv',
]


def load_codebook(files: Iterable[Path]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for path in files:
        if not path.exists():
            continue
        try:
            book = pd.read_csv(path)
        except Exception:
            continue
        if 'CODE' in book.columns and 'DESCRIPTION' in book.columns:
            book = book.dropna(subset=['CODE'])
            codes = book['CODE'].astype(str).str.upper()
            desc = book['DESCRIPTION'].astype(str)
            mapping.update(dict(zip(codes, desc)))
    return mapping


def normalize_station(value: str) -> str:
    if not isinstance(value, str):
        return ''
    value = value.upper().strip()
    for token in [' STATION', ' STN', ' PLATFORM', ' BD', ' YUS', ' RT', ' LINE', ' SUBWAY', ' LOOP']:
        value = value.replace(token, '')
    value = value.replace("'", '')
    value = ' '.join(value.split())
    if ' AND ' in value:
        value = value.split(' AND ')[0]
    return value.strip()


def parse_datetime(date_col: pd.Series, time_col: pd.Series) -> pd.Series:
    dates = pd.to_datetime(date_col, errors='coerce')
    times = pd.to_datetime(time_col, format='%H:%M', errors='coerce').dt.time
    timestamps = []
    for d, t in zip(dates, times):
        if pd.isna(d) or pd.isna(t):
            timestamps.append(pd.NaT)
            continue
        ts = pd.Timestamp.combine(d.date(), t).tz_localize(TORONTO_TZ)
        timestamps.append(ts)
    return pd.Series(timestamps, index=date_col.index)


def load_mode_file(path: Path, mode: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.dropna(subset=['Station', 'Time', 'Date'])
    df['timestamp'] = parse_datetime(df['Date'], df['Time'])
    df = df.dropna(subset=['timestamp'])
    df['station_key'] = df['Station'].map(normalize_station)
    df = df[df['station_key'] != '']
    df['Min Delay'] = pd.to_numeric(df['Min Delay'], errors='coerce')
    df['Min Gap'] = pd.to_numeric(df.get('Min Gap'), errors='coerce')
    df['mode'] = mode
    return df


def clean_ascii(text: str) -> str:
    return text.encode('ascii', 'ignore').decode()


def match_surface_events(
    subway_df: pd.DataFrame,
    surface_df: pd.DataFrame,
    window_minutes: int = WINDOW_MINUTES,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    window = pd.Timedelta(minutes=window_minutes)
    surface_by_station = {key: grp.sort_values('timestamp') for key, grp in surface_df.groupby('station_key')}
    match_rows = []
    incident_rows = []

    for station_key, sub_grp in subway_df.groupby('station_key'):
        if 'TORONTO TRANSIT' in station_key:
            continue
        surf_grp = surface_by_station.get(station_key)
        if surf_grp is None or surf_grp.empty:
            continue
        sub_grp = sub_grp.sort_values('timestamp')
        for _, sub_row in sub_grp.iterrows():
            start = sub_row['timestamp']
            end = start + window
            subset = surf_grp[(surf_grp['timestamp'] >= start) & (surf_grp['timestamp'] <= end)]
            matched_count = len(subset)
            total_delay = float(subset['Min Delay'].fillna(0).sum()) if matched_count else 0.0
            max_offset = (
                (subset['timestamp'].max() - start).total_seconds() / 60.0
                if matched_count
                else 0.0
            )

            incident_rows.append(
                {
                    'subway_id': sub_row['_id'] if '_id' in sub_row else None,
                    'station_key': station_key,
                    'subway_timestamp': start,
                    'subway_code': str(sub_row.get('Code', '')).upper(),
                    'subway_line': sub_row.get('Line'),
                    'subway_delay': float(sub_row.get('Min Delay')) if pd.notna(sub_row.get('Min Delay')) else 0.0,
                    'matched_surface_events': matched_count,
                    'matched_surface_delay': total_delay,
                    'max_offset_minutes': max_offset,
                }
            )

            if matched_count == 0:
                continue

            for _, surf_row in subset.iterrows():
                match_rows.append(
                    {
                        'subway_id': sub_row['_id'] if '_id' in sub_row else None,
                        'station_key': station_key,
                        'subway_timestamp': start,
                        'subway_code': str(sub_row.get('Code', '')).upper(),
                        'subway_line': sub_row.get('Line'),
                        'surface_timestamp': surf_row['timestamp'],
                        'mode': surf_row['mode'],
                        'surface_delay': float(surf_row.get('Min Delay')) if pd.notna(surf_row.get('Min Delay')) else 0.0,
                        'surface_gap': float(surf_row.get('Min Gap')) if pd.notna(surf_row.get('Min Gap')) else 0.0,
                        'surface_line': surf_row.get('Line'),
                        'surface_station_key': surf_row['station_key'],
                        'surface_code': str(surf_row.get('Code', '')).upper(),
                        'offset_minutes': (surf_row['timestamp'] - start).total_seconds() / 60.0,
                    }
                )

    matches = pd.DataFrame(match_rows)
    incident_summary = pd.DataFrame(incident_rows)
    return matches, incident_summary


def aggregate_surface_impacts(
    subway_df: pd.DataFrame,
    surface_df: pd.DataFrame,
    codebook: dict[str, str],
    window_minutes: int = WINDOW_MINUTES,
) -> pd.DataFrame:
    window = pd.Timedelta(minutes=window_minutes)
    surface_by_station = {key: grp.sort_values('timestamp') for key, grp in surface_df.groupby('station_key')}
    rows = []

    for station_key, sub_grp in subway_df.groupby('station_key'):
        if 'TORONTO TRANSIT' in station_key:
            continue
        surf_grp = surface_by_station.get(station_key)
        if surf_grp is None or surf_grp.empty:
            continue
        sub_grp = sub_grp.sort_values('timestamp')
        subway_incidents = len(sub_grp)
        total_surface_events = 0
        total_surface_delay = 0.0
        bus_hits = 0
        streetcar_hits = 0

        for _, sub_row in sub_grp.iterrows():
            start = sub_row['timestamp']
            end = start + window
            subset = surf_grp[(surf_grp['timestamp'] >= start) & (surf_grp['timestamp'] <= end)]
            if subset.empty:
                continue
            matched_delay = float(subset['Min Delay'].fillna(0).sum())
            total_surface_events += len(subset)
            total_surface_delay += matched_delay
            bus_hits += int((subset['mode'] == 'bus').sum())
            streetcar_hits += int((subset['mode'] == 'streetcar').sum())

        if subway_incidents == 0 or total_surface_events == 0:
            continue

        top_codes = sub_grp['Code'].astype(str).str.upper().value_counts().head(3)
        top_code_descriptions = []
        for code in top_codes.index.tolist():
            desc = codebook.get(code, 'Unknown')
            top_code_descriptions.append(clean_ascii(f"{code}: {desc}"))

        rows.append(
            {
                'station_key': station_key,
                'subway_incidents': subway_incidents,
                'surface_events_in_window': total_surface_events,
                'surface_delay_minutes': total_surface_delay,
                'bus_hits': bus_hits,
                'streetcar_hits': streetcar_hits,
                'surface_events_per_subway': total_surface_events / subway_incidents,
                'avg_surface_delay_per_subway': total_surface_delay / subway_incidents,
                'avg_subway_delay': sub_grp['Min Delay'].mean(),
                'top_codes': '; '.join(top_code_descriptions),
            }
        )

    if not rows:
        return pd.DataFrame()

    summary = pd.DataFrame(rows)
    summary = summary.sort_values('surface_events_per_subway', ascending=False)
    return summary


def render_report(summary: pd.DataFrame, subway_df: pd.DataFrame, surface_df: pd.DataFrame) -> None:
    if summary.empty:
        text = "# Cross-Mode Disruption Propagation\n\nNo overlapping incidents were detected between subway and surface datasets."
        REPORT_PATH.write_text(text, encoding='utf-8')
        return

    top10 = summary.head(10)
    table = top10[
        [
            'station_key',
            'subway_incidents',
            'surface_events_per_subway',
            'avg_surface_delay_per_subway',
            'bus_hits',
            'streetcar_hits',
            'avg_subway_delay',
            'top_codes',
        ]
    ].rename(
        columns={
            'station_key': 'Station',
            'subway_incidents': 'Subway Incidents',
            'surface_events_per_subway': 'Surface Events / Subway',
            'avg_surface_delay_per_subway': 'Surface Delay min / Subway',
            'bus_hits': 'Bus Hits',
            'streetcar_hits': 'Streetcar Hits',
            'avg_subway_delay': 'Avg Subway Delay (min)',
            'top_codes': 'Top Subway Causes',
        }
    )
    table['Surface Events / Subway'] = table['Surface Events / Subway'].map(lambda x: f"{x:0.2f}")
    table['Surface Delay min / Subway'] = table['Surface Delay min / Subway'].map(lambda x: f"{x:0.1f}")
    table['Avg Subway Delay (min)'] = table['Avg Subway Delay (min)'].map(lambda x: f"{x:0.1f}")

    overview = textwrap.dedent(
        f"""
        # Cross-Mode Disruption Propagation (2025 data)

        - Subway incidents analysed: {len(subway_df):,}
        - Surface delay events (bus+streetcar): {len(surface_df):,}
        - Stations with linked impacts (within {WINDOW_MINUTES} min): {summary.shape[0]}
        - Median surface events per subway incident: {summary['surface_events_per_subway'].median():0.2f}
        - Median added surface delay minutes per subway incident: {summary['avg_surface_delay_per_subway'].median():0.1f}
        """
    ).strip()

    table_md = table.to_markdown(index=False)

    insights = []
    top_station = summary.iloc[0]
    insights.append(
        f"**{top_station['station_key']}** sees the largest ripple: "
        f"{top_station['surface_events_per_subway']:0.1f} surface incidents per subway event, "
        f"adding {top_station['avg_surface_delay_per_subway']:0.1f} minutes of downstream delay on average."
    )
    surface_total = summary['surface_events_in_window'].sum()
    bus_share = (summary['bus_hits'].sum() / surface_total) if surface_total else 0
    insights.append(
        f"Buses account for {bus_share:0.1%} of linked surface incidents, highlighting which feeder services bear the brunt."
    )
    high_delay = summary.sort_values('avg_surface_delay_per_subway', ascending=False).iloc[0]
    insights.append(
        f"Peak added delay occurs at **{high_delay['station_key']}**, where surface riders absorb "
        f"{high_delay['avg_surface_delay_per_subway']:0.1f} minutes per subway disruption; "
        f"common subway causes: {high_delay['top_codes']}."
    )

    markdown = "\n\n".join(
        [
            overview,
            '## Top Ripple Stations\n\n' + table_md,
            '## Notable Findings\n\n' + '\n'.join(f"- {point}" for point in insights),
        ]
    )
    REPORT_PATH.write_text(markdown, encoding='utf-8')


def main() -> None:
    codebook = load_codebook(CODE_FILES)
    subway = load_mode_file(SUBWAY_FILE, mode='subway')
    bus = load_mode_file(BUS_FILE, mode='bus')
    streetcar = load_mode_file(STREETCAR_FILE, mode='streetcar')

    surface = pd.concat([bus, streetcar], ignore_index=True)

    summary = aggregate_surface_impacts(subway, surface, codebook)
    render_report(summary, subway, surface)


if __name__ == '__main__':
    main()
