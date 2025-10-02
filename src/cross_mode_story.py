from __future__ import annotations

from pathlib import Path
from typing import Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from src import cross_mode_analysis as cma
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    import src.cross_mode_analysis as cma  # type: ignore

FIG_DIR = Path('reports/figures')
FIG_DIR.mkdir(parents=True, exist_ok=True)
STORY_PATH = Path('reports/cross_mode_story.md')


def select_focus_station(summary: pd.DataFrame, matches: pd.DataFrame) -> str:
    if not summary.empty:
        return summary.sort_values('surface_events_in_window', ascending=False).iloc[0]['station_key']
    return matches['station_key'].mode().iat[0]


def build_timeline_plot(
    station: str,
    matches: pd.DataFrame,
    incidents: pd.DataFrame,
    output_path: Path,
) -> Tuple[str, float, float]:
    station_matches = matches[matches['station_key'] == station].copy()
    station_incidents = incidents[incidents['station_key'] == station].copy()
    if station_matches.empty:
        return '', 0.0, 0.0

    station_matches['subway_date'] = station_matches['subway_timestamp'].dt.date
    busiest_day = station_matches.groupby('subway_date').size().idxmax()

    subway_day = station_incidents[station_incidents['subway_timestamp'].dt.date == busiest_day]
    surface_day = station_matches[station_matches['subway_timestamp'].dt.date == busiest_day]

    if subway_day.empty:
        return '', 0.0, 0.0

    start = min(subway_day['subway_timestamp'].min(), surface_day['surface_timestamp'].min()).floor('15min')
    end = max(subway_day['subway_timestamp'].max(), surface_day['surface_timestamp'].max()).ceil('15min')
    time_index = pd.date_range(start=start, end=end, freq='15min')

    subway_counts = (
        subway_day.set_index('subway_timestamp')
        .resample('15min')
        .size()
        .reindex(time_index, fill_value=0)
    )
    surface_counts = (
        surface_day.set_index('surface_timestamp')
        .resample('15min')
        .size()
        .reindex(time_index, fill_value=0)
    )
    surface_delay = (
        surface_day.set_index('surface_timestamp')['surface_delay']
        .resample('15min')
        .sum()
        .reindex(time_index, fill_value=0)
    )

    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax_top.step(time_index, subway_counts, where='post', label='Subway incidents', color='#1f77b4')
    ax_top.bar(time_index, surface_counts, width=0.01, alpha=0.4, label='Surface ripple events', color='#ff7f0e')
    ax_top.set_ylabel('Events per 15 min')
    ax_top.set_title(f'{station.title()} ripple timeline on {busiest_day}')
    ax_top.legend(loc='upper left')

    ax_bottom.fill_between(time_index, surface_delay, step='post', alpha=0.3, color='#2ca02c')
    ax_bottom.plot(time_index, surface_delay.cumsum(), color='#2ca02c', label='Cumulative surface delay')
    ax_bottom.set_ylabel('Delay minutes')
    ax_bottom.set_xlabel('Time (local)')
    ax_bottom.legend(loc='upper left')

    ax_bottom.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    total_events = float(surface_counts.sum())
    total_delay = float(surface_delay.sum())
    return busiest_day.strftime('%Y-%m-%d'), total_events, total_delay


def plot_cause_breakdown(
    incidents: pd.DataFrame,
    codebook: dict[str, str],
    output_path: Path,
) -> Tuple[str, float]:
    if incidents.empty:
        return '', 0.0
    data = incidents.copy()
    data['has_surface'] = data['matched_surface_events'] > 0
    grouped = (
        data.groupby('subway_code')
        .agg(
            total_incidents=('subway_id', 'count'),
            share_with_surface=('has_surface', 'mean'),
            avg_surface_events=('matched_surface_events', 'mean'),
            avg_surface_delay=('matched_surface_delay', 'mean'),
        )
        .reset_index()
    )
    grouped = grouped[grouped['total_incidents'] >= 30]
    if grouped.empty:
        return '', 0.0
    grouped['description'] = grouped['subway_code'].map(lambda c: cma.clean_ascii(codebook.get(c, 'Unknown')))
    grouped['label'] = grouped['subway_code'] + ' - ' + grouped['description']
    grouped = grouped.sort_values('avg_surface_events', ascending=False).head(12)

    colors = plt.cm.Blues(grouped['share_with_surface'])

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.barh(grouped['label'], grouped['avg_surface_events'], color=colors)
    ax.invert_yaxis()
    ax.set_xlabel('Average surface events per subway incident')
    ax.set_title('Which subway incident causes ripple onto the surface?')

    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.Blues), ax=ax)
    cbar.set_label('Share of incidents with surface ripple')

    for i, val in enumerate(grouped['avg_surface_events']):
        ax.text(val + 0.02, i, f'{val:.2f}', va='center')

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    top_cause = grouped.iloc[0]
    return top_cause['label'], float(top_cause['avg_surface_delay'])


def plot_heatmap(incidents: pd.DataFrame, output_path: Path) -> Tuple[str, float]:
    data = incidents[incidents['matched_surface_events'] > 0].copy()
    if data.empty:
        return '', 0.0
    dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    data['hour'] = data['subway_timestamp'].dt.hour
    data['dow'] = pd.Categorical(data['subway_timestamp'].dt.day_name(), categories=dow_order, ordered=True)
    pivot = data.pivot_table(index='dow', columns='hour', values='matched_surface_events', aggfunc='sum').fillna(0)

    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(pivot.values, aspect='auto', cmap='Oranges')
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xticks(range(0, 24, 2))
    ax.set_xticklabels(range(0, 24, 2))
    ax.set_xlabel('Hour of day')
    ax.set_title('Surface ripple intensity by day and hour')
    fig.colorbar(im, ax=ax, label='Total surface events linked to subway incidents')
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    max_idx = np.unravel_index(np.argmax(pivot.values), pivot.values.shape)
    peak_dow = pivot.index[max_idx[0]]
    peak_hour = pivot.columns[max_idx[1]]
    peak_value = pivot.values[max_idx]
    return f'{peak_dow} {peak_hour:02d}:00', float(peak_value)


def plot_recovery(incidents: pd.DataFrame, output_path: Path) -> Tuple[str, float]:
    data = incidents[incidents['matched_surface_events'] > 0].copy()
    if data.empty:
        return '', 0.0
    recovery = (
        data.groupby('station_key')['max_offset_minutes']
        .mean()
        .sort_values(ascending=False)
        .head(10)
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(recovery.index, recovery.values, color='#9467bd')
    ax.invert_yaxis()
    ax.set_xlabel('Average minutes until surface ripple subsides')
    ax.set_title('How long do surface delays linger after a subway incident?')
    for i, val in enumerate(recovery.values):
        ax.text(val + 0.3, i, f'{val:.1f}', va='center')
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return recovery.index[0], float(recovery.iloc[0])


def main() -> None:
    codebook = cma.load_codebook(cma.CODE_FILES)
    subway = cma.load_mode_file(cma.SUBWAY_FILE, mode='subway')
    bus = cma.load_mode_file(cma.BUS_FILE, mode='bus')
    streetcar = cma.load_mode_file(cma.STREETCAR_FILE, mode='streetcar')
    surface = pd.concat([bus, streetcar], ignore_index=True)

    matches, incident_summary = cma.match_surface_events(subway, surface)
    summary = cma.aggregate_surface_impacts(subway, surface, codebook)

    if matches.empty:
        STORY_PATH.write_text('No overlapping incidents were detected.', encoding='utf-8')
        print('No overlapping incidents were detected.')
        return

    focus_station = select_focus_station(summary, matches)

    timeline_path = FIG_DIR / 'story_timeline.png'
    cause_path = FIG_DIR / 'story_cause_breakdown.png'
    heatmap_path = FIG_DIR / 'story_ripple_heatmap.png'
    recovery_path = FIG_DIR / 'story_recovery.png'

    day_string, day_events, day_delay = build_timeline_plot(focus_station, matches, incident_summary, timeline_path)
    top_cause_label, top_cause_delay = plot_cause_breakdown(incident_summary, codebook, cause_path)
    peak_slot, peak_value = plot_heatmap(incident_summary, heatmap_path)
    slow_station, slow_minutes = plot_recovery(incident_summary, recovery_path)

    lines = [
        '# Cross-Mode Propagation Story (2025)',
        '',
        f'- Focus station: **{focus_station.title()}** with the heaviest surface ripple.',
    ]
    if day_string:
        lines.append(
            f'- On **{day_string}**, {focus_station.title()} triggered {day_events:.0f} linked bus/streetcar incidents '
            f'adding {day_delay:.1f} minutes of surface delay (see `reports/figures/story_timeline.png`).'
        )
    if top_cause_label:
        lines.append(
            f'- The most surface-prone subway cause is **{top_cause_label}**, averaging {top_cause_delay:.1f} downstream delay minutes per incident '
            f'(`reports/figures/story_cause_breakdown.png`).'
        )
    if peak_slot:
        lines.append(
            f'- Surface ripple intensity peaks around **{peak_slot}**, with {peak_value:.0f} linked events in that slot '
            f'(`reports/figures/story_ripple_heatmap.png`).'
        )
    if slow_station:
        lines.append(
            f'- **{slow_station.title()}** is slowest to recover, with surface delays lingering {slow_minutes:.1f} minutes on average '
            f'(`reports/figures/story_recovery.png`).'
        )
    lines.append('')
    lines.append('These visuals highlight where a subway disruption cascades onto surface routes and how long riders stay affected.')

    STORY_PATH.write_text('\n'.join(lines), encoding='utf-8')
    print('Story visuals and summary saved to reports/figures and reports/cross_mode_story.md')


if __name__ == '__main__':
    main()
