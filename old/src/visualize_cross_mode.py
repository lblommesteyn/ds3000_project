from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

try:
    from src import cross_mode_analysis as cma
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    import src.cross_mode_analysis as cma  # type: ignore

FIG_DIR = Path('reports/figures')
FIG_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    codebook = cma.load_codebook(cma.CODE_FILES)
    subway = cma.load_mode_file(cma.SUBWAY_FILE, mode='subway')
    bus = cma.load_mode_file(cma.BUS_FILE, mode='bus')
    streetcar = cma.load_mode_file(cma.STREETCAR_FILE, mode='streetcar')
    surface = pd.concat([bus, streetcar], ignore_index=True)
    summary = cma.aggregate_surface_impacts(
        subway,
        surface,
        codebook,
        window_minutes=cma.WINDOW_MINUTES,
    )
    if summary.empty:
        print('No overlapping incidents detected; no figures generated.')
        return

    top_incident = summary.head(10)

    try:
        plt.style.use('seaborn-v0_8')
    except OSError:
        plt.style.use('seaborn')

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top_incident['station_key'], top_incident['surface_events_per_subway'], color='#1f77b4')
    ax.invert_yaxis()
    ax.set_xlabel('Surface incidents per subway incident')
    ax.set_ylabel('Station')
    ax.set_title('Top Stations by Surface Ripple Rate (within 60 minutes)')
    for i, val in enumerate(top_incident['surface_events_per_subway']):
        ax.text(val + 0.01, i, f'{val:.2f}', va='center')
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'top_surface_incidents_per_subway.png', dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top_incident['station_key'], top_incident['avg_surface_delay_per_subway'], color='#ff7f0e')
    ax.invert_yaxis()
    ax.set_xlabel('Added surface delay minutes per subway incident')
    ax.set_ylabel('Station')
    ax.set_title('Top Stations by Added Surface Delay per Subway Incident')
    for i, val in enumerate(top_incident['avg_surface_delay_per_subway']):
        ax.text(val + 0.1, i, f'{val:.1f}', va='center')
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'top_surface_delay_per_subway.png', dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    indices = range(len(top_incident))
    ax.barh(indices, top_incident['bus_hits'], height=0.5, label='Bus', color='#2ca02c')
    ax.barh(indices, top_incident['streetcar_hits'], height=0.5, left=top_incident['bus_hits'], label='Streetcar', color='#d62728')
    ax.set_yticks(list(indices))
    ax.set_yticklabels(top_incident['station_key'])
    ax.invert_yaxis()
    ax.set_xlabel('Linked surface delay events (within 60 minutes)')
    ax.set_title('Mode Mix of Surface Ripple Events (Top Stations)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'surface_mode_mix_top_stations.png', dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(
        summary['surface_events_per_subway'],
        summary['avg_surface_delay_per_subway'],
        s=summary['subway_incidents'] * 0.05,
        alpha=0.6,
        color='#9467bd'
    )
    for _, row in summary.nlargest(6, 'avg_surface_delay_per_subway').iterrows():
        ax.annotate(
            row['station_key'],
            (row['surface_events_per_subway'], row['avg_surface_delay_per_subway']),
            textcoords='offset points',
            xytext=(5, 5),
            fontsize=8,
        )
    ax.set_xlabel('Surface incidents per subway incident')
    ax.set_ylabel('Added surface delay minutes per subway incident')
    ax.set_title('Ripple Severity vs Frequency by Station (bubble size ~ subway incidents)')
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'ripple_severity_vs_frequency.png', dpi=150)
    plt.close(fig)

    print('Visualizations saved to', FIG_DIR)


if __name__ == '__main__':
    main()
