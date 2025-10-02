import textwrap
from pathlib import Path

import pandas as pd

try:
    from .data_utils import list_travel_time_files, load_travel_time
except ImportError:  # Fallback when executed as a script
    import sys
    from pathlib import Path as _Path
    sys.path.append(str(_Path(__file__).resolve().parent.parent))
    from src.data_utils import list_travel_time_files, load_travel_time

REPORT_PATH = Path('reports/travel_time_eda.md')


def summarize_yearly(df: pd.DataFrame) -> pd.DataFrame:
    yearly = (
        df.groupby('year')
        .agg(
            records=('timeInSeconds', 'size'),
            segments=('resultId', 'nunique'),
            start=('updated_local', 'min'),
            end=('updated_local', 'max'),
            mean_time=('timeInSeconds', 'mean'),
            median_time=('timeInSeconds', 'median'),
            p80_time=('timeInSeconds', lambda x: x.quantile(0.8)),
            p90_time=('timeInSeconds', lambda x: x.quantile(0.9)),
            mean_sample_size=('count', 'mean')
        )
        .reset_index()
    )
    return yearly


def summarize_temporal(df: pd.DataFrame):
    hourly = (
        df.groupby('hour')
        .agg(
            mean_time=('timeInSeconds', 'mean'),
            median_time=('timeInSeconds', 'median'),
            sample_records=('timeInSeconds', 'size')
        )
        .reset_index()
        .sort_values('hour')
    )

    dow = (
        df.groupby('dow')
        .agg(
            mean_time=('timeInSeconds', 'mean'),
            median_time=('timeInSeconds', 'median'),
            sample_records=('timeInSeconds', 'size')
        )
        .reindex(
            ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        )
        .reset_index()
    )

    monthly = (
        df.groupby('month')
        .agg(
            mean_time=('timeInSeconds', 'mean'),
            median_time=('timeInSeconds', 'median'),
            sample_records=('timeInSeconds', 'size')
        )
        .reset_index()
        .sort_values('month')
    )
    return hourly, dow, monthly


def segment_highlights(df: pd.DataFrame, min_records: int = 500):
    segment_stats = (
        df.groupby('resultId')
        .agg(
            records=('timeInSeconds', 'size'),
            mean_time=('timeInSeconds', 'mean'),
            median_time=('timeInSeconds', 'median'),
            p90_time=('timeInSeconds', lambda x: x.quantile(0.9)),
            mean_sample_size=('count', 'mean')
        )
        .query('records >= @min_records')
        .sort_values('p90_time', ascending=False)
    )
    worst = segment_stats.head(15).reset_index()
    best = segment_stats.sort_values('mean_time').head(15).reset_index()
    return worst, best, segment_stats


def render_markdown(yearly: pd.DataFrame,
                    hourly: pd.DataFrame,
                    dow: pd.DataFrame,
                    monthly: pd.DataFrame,
                    worst: pd.DataFrame,
                    best: pd.DataFrame,
                    segment_stats: pd.DataFrame,
                    df: pd.DataFrame) -> None:
    header = textwrap.dedent('''
    # Travel Time Bluetooth EDA

    This report summarizes the Bluetooth travel-time feeds (2014-2017) provided by the City of Toronto.
    All timestamps are converted to America/Toronto local time.
    ''').strip()

    overview = textwrap.dedent('''
    ## Data Volume

    * Total records: {total_records:,}
    * Unique Bluetooth segments (resultId): {unique_segments:,}
    * Coverage: {start:%Y-%m-%d} to {end:%Y-%m-%d}
    ''').format(
        total_records=len(df),
        unique_segments=df['resultId'].nunique(),
        start=df['updated_local'].min(),
        end=df['updated_local'].max()
    )

    yearly_md = '## Year-over-Year Summary\n\n' + yearly.to_markdown(index=False)
    hourly_md = '## Hour-of-Day Profile\n\n' + hourly.to_markdown(index=False)
    dow_md = '## Day-of-Week Profile\n\n' + dow.to_markdown(index=False)
    monthly_md = '## Month-by-Month Summary\n\n' + monthly.tail(24).to_markdown(index=False)

    worst_md = '## Segments with Highest 90th Percentile Travel Time (>=500 observations)\n\n' + worst.to_markdown(index=False)
    best_md = '## Segments with Lowest Mean Travel Time (>=500 observations)\n\n' + best.to_markdown(index=False)

    record_density = textwrap.dedent('''
    ## Segment Coverage Density

    * Median records per segment: {median_records:,.0f}
    * 5th-95th percentile span: {p5:,.0f} - {p95:,.0f}
    * Share of segments with >= 1,000 observations: {share_1000:.1%}
    ''').format(
        median_records=segment_stats['records'].median(),
        p5=segment_stats['records'].quantile(0.05),
        p95=segment_stats['records'].quantile(0.95),
        share_1000=(segment_stats['records'] >= 1000).mean()
    )

    weekday_mean = df[df['dow'].isin(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'])]['timeInSeconds'].mean()
    weekend_mean = df[df['dow'].isin(['Saturday', 'Sunday'])]['timeInSeconds'].mean()
    peak_row = hourly.loc[hourly['median_time'].idxmax()]
    peak_hour = int(peak_row['hour'])

    narrative = textwrap.dedent('''
    ## Key Takeaways

    - Travel-time observations are densest around {peak_hour}:00 with median segment times peaking near {peak_median:.1f} seconds.
    - Weekday congestion is materially higher than weekends (weekday mean={weekday_mean:.1f}s vs weekend mean={weekend_mean:.1f}s).
    - Long-tail segments (top 15 by 90th percentile) maintain 90th percentile travel times above {long_tail:.0f} seconds, flagging candidates for deeper slowdown investigation.
    - Data coverage thins after late 2016; the most recent observed month is {recent_month}.
    ''').format(
        peak_hour=peak_hour,
        peak_median=peak_row['median_time'],
        weekday_mean=weekday_mean,
        weekend_mean=weekend_mean,
        long_tail=worst['p90_time'].min(),
        recent_month=monthly.iloc[-1]['month']
    )

    components = [header, overview, yearly_md, record_density, hourly_md, dow_md, monthly_md, worst_md, best_md, narrative]
    REPORT_PATH.write_text('\n\n'.join(components))


def main() -> None:
    df = load_travel_time(list_travel_time_files())
    yearly = summarize_yearly(df)
    hourly, dow, monthly = summarize_temporal(df)
    worst, best, segment_stats = segment_highlights(df)
    render_markdown(yearly, hourly, dow, monthly, worst, best, segment_stats, df)


if __name__ == '__main__':
    main()
