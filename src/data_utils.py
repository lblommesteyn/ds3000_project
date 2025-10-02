from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
from dateutil import tz

DATA_ROOT = Path('data')
TORONTO_TZ = tz.gettz('America/Toronto')


def list_travel_time_files() -> list[Path]:
    return sorted(DATA_ROOT.glob('travel-time-20*/travel-time-20*.csv'))


def load_travel_time(files: Optional[Iterable[Path]] = None,
                     add_time_features: bool = True) -> pd.DataFrame:
    files = list(files) if files is not None else list_travel_time_files()
    frames: list[pd.DataFrame] = []
    for path in files:
        year = int(path.stem.split('-')[-1])
        df = pd.read_csv(path, parse_dates=['updated'])
        df['year'] = year
        df['updated'] = pd.to_datetime(df['updated'], utc=True)
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    if add_time_features:
        combined['updated_local'] = combined['updated'].dt.tz_convert(TORONTO_TZ)
        combined['hour'] = combined['updated_local'].dt.hour.astype('int16')
        combined['dow'] = combined['updated_local'].dt.day_name()
        combined['month'] = combined['updated_local'].dt.to_period('M').astype(str)
    return combined
