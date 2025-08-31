# features/time_advantage.py
def define_time_advantages(game_df, timezones, overrides):
    import pandas as pd
    import numpy as np

    def apply_tz_overrides(row):
        for side in ['home', 'away']:
            override = overrides.get(row[f'{side}_team'])
            if override and row['season'] <= override['season']:
                row[f'{side}_tz'] = override['tz_override']
        return row

    temp_df = game_df.copy()
    peak_time = '17:00'
    temp_df['home_tz'] = temp_df['home_team'].replace(timezones).fillna('ET')
    temp_df['away_tz'] = temp_df['away_team'].replace(timezones).fillna('ET')
    temp_df = temp_df.apply(apply_tz_overrides, axis=1)

    temp_df['home_opt_et'] = pd.Timestamp(peak_time)
    temp_df['away_opt_et'] = pd.Timestamp(peak_time)

    def shift_time(tz):
        return {
            'ET': 0,
            'CT': 1,
            'MT': 2,
            'PT': 3
        }.get(tz, 0)

    temp_df['home_opt_et'] += pd.to_timedelta(temp_df['home_tz'].map(shift_time), unit='h')
    temp_df['away_opt_et'] += pd.to_timedelta(temp_df['away_tz'].map(shift_time), unit='h')

    temp_df['gametime_obj'] = pd.to_datetime(temp_df['gametime'], format='%H:%M').dt.time
    temp_df['home_time_advantage'] = (
        np.abs(pd.to_datetime(temp_df['gametime_obj'].astype(str)) - pd.to_datetime(temp_df['away_opt_et'].dt.time.astype(str))) -
        np.abs(pd.to_datetime(temp_df['gametime_obj'].astype(str)) - pd.to_datetime(temp_df['home_opt_et'].dt.time.astype(str)))
    ).dt.total_seconds() / 3600

    return temp_df['home_time_advantage'].fillna(0)