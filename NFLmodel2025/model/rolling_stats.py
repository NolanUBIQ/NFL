# model/rolling_stats.py

import pandas as pd

def add_game_numbers(df):
    df = df.sort_values(by=['team', 'game_id'])
    df['all_time_game_number'] = df.groupby(['team']).cumcount() + 1
    df['season_game_number'] = df.groupby(['team', 'season']).cumcount() + 1
    return df

def apply_rolling_epa(df, slope, intercept):
    df = df.copy()
    df['wepa_margin'] = slope * df['net_wepa'] + intercept
    return df
