# data/preprocess.py

import pandas as pd
from config.settings import team_standardization

def standardize_teams(df):
    df["home_team"] = df["home_team"].replace(team_standardization)
    df["away_team"] = df["away_team"].replace(team_standardization)
    return df

def merge_game_features(schedule_df, qb_df, wepa_df):
    # Placeholder merge logic – extend with actual logic
    merged_df = pd.merge(schedule_df, qb_df, how='left', left_on=['season', 'home_team'], right_on=['season', 'team1'])
    merged_df = pd.merge(merged_df, wepa_df, on="game_id", how='left')
    return merged_df

def generate_result_column(df):
    df["home_margin"] = df["home_score"] - df["away_score"]
    df["away_margin"] = df["away_score"] - df["home_score"]
    return df
