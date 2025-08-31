# data/load_data.py

import pandas as pd
import nfl_data_py as nfl
from config.settings import team_standardization


def load_schedules(seasons):
    df = nfl.import_schedules(seasons)
    df["home_team"] = df["home_team"].replace(team_standardization)
    df["away_team"] = df["away_team"].replace(team_standardization)

    df["game_id"] = (
        df["season"].astype(str) + "_" +
        df["week"].apply(lambda x: f"{int(x):02d}") + "_" +
        df["away_team"] + "_" +
        df["home_team"]
    )
    return df

def load_qb_elo(filepath):
    return pd.read_csv(filepath)

def load_wepa(filepath):
    df = pd.read_csv(filepath)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    return df
