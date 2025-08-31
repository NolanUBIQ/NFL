# features/surface.py
import pandas as pd
import datetime
def define_field_surfaces(game_df, surface_repl):

    temp_df = game_df.copy()
    temp_df = temp_df[(temp_df['neutral'] != 1) & (~pd.isnull(temp_df['home_score']))].copy()
    temp_df['surface'] = temp_df['surface'].replace(surface_repl)

    fields_df = temp_df.groupby(['home_team', 'season', 'surface']).agg(games_played=('home_score', 'count')).reset_index()
    fields_df = fields_df.sort_values(by='games_played', ascending=False).reset_index(drop=True)
    fields_df = fields_df.groupby(['home_team', 'season']).head(1)

    last_season = int(temp_df['season'].max())
    last_week = temp_df[temp_df['season'] == last_season]['week'].max()
    curr_month = datetime.datetime.now().month
    if 4 <= curr_month <= 9 and last_week > 1:
        last_season += 1

    all_team_season = [{'team': team, 'season': season} for season in range(int(temp_df['season'].min()), last_season+1) for team in temp_df['home_team'].unique()]
    all_team_season_df = pd.DataFrame(all_team_season)

    all_team_season_df = pd.merge(all_team_season_df, fields_df.rename(columns={'home_team': 'team'}), on=['team', 'season'], how='left')
    all_team_season_df = all_team_season_df.sort_values(by=['team', 'season']).reset_index(drop=True)
    all_team_season_df['surface'] = all_team_season_df.groupby('team')['surface'].transform(lambda x: x.bfill().ffill())
    return all_team_season_df.drop_duplicates()
