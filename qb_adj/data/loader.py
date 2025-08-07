# loader.py
import pandas as pd
import numpy as np

df_pff_ids = pd.read_csv("D:/NFL/qb_adj/data/QB_pff_ids.csv")
df_pff_ids["Player"] = df_pff_ids["Player"].str.title()

class DataLoader:
    def __init__(self):
        self.model_df = None
        self.games = None

        self.player_file_repl = {
            'LA': 'LAR',
            'LV': 'OAK',
        }
        self.games_file_repl = {
            'LA': 'LAR',
            'LV': 'OAK',
            'STL': 'LAR',
            'SD': 'LAC',
        }

        self.stat_cols = [
            'completions', 'attempts', 'passing_yards', 'passing_tds',
            'interceptions', 'sacks', 'carries', 'rushing_yards', 'rushing_tds'
        ]

        self.load_data()


    def iso_top_passer(self, df):
        ## So as not to update the rating of a QB who had few passes, only include
        ## the top passer ##
        ## however, if this player was not the starter, then we need to override ##
        ## add starter info ##
        ## this needs to be cleaned up -- i think the attempts are not relevant as we are just using starter
        df['is_starter'] = np.where(
            df['player_id'] == df['starter_id'],
            1,
            np.nan
        )
        return df.sort_values(
            by=['game_id', 'is_starter', 'attempts'],
            ascending=[True, False, False]
        ).groupby(['game_id', 'team']).head(1).reset_index(drop=True)

    def format_top_passer(self, df):
        ## add the start number to the top passer and get rid of unecessary fields ##
        ## note, since we arent pre-loading the existing CSV with data before 1999, this number ##
        ## is an approximation ##
        ## since we will eventually throw out data pre-2022, this is fine (probably) ##
        df['start_number'] = df.groupby(['player_id']).cumcount() + 1
        return df[[
            'game_id', 'season', 'week', 'gameday', 'team', 'opponent', 'player_id', 'player_name', 'player_display_name',
            'start_number', 'rookie_season', 'draft_year', 'draft_pick',
            'completions', 'attempts', 'passing_yards', 'passing_tds',
            'interceptions', 'sacks', 'carries', 'rushing_yards', 'rushing_tds'
        ]].copy()

    def load_data(self):
        stats_url = "https://github.com/nflverse/nflverse-data/releases/download/player_stats/player_stats.csv.gz"
        players_url = "https://github.com/nflverse/nflverse-data/releases/download/players/players.csv"
        games_url = "https://github.com/nflverse/nfldata/raw/master/data/games.csv"


        def add_missing_draft_data(df):
            ## load missing draft data ##
            missing_draft_data = 'D:/NFL/qb_adj/data/missing_draft_data.csv'

            missing_draft = pd.read_csv(
                missing_draft_data,
                index_col=0
            )
            ## groupby id to ensure no dupes ##

            missing_draft = missing_draft.groupby(['player_id']).head(1)
            ## rename the cols, which will fill if main in NA ##
            missing_draft = missing_draft.rename(columns={
                'rookie_year': 'rookie_season_fill',
                'draft_number': 'draft_pick_fill',
                'entry_year': 'draft_year_fill',
                'birth_date': 'birth_date_fill',
            })
            ## add to data ##
            df = pd.merge(
                df,
                missing_draft[[
                    'player_id', 'rookie_season_fill', 'draft_pick_fill',
                    'draft_year_fill', 'birth_date_fill'
                ]],
                on=['player_id'],
                how='left'
            )
            ## fill in missing data ##

            for col in [
                'rookie_season', 'draft_pick', 'draft_year', 'birth_date'
            ]:
                ## fill in missing data ##
                df[col] = df[col].combine_first(df[col + '_fill'])
                ## and then drop fill col ##
                df = df.drop(columns=[col + '_fill'])
            ## return ##
            return df


        stats = pd.read_csv(stats_url, compression='gzip',low_memory=False)
        stats = stats[stats['position'] == 'QB'].copy()
        stats['recent_team'] = stats['recent_team'].replace(self.player_file_repl)
        stats = stats.rename(columns={'recent_team': 'team',})

        players = pd.read_csv(players_url)
        players = players.groupby(['gsis_id']).head(1)

        players = players.rename(columns={"gsis_id": "player_id"})

        players = add_missing_draft_data(players)

        merged = stats.merge(players, on='player_id', how='left')

        games = pd.read_csv(games_url)
        games = games[games['season'] >= 2006]
        games['home_team'] = games['home_team'].replace(self.games_file_repl)
        games['away_team'] = games['away_team'].replace(self.games_file_repl)

        games['game_id'] = (
            games['season'].astype(str) + '_' +
            games['week'].astype(str).str.zfill(2) + '_' +
            games['away_team'] + '_' +
            games['home_team']
        )

        #games = games[['season', 'week', 'home_team', 'away_team', 'home_qb_id', 'away_qb_id',
                       #'home_qb_name', 'away_qb_name', 'gameday', 'game_id', 'temp', 'wind', 'result']]

        self.games = games.copy()


        game_flat = pd.concat([
                games[[
                    'game_id', 'gameday', 'season', 'week',
                    'home_team', 'away_team',
                    'home_qb_id', 'home_qb_name',
                    'away_qb_id', 'away_qb_name', 'temp', 'wind'
                ]].rename(columns={
                    'home_team': 'team',
                    'home_qb_id': 'starter_id',
                    'home_qb_name': 'starter_name',
                    'away_team': 'opponent',
                    'away_qb_id': 'opponent_starter_id',
                    'away_qb_name': 'opponent_starter_name'
                }),
                games[[
                    'game_id', 'gameday', 'season', 'week',
                    'home_team', 'away_team',
                    'home_qb_id', 'home_qb_name',
                    'away_qb_id', 'away_qb_name', 'temp', 'wind'
                ]].rename(columns={
                    'away_team': 'team',
                    'away_qb_id': 'starter_id',
                    'away_qb_name': 'starter_name',
                    'home_team': 'opponent',
                    'home_qb_id': 'opponent_starter_id',
                    'home_qb_name': 'opponent_starter_name',
                })
            ])

        model = merged.merge(game_flat, on=['season', 'week', 'team'], how='left')

        df_team = model.groupby(['game_id', 'season', 'week', 'gameday','team']).agg(
            completions=('completions', 'sum'),
            attempts=('attempts', 'sum'),
            passing_yards=('passing_yards', 'sum'),
            passing_tds=('passing_tds', 'sum'),
            interceptions=('interceptions', 'sum'),
            sacks=('sacks', 'sum'),
            carries=('carries', 'sum'),
            rushing_yards=('rushing_yards', 'sum'),
            rushing_tds=('rushing_tds', 'sum'),
            temp=('temp','first'),
            wind=('wind', 'first')
        ).reset_index().rename(columns={
            'team': 'team'
        })
        df_team['team_VALUE'] = (
            -2.2 * df_team['attempts'] + 3.7 * df_team['completions'] +
            (df_team['passing_yards'] / 5) + 11.3 * df_team['passing_tds'] -
            14.1 * df_team['interceptions'] - 8 * df_team['sacks'] -
            1.1 * df_team['carries'] + 0.6 * df_team['rushing_yards'] +
            15.9 * df_team['rushing_tds']
        )


        model = self.iso_top_passer(model)
        model = self.format_top_passer(model)

        model['start_number'] = model.groupby('player_id').cumcount() + 1
        df_team = df_team.drop(columns=self.stat_cols)



        model['player_VALUE'] = (
            -2.2 * model['attempts'] + 3.7 * model['completions'] +
            (model['passing_yards'] / 5) + 11.3 * model['passing_tds'] -
            14.1 * model['interceptions'] - 8 * model['sacks'] -
            1.1 * model['carries'] + 0.6 * model['rushing_yards'] +
            15.9 * model['rushing_tds']
        )


        model = pd.merge(
            model,
            df_team[['game_id', 'team', 'team_VALUE', 'temp', 'wind']],
            on=['game_id', 'team'],
            how='left'
        )
        model = pd.merge(left=model, right=df_pff_ids, how="left", left_on="player_display_name", right_on='Player').drop(
            columns=["Player", "Pos", "Team"])

        model["player_id"] = model["PFF Ref"].copy()
        self.model_df = model.copy()

        model.to_csv("D:/NFL/qb_adj/data/model_file_test.csv")
        games.to_csv("D:/NFL/qb_adj/data/Nolan_test.csv")
