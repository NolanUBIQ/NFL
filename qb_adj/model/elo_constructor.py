
import pandas as pd
import numpy as np

class EloConstructor:
    def __init__(self, games, qb_model, at_wrapper, export_loc):
        self.games = games.copy()
        self.qb_model = qb_model
        self.at_wrapper = at_wrapper
        self.export_loc = export_loc
        self.qb_values = pd.DataFrame(qb_model.data)
        self.new_games = None
        self.next_games = None
        self.new_file_games = None

    def determine_new_games(self):
        self.new_games = self.games[
            (self.games['gameday'] > '2023-02-12') &
            (~pd.isnull(self.games['result']))
        ].copy()

    def add_qbs_to_new_games(self):
        ## combine model_df, which is flat, with new games ##
        ## elo file is not flat ##
        ## if new games is none, update ##
        if self.new_games is None:
            self.determine_new_games()
        ## if there have been no new games, return without updating ##
        if self.new_games is None:
            return
        ## add home qb ##

        self.qb_values.to_csv("qb_values.csv")

        self.new_games = pd.merge(
            self.new_games,
            self.qb_values[[
                'game_id', 'team', 'player_id', 'player_display_name',
                'qb_value_pre', 'qb_adj', 'player_VALUE_adj', 'qb_value_post'
            ]].rename(columns={
                'team': 'home_team',
                'player_id': 'qb1_id',
                'player_display_name': 'qb1',
                'qb_value_pre': 'qb1_value_pre',
                'qb_adj': 'qb1_adj',
                'player_VALUE_adj': 'qb1_game_value',
                'qb_value_post': 'qb1_value_post'
            }),
            on=['game_id', 'home_team'],
            how='left'
        )
        #print(self.qb_values.columns)
        ## add away qb ##
        self.new_games = pd.merge(
            self.new_games,
            self.qb_values[[
                'game_id', 'team', 'player_id', 'player_display_name',
                'qb_value_pre', 'qb_adj', 'player_VALUE_adj', 'qb_value_post'
            ]].rename(columns={
                'team': 'away_team',
                'player_id': 'qb2_id',
                'player_display_name': 'qb2',
                'qb_value_pre': 'qb2_value_pre',
                'qb_adj': 'qb2_adj',
                'player_VALUE_adj': 'qb2_game_value',
                'qb_value_post': 'qb2_value_post'
            }),
            on=['game_id', 'away_team'],
            how='left'
        )



    def get_next_games(self):
        unplayed = self.games[pd.isnull(self.games['result'])].copy()
        if len(unplayed) == 0:
            return
        current_week = unplayed.iloc[0]['week']
        self.next_games = unplayed[unplayed['week'] == current_week].copy()

    def add_starters(self):
        self.at_wrapper.pull_current_starters()
        starters = self.at_wrapper.starters_df

        starter_map = {}
        for _, row in starters.iterrows():
            starter_map[row['team']] = {
                'qb_id': row['player_id'],
                'qb_name': row['player_display_name'],
                'draft_pick': row['draft_pick']
            }

        def apply(row):
            row['qb1_id'] = starter_map[row['home_team']]['qb_id']
            row['qb2_id'] = starter_map[row['away_team']]['qb_id']
            row['qb1'] = starter_map[row['home_team']]['qb_name']
            row['qb2'] = starter_map[row['away_team']]['qb_name']
            row['qb1_value_pre'] = self.qb_model.get_qb_value({
                'player_id': row['qb1_id'],
                'season': row['season'],
                'team': row['home_team'],
                'draft_pick': starter_map[row['home_team']]['draft_pick'],
                'gameday': row['gameday']
            })
            row['qb2_value_pre'] = self.qb_model.get_qb_value({
                'player_id': row['qb2_id'],
                'season': row['season'],
                'team': row['away_team'],
                'draft_pick': starter_map[row['away_team']]['draft_pick'],
                'gameday': row['gameday']
            })
            return row

        self.next_games = self.next_games.apply(apply, axis=1)

    def add_team_values(self):
        def team_vals(row):
            home_val, home_adj = self.qb_model.get_team_off_value(row['home_team'], row['qb1_value_pre'], row['season'])
            away_val, away_adj = self.qb_model.get_team_off_value(row['away_team'], row['qb2_value_pre'], row['season'])
            row['qb1_adj'] = home_adj
            row['qb2_adj'] = away_adj
            return row

        self.next_games = self.next_games.apply(team_vals, axis=1)

    def merge_new_and_next(self):
        if self.new_games is not None and self.next_games is not None:


            next_games_padded = self.next_games.reindex(columns=self.new_games.columns)

            # Concatenate the two
            self.new_file_games = pd.concat([self.new_games, next_games_padded], ignore_index=True)

            self.new_file_games["qb1_adj"] = self.new_file_games["qb1_adj"]*3.3
            self.new_file_games["qb2_adj"] = self.new_file_games["qb2_adj"] * 3.3
            self.new_file_games["qb1_value_pre"] = self.new_file_games["qb1_value_pre"] * 3.3
            self.new_file_games["qb2_value_pre"] = self.new_file_games["qb2_value_pre"] * 3.3
            self.new_file_games["qb1_value_post"] = self.new_file_games["qb1_value_post"] * 3.3
            self.new_file_games["qb2_value_post"] = self.new_file_games["qb2_value_post"] * 3.3

        elif self.new_games is not None:
            self.new_file_games = self.new_games
        elif self.next_games is not None:
            self.new_file_games = self.next_games



    def export(self):
        if self.new_file_games is not None:
            self.new_file_games.to_csv(self.export_loc, index=False)
            print(f"Exported updated Elo file to: {self.export_loc}")
        else:
            print("No new data to export.")
