from model.game_context import GameContext
from utils.s_curve import s_curve
from utils.progression import prog_disc
import numpy as np
import pandas as pd
import math
import time

class QBModel:
    def __init__(self, games, model_config):
        self.games = games
        self.config = model_config
        self.qbs = {}
        self.teams = {}
        self.team_avgs = {}
        self.season_avgs = {}
        self.data = []
        self.current_week = 1
        self.league_avg_def = self.config['init_value']
        self.model_runtime = 0
        self.chrono_sort()
        self.add_averages()

    def chrono_sort(self):
        self.games = self.games.sort_values(by=['season', 'week', 'game_id']).reset_index(drop=True)

    def init_team(self, team):
        self.teams[team] = {
            'off_value': self.config['init_value'],
            'def_value': self.config['init_value']
        }

    def add_averages(self):
        ## adds the avg QB values for teams and leagues which are used in reversion ##
        ## calc team averages ##
        team_avgs = self.games.groupby(
            ['season', 'team']
        )['team_VALUE'].mean().reset_index()
        ## calc league average ##
        season_avgs = self.games.groupby(
            ['season']
        )['team_VALUE'].mean().reset_index()
        ## write to stoarge ##
        for index, row in team_avgs.iterrows():
            self.team_avgs['{0}{1}'.format(row['season'], row['team'])] = row['team_VALUE']
        for index, row in season_avgs.iterrows():
            self.season_avgs[row['season']] = row['team_VALUE']

    def get_team_def_value(self, team, week):
        if self.current_week != week:
            self.update_league_avg_def()
            self.current_week = week
        if team not in self.teams:
            self.init_team(team)
        if week == 1:
            self.teams[team]['def_value'] = (
                (1 - self.config['team_def_reversion']) * self.teams[team]['def_value']
            )
        val = self.teams[team]['def_value']
        return val, val - self.league_avg_def

    def update_league_avg_def(self):
        self.league_avg_def = np.mean([team['def_value'] for team in self.teams.values()])

    def update_team_def_value(self, team, value):
        self.teams[team]['def_value'] = (
            self.config['team_def_sf'] * value +
            (1 - self.config['team_def_sf']) * self.teams[team]['def_value']
        )

    def get_team_off_value(self, team, qb_val, season):
        if team not in self.teams:
            self.init_team(team)
        if self.current_week == 1:
            self.teams[team]['off_value'] = (
                (1 - self.config['team_off_league_reversion'] - self.config['team_off_qb_reversion']) * self.teams[team]['off_value'] +
                self.config['team_off_qb_reversion'] * qb_val +
                self.config['team_off_league_reversion'] * self.get_prev_season_league_avg(season)
            )
        off_val = self.teams[team]['off_value']
        return off_val, qb_val - off_val

    def update_team_off_value(self, team, value):
        self.teams[team]['off_value'] = (
            self.config['team_off_sf'] * value +
            (1 - self.config['team_off_sf']) * self.teams[team]['off_value']
        )

    def init_qb(self, qb_id, season, team, draft_number, gameday):
        if pd.isnull(draft_number):
            draft_number = self.config['rookie_undrafted_draft_number']
        else:
            draft_number = draft_number[0] if isinstance(draft_number, list) else draft_number

        prev_season_team_avg = self.get_prev_season_team_avg(season, team)
        prev_season_league_avg = self.get_prev_season_league_avg(season)

        val = min(
            self.config['rookie_draft_intercept'] + self.config['rookie_draft_slope'] * math.log(draft_number) +
            ((1 - self.config['rookie_league_reg']) * prev_season_team_avg + self.config['rookie_league_reg'] * prev_season_league_avg),
            (1 + self.config['rookie_league_cap']) * prev_season_league_avg
        )
        self.qbs[qb_id] = {
            'current_value': val,
            'current_variance': val,
            'rolling_value': val,
            'starts': 0,
            'season_starts': 0,
            'first_game_date': gameday,
            'first_game_season': season,
            'last_game_date': None,
            'last_game_season': None
        }

    def get_qb_value(self, row):
        qb_id = row['player_id']
        if qb_id not in self.qbs:
            print(row)
            self.init_qb(qb_id, row['season'], row['team'], row['draft_pick'], row['gameday'])
        qb = self.qbs[qb_id]
        if qb['last_game_season'] is not None and row['season'] > qb['last_game_season']:
            qb = self.handle_qb_regression(qb, row['season'])

            qb['season_starts'] = 0
        return qb['current_value']

    def update_qb_value(self, qb_id, value, proj_value, gameday, season, team):
        qb = self.qbs[qb_id]
        old_val = qb['current_value']
        adj_val = prog_disc(value, proj_value, 15, self.config['player_prog_disc_alpha'])
        qb['current_value'] = self.config['player_sf'] * adj_val + (1 - self.config['player_sf']) * qb['current_value']
        sf = self.config['player_career_sf_base'] + s_curve(self.config['player_career_sf_height'], self.config['player_career_sf_mp'], qb['starts'], 'down')
        qb['rolling_value'] = sf * adj_val + (1 - sf) * qb['rolling_value']
        qb['current_variance'] = (
            self.config['player_sf'] * (value - old_val) * (value - qb['current_value']) +
            (1 - self.config['player_sf']) * qb['current_variance']
        )
        qb['starts'] += 1
        qb['season_starts'] += 1
        qb['last_game_date'] = gameday
        qb['last_game_season'] = season
        self.qbs[qb_id] = qb

    def run_model(self):
        start_time = time.time()
        for _, row in self.games.iterrows():
            qb_val = self.get_qb_value(row)
            context = GameContext(row['game_id'], self.config, row['temp'], row['wind'])
            weather_adj = context.weather_adj()
            def_val, def_adj = self.get_team_def_value(row['opponent'], row['week'])
            off_val, qb_adj = self.get_team_off_value(row['team'], qb_val, row['season'])

            expected_val = qb_val - def_val + weather_adj

            def_adj_perf = row['player_VALUE'] + def_val - weather_adj

            self.update_qb_value(row['player_id'], def_adj_perf, expected_val, row['gameday'], row['season'], row['team'])


            self.update_team_def_value(row['opponent'], qb_val - (row['player_VALUE'] - weather_adj))



            self.update_team_off_value(row['team'], def_adj_perf)

            # if row["game_id"]=='2024_20_WAS_DET':
            #     print(f"player_VALUE is {row['player_VALUE']}")
            #     print(f"def_val is {def_val}")
            #     print(f"weather_adj is {weather_adj}")



            row['qb_value_pre'] = qb_val
            row['team_value_pre'] = off_val
            row['qb_adj'] = qb_adj
            row['opponent_def_value_pre'] = def_val
            row['opponent_def_adj'] = def_adj
            row['player_VALUE_adj'] = def_adj_perf
            row['qb_value_post'] = self.qbs[row['player_id']]['current_value']
            row['team_value_post'] = self.teams[row['team']]['off_value']
            row['opponent_def_value_post'] = self.teams[row['opponent']]['def_value']
            self.data.append(row)
        self.model_runtime = time.time() - start_time

    def score_model(self):
        df = pd.DataFrame(self.data)
        df['se'] = (df['qb_value_pre'] - df['player_VALUE_adj']) ** 2
        df['abs_error'] = np.abs(df['qb_value_pre'] - df['player_VALUE_adj'])
        record = self.config.copy()
        record['rmse'] = df['se'].mean() ** 0.5
        record['mae'] = df['abs_error'].mean()
        record['model_runtime'] = self.model_runtime
        return record

    def handle_qb_regression(self, qb, season):
        ## regress qb to the league average ##
        ## first, get the previous season average ##
        prev_season_league_avg = self.get_prev_season_league_avg(season)
        ## determine regression amounts based on model curves ##
        league_regression = s_curve(
            self.config['player_regression_league_height'],
            self.config['player_regression_league_mp'],
            qb['starts'],
            'down'
        )
        career_regression = s_curve(
            self.config['player_regression_career_height'],
            self.config['player_regression_career_mp'],
            qb['starts'],
            'up'
        )
        ## calculate the new value ##
        ## if the qb didnt play much the previous (ie was a backup) this is ##
        ## signal that they are not league average quality ##
        ## In this case, we discount the league average regression portion ##
        league_regression = (
                league_regression *
                s_curve(
                    1,
                    4,
                    qb['season_starts'],
                    'up'
                )
        )
        ## normalize the combined career and league regression to not exceed 100% ##
        total_regression = league_regression + career_regression
        if total_regression > 1:
            league_regression = league_regression / total_regression
            career_regression = career_regression / total_regression
        ## calculate value ##
        qb['current_value'] = (
                (1 - league_regression - career_regression) * qb['current_value'] +
                (league_regression * prev_season_league_avg) +
                (career_regression * qb['rolling_value'])
        )
        ## update season ##
        ## return the qb object ##
        return qb

    def get_prev_season_league_avg(self, season):
        ## get the leagues previous season average while controlling for errors ##
        ## the first season will not have a pervous season ##
        return self.season_avgs.get(season - 1, self.config['init_value'])

    def get_prev_season_team_avg(self, season, team):
        ## get the teams previous season average while controlling for errors ##
        ## the first season will not have a pervous season ##
        return self.team_avgs.get('{0}{1}'.format(
            season - 1, team
        ), self.config['init_value'])
