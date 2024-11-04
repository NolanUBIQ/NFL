
import pandas as pd
import numpy as np
import math
import statistics
import pathlib
import json
import json
import statsmodels.api as sm
import datetime
import scipy
import os
import pickle
import nfl_data_py as nfl

os.chdir('D:\\NFL\\NFL Beyond')

team_standardization = {
    "ARI": "ARI",
    "ATL": "ATL",
    "BAL": "BAL",
    "BUF": "BUF",
    "CAR": "CAR",
    "CHI": "CHI",
    "CIN": "CIN",
    "CLE": "CLE",
    "DAL": "DAL",
    "DEN": "DEN",
    "DET": "DET",
    "GB": "GB",
    "HOU": "HOU",
    "IND": "IND",
    "JAC": "JAX",
    "JAX": "JAX",
    "KC": "KC",
    "LA": "LAR",
    "LAC": "LAC",
    "LV": "OAK",
    "MIA": "MIA",
    "MIN": "MIN",
    "NE": "NE",
    "NO": "NO",
    "NYG": "NYG",
    "NYJ": "NYJ",
    "OAK": "OAK",
    "PHI": "PHI",
    "PIT": "PIT",
    "SD": "LAC",
    "SEA": "SEA",
    "SF": "SF",
    "STL": "LAR",
    "TB": "TB",
    "TEN": "TEN",
    "WAS": "WAS"
}

surface_repl = {
    "fieldturf": "artificial",
    "fieldturf ": "artificial",
    "matrixturf": "artificial",
    "sportturf": "artificial",
    "astroturf": "artificial",
    "astroplay": "artificial",
    "a_turf": "artificial",
    "grass": "natural",
    "dessograss": "natural"
}
pbp_surface_repl= {
        "fieldturf" : "artificial",
        "fieldturf " : "artificial",
        "matrixturf" : "artificial",
        "sportturf" : "artificial",
        "astroturf" : "artificial",
        "astroplay" : "artificial",
        "a_turf" : "artificial",
        "grass" : "natural",
        "dessograss" : "natural"
      }



timezones = {
    "ARI": "MT",
    "ATL": "ET",
    "BAL": "ET",
    "BUF": "ET",
    "CAR": "ET",
    "CHI": "CT",
    "CIN": "ET",
    "CLE": "ET",
    "DAL": "CT",
    "DEN": "MT",
    "DET": "ET",
    "GB": "CT",
    "HOU": "CT",
    "IND": "ET",
    "JAX": "ET",
    "KC": "CT",
    "LAR": "PT",
    "LAC": "PT",
    "MIA": "ET",
    "MIN": "CT",
    "NE": "ET",
    "NO": "CT",
    "NYG": "ET",
    "NYJ": "ET",
    "OAK": "PT",
    "PHI": "ET",
    "PIT": "ET",
    "SEA": "PT",
    "SF": "PT",
    "TB": "ET",
    "TEN": "CT",
    "WAS": "ET"
}

timezone_overrides = {
    "LAR": {
        "season": 2015,
        "tz_override": "CT"
    }}

game_headers = [
    "game_id",
    "type",
    "season",
    "week",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "gameday",
    "weekday",
    "stadium",
    "stadium_id",
    "roof",
    "surface",
    "temp",
    "wind",
    "away_moneyline",
    "home_moneyline",
    "away_spread_odds",
    "home_spread_odds",
    "gametime",
    "old_game_id"
]

game_repl = {
    "gameday": "game_date",
    "weekday": "game_day",
    "temp": "temperature"
}

final_headers = [
    "game_id",
    "type",
    "season",
    "week",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "home_line_open",
    "home_line_close",
    "home_ats_pct",
    "game_date",
    "game_day",
    "stadium",
    "stadium_id",
    "roof",
    "surface",
    "temperature",
    "wind",
    "divisional_game",
    "neutral_field",
    "home_total_dvoa_begining",
    "home_total_dvoa",
    "home_projected_dvoa",
    "home_blended_dvoa_begining",
    "home_blended_dvoa",
    "home_overall_grade",
    "home_pff_point_margin",
    "away_total_dvoa_begining",
    "away_total_dvoa",
    "away_projected_dvoa",
    "away_blended_dvoa_begining",
    "away_blended_dvoa",
    "away_overall_grade",
    "away_pff_point_margin",
    "away_moneyline",
    "home_moneyline",
    "away_spread_odds",
    "home_spread_odds",
    "market_implied_elo_dif",
    "home_surface_advantage",
    "home_time_advantage",
    "home_temp_advantage",
    "old_game_id",
    "gametime"
]

manual_clean_dict = {
    "2009_17_IND_BUF": {"home_score": 30, "away_score": 7},
    "2013_07_CIN_DET": {"home_score": 24, "away_score": 27},
    "2015_06_ARI_PIT": {"home_score": 25, "away_score": 13},
    "2015_09_PHI_DAL": {"home_score": 27, "away_score": 33},
    "2015_15_KC_BAL": {"home_score": 14, "away_score": 34},
    "2016_01_MIN_TEN": {"home_score": 16, "away_score": 25},
    "2016_05_NE_CLE": {"home_score": 13, "away_score": 33}
}
pbp_timezones = {
        "ARI" : "MT",
        "ATL" : "ET",
        "BAL" : "ET",
        "BUF" : "ET",
        "CAR" : "ET",
        "CHI" : "CT",
        "CIN" : "ET",
        "CLE" : "ET",
        "DAL" : "CT",
        "DEN" : "MT",
        "DET" : "ET",
        "GB" : "CT",
        "HOU" : "CT",
        "IND" : "ET",
        "JAX" : "ET",
        "KC" : "CT",
        "LAR" : "PT",
        "LAC" : "PT",
        "MIA" : "ET",
        "MIN" : "CT",
        "NE" : "ET",
        "NO" : "CT",
        "NYG" : "ET",
        "NYJ" : "ET",
        "OAK" : "PT",
        "PHI" : "ET",
        "PIT" : "ET",
        "SEA" : "PT",
        "SF" : "PT",
        "TB" : "ET",
        "TEN" : "CT",
        "WAS" : "ET"
      }
pbp_timezone_overrides = {
        "LAR" : {
          "season" : 2015,
          "tz_override" : "CT"
        }}



level_weeks=10
reg_weeks = 140
kick_in= 0.75

current_week = 9
current_season=2024


def calc_rolling_hfa(current_df, level_weeks, reg_weeks):
    print('     Calculating rolling HFA...')
    hfa_df = current_df.copy()
    hfa_df['expected_result'] = hfa_df['home_rating'] - hfa_df['away_rating']

    hfa_df['home_margin_error'] = (
            hfa_df['result'] -
            hfa_df['expected_result']
    )

    hfa_df_temp=hfa_df[
            (~pd.isnull(hfa_df['expected_result'])) &
            (hfa_df['season'] != 2020) &
            (hfa_df['game_type'] == 'REG') &
            (hfa_df['location'] != 'Neutral')
        ].copy().reset_index(drop=True)


    ## agg on weeks for smoothing ##
    hfa_df_temp = hfa_df_temp.groupby(['season', 'week']).agg(
            avg_error = ('home_margin_error', 'mean')
        ).reset_index()
    ## to avoid overweighting playoffs, they are removed from regression ##
    ## we also remove COVID season, which skews data ##



    ## ema init and congig ##
    a = 2 / (level_weeks + 1)
    hfa_df_temp['level'] = 2.50
    ## regression init and config ##
    hfa_df_temp['intercept_constant'] = 1
    for index, row in hfa_df_temp.iterrows():
        if index < kick_in * reg_weeks or index < level_weeks:
            pass
        else:
            ## REGRESSION ##
            ## window data ##
            window_start = int(max(index - reg_weeks, 0))
            a_ = a * min(index / reg_weeks, 1)
            trailing_window = hfa_df_temp.iloc[
                              window_start:index
                              ].copy()
            trailing_window['week_num'] = np.arange(
                len(trailing_window)
            ) + 1
            ## fit ##
            reg = sm.OLS(
                trailing_window['avg_error'],
                trailing_window[['week_num', 'intercept_constant']],
                hasconst=True
            ).fit()
            ## get update value ##
            update_val = (
                    reg.params.intercept_constant +
                    (
                            trailing_window['week_num'].max() *
                            reg.params.week_num
                    )
            )
            ## EMA ##
            ## get previous value ##
            prev_level = hfa_df_temp.iloc[index - 1]['level']
            ## update prev week value ##
            hfa_df_temp.loc[index, 'level'] = (
                    a_ * update_val +
                    (1 - a_) * prev_level
            )
            ## note, this level is *end of week* and needs to be shifted forward ##
    ## shift updated value as next weeks forcast ##
    ## weeks w/ not enough data in trailing window get 2.50 starting point ##
    hfa_df_temp['rolling_hfa'] = hfa_df_temp['level'].shift(1).fillna(2.50)
    ## add back to HFA df ##
    hfa_df = pd.merge(
        hfa_df[['season', 'week']],
        hfa_df_temp[['season', 'week', 'rolling_hfa']],
        on=['season', 'week'],
        how='left'
    )
    ## fill 2020 with 0.25 for COVID ##
    hfa_df['rolling_hfa'] = np.where(
        hfa_df['season'] == 2020,
        0.25,
        hfa_df['rolling_hfa']
    )
    ## then forward fill forecasts for playoffs ##
    hfa_df['rolling_hfa'] = hfa_df['rolling_hfa'].ffill()
    hfa_df['rolling_hfa'] = hfa_df['rolling_hfa'].fillna(2.50)

    ## then return ##
    return hfa_df

def define_field_surfaces(game_df, surface_repl):
    ## copy frame ##
    temp_df = game_df.copy()
    ## remove neutrals ##
    temp_df = temp_df[
        (temp_df['neutral'] != 1) &
        (~pd.isnull(temp_df['home_score']))
    ].copy()
    ## standardize turf types ##
    temp_df['surface'] = temp_df['surface'].replace(surface_repl)
    ## generate df of field types by team ##
    fields_df = temp_df.groupby(
        ['home_team', 'season', 'surface']
    ).agg(
        games_played = ('home_score', 'count'),
    ).reset_index()
    ## get most played surface ##
    fields_df = fields_df.sort_values(
        by=['games_played'],
        ascending=[False]
    ).reset_index(drop=True)
    fields_df = fields_df.groupby(
        ['home_team', 'season']
    ).head(1)
    ## create new struc for handling start of season where team may not have a home game yet ##
    last_season = int(temp_df['season'].max())
    last_week = temp_df[
        temp_df['season'] == last_season
    ]['week'].max()
    curr_month = datetime.datetime.now().month
    ## if it's before week 1, increment current season ##
    if curr_month <= 9 and curr_month >= 4 and last_week > 1:
        last_season += 1
    else:
        pass
    ## new struc containing every team and season ##
    all_team_season_struc = []
    for season in range(int(temp_df['season'].min()), last_season + 1):
        for team in temp_df['home_team'].unique().tolist():
            all_team_season_struc.append({
                'team' : team,
                'season' : season
            })
    all_team_season_df = pd.DataFrame(all_team_season_struc)
    ## add fields ##
    all_team_season_df = pd.merge(
        all_team_season_df,
        fields_df[[
            'home_team', 'season', 'surface'
        ]].rename(columns={
            'home_team' : 'team'
        }),
        on=['team', 'season'],
        how='left'
    )
    ## fill missing ##
    all_team_season_df = all_team_season_df.sort_values(
        by=['team', 'season'],
        ascending=[True, True]
    ).reset_index(drop=True)
    all_team_season_df['surface'] = all_team_season_df.groupby(
        ['team']
    )['surface'].transform(lambda x: x.bfill().ffill())
    ## eliminate any possibility of duping on eventual merge w/ unique records ##
    all_team_season_df = all_team_season_df.drop_duplicates()
    return all_team_season_df

def define_time_advantages(game_df, timezones, timezone_overrides):
    ## helper to apply overrides ##
    def apply_tz_overrides(row, timezone_overrides):
        home_overide = None
        away_overide = None
        ## try to load overrides ##
        try:
            home_overide = timezone_overrides[row['home_team']]
        except:
            pass
        try:
            away_overide = timezone_overrides[row['away_team']]
        except:
            pass
        ## apply override if applicable ##
        ## home ##
        if home_overide is None:
            pass
        elif row['season'] <= home_overide['season']:
            row['home_tz'] = home_overide['tz_override']
        else:
            pass
        ## away ##
        if away_overide is None:
            pass
        elif row['season'] <= away_overide['season']:
            row['away_tz'] = away_overide['tz_override']
        else:
            pass
        return row
    ## copy frame ##
    temp_df = game_df.copy()
    peak_time = '17:00'
    ## add time zones ##
    temp_df['home_tz'] = temp_df['home_team'].replace(timezones).fillna('ET')
    temp_df['away_tz'] = temp_df['away_team'].replace(timezones).fillna('ET')
    ## apply overrides ##
    temp_df = temp_df.apply(
        apply_tz_overrides,
        timezone_overrides=timezone_overrides,
        axis=1
    )
    ## define optimals in ET ##
    temp_df['home_optimal_in_et'] = pd.Timestamp(peak_time)
    temp_df['away_optimal_in_et'] = pd.Timestamp(peak_time)
    ## home ##
    temp_df['home_optimal_in_et'] = np.where(
        temp_df['home_tz'] == 'ET',
        temp_df['home_optimal_in_et'].dt.time,
        np.where(
            temp_df['home_tz'] == 'CT',
            (temp_df['home_optimal_in_et'] + pd.Timedelta(hours=1)).dt.time,
            np.where(
                temp_df['home_tz'] == 'MT',
                (temp_df['home_optimal_in_et'] + pd.Timedelta(hours=2)).dt.time,
                np.where(
                    temp_df['home_tz'] == 'PT',
                    (temp_df['home_optimal_in_et'] + pd.Timedelta(hours=3)).dt.time,
                    temp_df['home_optimal_in_et'].dt.time
                )
            )
        )
    )
    ## away ##
    temp_df['away_optimal_in_et'] = np.where(
        temp_df['away_tz'] == 'ET',
        temp_df['away_optimal_in_et'].dt.time,
        np.where(
            temp_df['away_tz'] == 'CT',
            (temp_df['away_optimal_in_et'] + pd.Timedelta(hours=1)).dt.time,
            np.where(
                temp_df['away_tz'] == 'MT',
                (temp_df['away_optimal_in_et'] + pd.Timedelta(hours=2)).dt.time,
                np.where(
                    temp_df['away_tz'] == 'PT',
                    (temp_df['away_optimal_in_et'] + pd.Timedelta(hours=3)).dt.time,
                    temp_df['away_optimal_in_et'].dt.time
                )
            )
        )
    )
    ## get kickoff ##
    temp_df['gametimestamp'] = pd.to_datetime(temp_df['gametime'], format='%H:%M').dt.time
    ## define advantage ##
    temp_df['home_time_advantage'] = np.round(
        np.absolute(
            (
                pd.to_datetime(temp_df['gametimestamp'], format='%H:%M:%S') -
                pd.to_datetime(temp_df['away_optimal_in_et'], format='%H:%M:%S')
            ) / np.timedelta64(1, 'h')
        ) -
        np.absolute(
            (
                pd.to_datetime(temp_df['gametimestamp'], format='%H:%M:%S') -
                pd.to_datetime(temp_df['home_optimal_in_et'], format='%H:%M:%S')
            ) / np.timedelta64(1, 'h')
        )
    )
    return temp_df['home_time_advantage'].fillna(0)

def shift_calc_helper(margin_measure, line_measure, line_market, config, is_home):
    ## establish line direction ##
    if is_home:
        line = line_measure * -1
        market_line = line_market * -1
    else:
        line = line_measure
        market_line = line_market
    ## establish k ##
    ## adjust teams more if vegas line was closer than nfelo line ##
    if abs(margin_measure - line) < 1 or (config['market_resist_factor'] == 0):
        adj_k_measure = config['k']
    elif abs(line - margin_measure) <= abs(market_line - margin_measure):
        adj_k_measure = config['k']
    else:
        adj_k_measure = config['k'] * (1 + (abs(market_line - line) / config['market_resist_factor']))
    ## create shift ##
    adj_pd_measure = abs(margin_measure - line)
    adj_mult_measure = math.log(max(adj_pd_measure, 1) + 1, config['b'])
    shift_measure = adj_k_measure * adj_mult_measure
    ## establish shift direction ##
    if margin_measure - line == 0:
        shift_measure = 0
    elif margin_measure - line > 0:
        shift_measure = shift_measure
    else:
        shift_measure = -1.0 * shift_measure
    return shift_measure

def create_data_struc(elo_game_df):
    print('          Creating data structure...')
    elo_dict = {}
    for index, row in elo_game_df.iterrows():
        ## create initial keys if necessary ##
        ## home ##
        if row['home_team'] in elo_dict:
            pass
        else:
            elo_dict[row['home_team']] = {}
        if row['away_team'] in elo_dict:
            pass
        else:
            elo_dict[row['away_team']] = {}
        ## attach structure ##
        ## home ##
        elo_dict[row['home_team']][row['game_id']] = {
            'starting' : None,
            'ending' : None,
            'week' : None,
            'rolling_nfelo_adj_start' : 0,
            'rolling_nfelo_adj_end' : 0,
            'rolling_model_se_start' : 0,
            'rolling_model_se_end' : 0,
            'rolling_market_se_start' : 0,
            'rolling_market_se_end' : 0,
        }
        ## away ##
        elo_dict[row['away_team']][row['game_id']] = {
            'starting' : None,
            'ending' : None,
            'week' : None,
            'rolling_nfelo_adj_start' : 0,
            'rolling_nfelo_adj_end' : 0,
            'rolling_model_se_start' : 0,
            'rolling_model_se_end' : 0,
            'rolling_market_se_start' : 0,
            'rolling_market_se_end' : 0,
        }
    return elo_dict

def normalize_probs(probs):
    return probs / probs.sum()


def calc_prob(spread, result, parameters):
    ## calc probability of a result given spread ##
    ## calculate baseline prob ##
    baseline_prob = scipy.stats.norm(spread, parameters[0]).pdf(result)

    ## add additional probs for key values ##
    additional_key_prob = 0
    for key in home_keys:
        if abs(spread - key) > 50:
            pass
        else:
            key_scaling = (parameters[2] * math.exp(-2 * ((abs(key - spread) / parameters[3]) ** parameters[4])))
            if math.isnan(key_scaling):
                key_scaling = 0
            additional_key_prob += key_scaling * scipy.stats.norm(key, parameters[5]).pdf(result)

    if (result < 0 and spread > 0) or (result > 0 and spread < 0):
        multiplier = 1 + (baseline_prob * parameters[6])
    else:
        multiplier = 1

    if result == 0:
        prob = parameters[7]
    else:
        prob = (baseline_prob * parameters[1] + additional_key_prob) * multiplier

    if math.isnan(float(prob)):
        return 0
    else:
        return float(prob)


def balance_probs(spread, parameters, possible_results):
    # Calculate probabilities for each possible result.
    probs = np.array([calc_prob(spread, result, parameters) for result in possible_results])

    # Iteratively adjust the distribution to balance total probabilities on each side of the spread
    tolerance = 1e-6

    # Normalize the probabilities
    probs = normalize_probs(probs)

    while True:

        lower_half_sum = probs[possible_results < spread].sum()
        upper_half_sum = probs[possible_results > spread].sum()
        difference = lower_half_sum - upper_half_sum

        if abs(difference) < tolerance:
            break

        elif difference > 0:
            adjustment = difference / 2 / len(probs[possible_results < spread])
            probs[possible_results < spread] -= adjustment
            probs[possible_results > spread] += adjustment

        else:
            adjustment = -difference / 2 / len(probs[possible_results > spread])
            probs[possible_results < spread] += adjustment
            probs[possible_results > spread] -= adjustment

    return probs

params = {'baseline_sdev': 13.7, 'baseline_weight': 1, 'key_scaling_height': 0.961, 'key_scaling_sdev': 17.503,
          'key_scaling_flatness': 0.602,
          'key_value_sdev': 0.746, 'opposite_sign_list:': 2.000, 'zero_value': 0.004}

def calculate_win_prob(spread, parameters=list(params.values()), possible_results=np.arange(-75, 76)):
    balanced_probs = balance_probs(spread, parameters, possible_results=np.arange(-75, 76))

    # Assume win is when result is greater than 0
    win_prob = balanced_probs[possible_results > 0].sum()

    return win_prob

def elo_to_prob(elo_dif:(int or float), z:(int or float)=400):
    '''
    Converts and elo difference to a win probability

    Parameters:
    * elo_dif (int or float): elo difference between two teams
    * z (int or float): config param that determines confidence

    Returns:
    * win_prob (float): the win probability implied by the elo dif
    '''
    ## handle potential div 0 ##
    if z <=0:
        raise Exception('NFELO CONFIG ERROR: Z must be greater than 0')
    ## return ##
    return 1 / (
        math.pow(
            10,
            (-elo_dif / z)
        ) +
        1
    )

def prob_to_elo(win_prob:(float), z:(int or float)=400):
    '''
    Converts a win probability to an elo difference. This is
    the inverse of elo_to_prob()

    Parameters:
    * win_prob (float): win probability of one team over another
    * z (int or float): config param that determines confidence

    Returns:
    * elo_dif (float): implied elo dif between the teams
    '''
    ## handle potential div 0 ##
    if z <=0:
        raise Exception('NFELO CONFIG ERROR: Z must be greater than 0')
    ## return the dif ##
    return (
        (-1 * z) *
        numpy.log10(
            (1/win_prob) -
            1
        )
    )

def calc_probs_favorite(proj_spread, market_spread, dist_df):
    ## flips signs of spreads ##
    proj_spread = -1 * round(proj_spread, 1)
    market_spread = -1 * round(market_spread, 1)
    ## get the probs for the nfelo projected spread ##
    temp_df = dist_df.copy()
    temp_df = temp_df[temp_df['spread_line'] == proj_spread]
    temp_df['loss_prob'] = np.where(
        temp_df['result'] < market_spread,
        temp_df['normalized_modeled_prob'],
        0
    )
    temp_df['push_prob'] = np.where(
        temp_df['result'] == market_spread,
        temp_df['normalized_modeled_prob'],
        0
    )
    temp_df['cover_prob'] = np.where(
        temp_df['result'] > market_spread,
        temp_df['normalized_modeled_prob'],
        0
    )

    temp_df['loss_prob'] = pd.to_numeric(temp_df['loss_prob'], errors='coerce')
    temp_df['push_prob'] = pd.to_numeric(temp_df['push_prob'], errors='coerce')
    temp_df['cover_prob'] = pd.to_numeric(temp_df['cover_prob'], errors='coerce')

    return [
        temp_df['loss_prob'].sum(),
        temp_df['push_prob'].sum(),
        temp_df['cover_prob'].sum()
    ]


def calc_probs_dog(proj_spread, market_spread, dist_df):
    ## flips signs of spreads ##
    proj_spread = -1 * round(proj_spread, 1)
    market_spread = -1 * round(market_spread, 1)
    ## get the probs for the nfelo projected spread ##
    temp_df = dist_df.copy()
    temp_df = temp_df[temp_df['spread_line'] == proj_spread]
    temp_df['loss_prob'] = np.where(
        temp_df['result'] < market_spread,
        temp_df['normalized_modeled_prob'],
        0
    )
    temp_df['push_prob'] = np.where(
        temp_df['result'] == market_spread,
        temp_df['normalized_modeled_prob'],
        0
    )
    temp_df['cover_prob'] = np.where(
        temp_df['result'] > market_spread,
        temp_df['normalized_modeled_prob'],
        0
    )

    temp_df['loss_prob'] = pd.to_numeric(temp_df['loss_prob'], errors='coerce')
    temp_df['push_prob'] = pd.to_numeric(temp_df['push_prob'], errors='coerce')
    temp_df['cover_prob'] = pd.to_numeric(temp_df['cover_prob'], errors='coerce')

    return [

        temp_df['loss_prob'].sum(),
        temp_df['push_prob'].sum(),
        temp_df['cover_prob'].sum()
    ]


def calc_rolling_info(merged_df, current_df, wepa_slope, wepa_intercept):
    ## calculate rolling information ##
    ## L16, L8, YTD ##
    print('     Calculating rolling information...')
    ## add game numbers ##
    ## keep a version w/ post season for elo ##
    merged_df_elo = merged_df.copy()
    merged_df = merged_df.sort_values(by=['team', 'game_id'])
    merged_df['all_time_game_number'] = merged_df.groupby(['team']).cumcount() + 1
    merged_df['season_game_number'] = merged_df.groupby(['team', 'season']).cumcount() + 1
    merged_df = merged_df.rename(columns={
        'wepa_net': 'net_wepa',
        'wepa': 'offensive_wepa',
        'd_wepa_against': 'defensive_wepa',
        'epa_net': 'net_epa',
        'epa': 'offensive_epa',
        'epa_against': 'defensive_epa',
    })
    merged_df['wepa_margin'] = wepa_slope * merged_df['net_wepa'] + wepa_intercept
    ## add wepa to current_file for game grade ##
    merge_df_gg_home = merged_df.copy()[['game_id', 'team', 'net_wepa']].rename(columns={
        'team': 'home_team',
        'net_wepa': 'home_net_wepa',
    })
    merge_df_gg_home['home_net_wepa_point_margin'] = wepa_slope * merge_df_gg_home['home_net_wepa'] + wepa_intercept
    merge_df_gg_away = merged_df.copy()[['game_id', 'team', 'net_wepa']].rename(columns={
        'team': 'away_team',
        'net_wepa': 'away_net_wepa',
    })
    merge_df_gg_away['away_net_wepa_point_margin'] = wepa_slope * merge_df_gg_away['away_net_wepa'] + wepa_intercept
    current_df = pd.merge(current_df, merge_df_gg_home, on=['game_id', 'home_team'], how='left')
    current_df = pd.merge(current_df, merge_df_gg_away, on=['game_id', 'away_team'], how='left')
    ## filter to only since 2009 and only played games ##
    current_df = current_df[
        (current_df['season'] >= 2009) &
        (~pd.isnull(current_df['home_score']))
        ]
    ## add windows ##
    merged_df = merged_df.apply(
        add_windows,
        window_length=rolling_window,
        merged_df=merged_df,
        axis=1
    )
    return merged_df, current_df, merged_df_elo

games_df=nfl.import_schedules([1999,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022,2023,2024])
#
games_df.to_csv("D:/NFL/NFL Beyond/games_df.csv",index=False)
# games_df=pd.read_csv("D:/NFL/NFL Beyond/games_df.csv")

games_df["home_team"]=games_df["home_team"].replace(team_standardization)
games_df["away_team"]=games_df["away_team"].replace(team_standardization)
games_df["game_id"]=games_df["season"].astype(str)+"_"+games_df["week"].apply(lambda x: f"{int(x):02d}")+"_"+games_df["away_team"]+"_"+games_df["home_team"]
# games_df=nfl.import_schedules([2022,2023])

games_df_2024 = games_df[games_df["season"]==2024]



games_df_2024_home_teams = games_df_2024[["game_id","home_team"]]


wepa_norm_params={"m":1.076534817725179, "b":0.18619338291338708 }

loc_538 = "D:/NFL/QB2023PreSeason"
filename_538 = "qb_elos.csv"
fivethirtyeight_df=pd.read_csv(f"{loc_538}/{filename_538}")

wepa_df=pd.read_csv("D:/NFL/NFL Beyond/data/pbp/wepa_game_flat_file.csv").iloc[:,1:]
wepa_df=pd.merge(left=wepa_df, right=games_df_2024_home_teams, on='game_id',how='left')


wepa_df["wepa_net_home"]= np.where(wepa_df["home_team"]==wepa_df["team"],wepa_df["wepa_net"],np.nan)
wepa_df["wepa_net_home"] = wepa_norm_params["m"]*wepa_df["wepa_net_home"]  + wepa_norm_params["b"]


wepa_df["wepa_net_away"]= np.where(wepa_df["home_team"]==wepa_df["team"],wepa_df["wepa_net_opponent"],np.nan)
wepa_df["wepa_net_away"] = wepa_norm_params["m"]*wepa_df["wepa_net_away"]  + wepa_norm_params["b"]

wepa_df = wepa_df.groupby(by='game_id').agg('max')[["wepa_net_home","wepa_net_away"]].reset_index()


wepa_df=pd.merge(left=wepa_df, right=games_df_2024_home_teams, on='game_id',how='left')

loc_pff = "D:/NFL/NFL Beyond/data/PFF/"


filename_pff="2019-2024-pff-with-sbr-spreads.xlsx"


pff_df = pd.read_excel(f"{loc_pff}{filename_pff}")[["GAME-ID","OVER_home","OVER_away"]]

pff_df.rename(columns={"GAME-ID":"game_id"},inplace=True)
pff_df['game_id'] = pff_df['game_id'].str.replace('-', '_')
pff_df['game_id'] = pff_df['game_id'].str.replace('WC', '19')
pff_df['game_id'] = pff_df['game_id'].str.replace('DP', '20')
pff_df['game_id'] = pff_df['game_id'].str.replace('CC', '21')
pff_df['game_id'] = pff_df['game_id'].str.replace('SB', '22')



# Split the game_id column into parts
pff_df[['year', 'week', 'team1', 'team2']] = pff_df['game_id'].str.split('_', expand=True)
pff_df['team1'] = pff_df['team1'].replace({"JAC":"JAX","LV":"OAK"})
pff_df['team2'] = pff_df['team2'].replace({"JAC":"JAX","LV":"OAK"})
# Format the week number to always have two digits
pff_df['week'] = pff_df['week'].apply(lambda x: f"{int(x):02d}")

# Rejoin the game_id column
pff_df['game_id'] = pff_df[['year', 'week', 'team1', 'team2']].apply('_'.join, axis=1)
pff_df = pff_df.drop(['year', 'week', 'team1', 'team2'], axis=1)



# Join on season and away team
merged_away = pd.merge(games_df,
                       fivethirtyeight_df[['season','date', 'team2', 'qb2', 'qb2_value_pre', 'qb2_game_value','qb2_adj']],
                       left_on=['gameday','away_team'],
                       right_on=['date', 'team2'],
                       how='left')

# Renaming QB columns after the join for the away team
merged_away.rename(columns={'qb2': 'away_qb_538', 'qb2_value_pre': 'away_qb_value_pre_538', 'qb2_game_value': 'away_qb_game_value_538','qb2_adj':'away_538_qb_adj'}, inplace=True)

# Join on season and home team
merged_home = pd.merge(merged_away,
                       fivethirtyeight_df[['season','date', 'team1', 'qb1', 'qb1_value_pre', 'qb1_game_value','neutral','qb1_adj']],
                       left_on=['gameday', 'home_team'],
                       right_on=['date', 'team1'],
                       how='left')

# Renaming QB columns after the join for the home team
merged_home.rename(columns={'qb1': 'home_qb_538', 'qb1_value_pre': 'home_qb_value_pre_538', 'qb1_game_value': 'home_qb_game_value_538','qb1_adj':'home_538_qb_adj'}, inplace=True)


print(merged_home.head())
# Select the desired columns from the merged DataFrame
games_df_merged = merged_home[['game_id', 'season','game_type', 'away_team', 'away_qb_name', 'away_qb_538', 'away_qb_value_pre_538', 'away_qb_game_value_538',
                         'home_team', 'home_qb_name', 'home_qb_538', 'home_qb_value_pre_538', 'home_qb_game_value_538','home_score', 'away_score','week','neutral','surface','gametime',
                               'div_game','home_538_qb_adj','away_538_qb_adj','gameday']]


games_df_merged=pd.merge(left=games_df_merged,right=wepa_df[["game_id","wepa_net_home","wepa_net_away"]],on="game_id",how="left")
games_df_merged=pd.merge(left=games_df_merged,right=pff_df,on="game_id",how="left")
games_df_merged.rename(columns={"home_team_x":"home_team","away_team_x":"away_team","season_x":"season" },inplace=True)
games_df_merged["actual_margin"] = games_df_merged["home_score"]-games_df_merged["away_score"]

games_df_merged = games_df_merged[games_df_merged["season"] > 2019]

modelrun_df=games_df_merged[["game_id","week","home_team","away_team","actual_margin","OVER_home","OVER_away","wepa_net_home","wepa_net_away"]]


modelrun_df['starting_nflbyond_home'] = None
modelrun_df['starting_nflbyond_away'] = None
modelrun_df['ending_nflbyond_home'] = None
modelrun_df['ending_nflbyond_away'] = None

preseason2024ratings_df = pd.read_csv("D:/NFL/NFL Beyond/preseason2024ratings.csv").iloc[:,1:]

preseason2024ratings_df['Team'] = preseason2024ratings_df['Team'].replace({"JAC":"JAX","LV":"OAK"})




df_srs_ratings=pd.read_csv("D:/NFL/srs_ratings.csv")
df_srs_ratings['proj_rating'] = df_srs_ratings.groupby([
            'team', 'season'
        ])['srs_rating_normalized'].shift(1)

df_srs_ratings['proj_rating'] = df_srs_ratings['proj_rating'].combine_first(
            df_srs_ratings['pre_season_wt_rating']
        )

df_srs_ratings['proj_rating'] = (
            df_srs_ratings['proj_rating'] +
            df_srs_ratings['qb_adjustment']
        )

df_srs_ratings=df_srs_ratings[['season', 'week', 'team', 'proj_rating']]



games_df=pd.merge(left=games_df, right=df_srs_ratings, how='left', left_on=['season','week', 'home_team'], right_on=['season', 'week', 'team'])
games_df=games_df.rename(columns={'proj_rating':'home_rating'})
games_df=pd.merge(left=games_df, right=df_srs_ratings, how='left', left_on=['season','week', 'away_team'], right_on=['season', 'week', 'team'])
games_df=games_df.rename(columns={'proj_rating':'away_rating'})






hfa_df=calc_rolling_hfa(games_df, level_weeks, reg_weeks)

hfa_df=hfa_df.groupby(["season","week"]).agg("mean")

games_df_merged=pd.merge(left=games_df_merged, right=hfa_df, on=["season","week"], how='left')
# games_df_merged['rolling_hfa'] = games_df_merged['rolling_hfa'].fillna(2.75)

games_df_merged['hfa_mod'] = np.round(games_df_merged['rolling_hfa'],3)*25

flat_game_file = pd.concat([games_df_merged[['game_id','home_team','season','week']].reset_index(drop=True),
                            games_df_merged[['game_id','away_team','season','week']].rename(columns={'away_team' : 'home_team'}).reset_index(drop=True)])

list_removed_low_motivation_games = ["2023_18_DAL_WAS", "2023_18_KC_LAC", "2023_18_PIT_BAL", "2023_18_LAR_SF"]

flat_game_file=flat_game_file[~flat_game_file["game_id"].isin(list_removed_low_motivation_games)].reset_index(drop=True)

flat_game_file = flat_game_file.sort_values(by=['home_team','game_id'])
flat_game_file['game_number_home'] = flat_game_file.groupby(['home_team','season']).cumcount() + 1
flat_game_file['all_time_game_number_home'] = flat_game_file.groupby(['home_team']).cumcount() + 1
flat_game_file = flat_game_file.sort_values(by=['home_team','game_id'])
flat_game_file['prev_game_id_home'] = flat_game_file.groupby(['home_team'])['game_id'].shift(1)
flat_game_file['prev_week_home'] = flat_game_file.groupby(['home_team'])['week'].shift(1)
flat_game_file = flat_game_file.drop(columns=['all_time_game_number_home'])
flat_game_file_away = flat_game_file.rename(columns={
    'home_team' : 'away_team',
    'game_number_home' : 'game_number_away',
    'prev_game_id_home' : 'prev_game_id_away',
    'prev_week_home' : 'prev_week_away',
})

games_df_merged = pd.merge(games_df_merged,flat_game_file,on=['game_id','home_team','season', 'week'],how='left')
games_df_merged = pd.merge(games_df_merged,flat_game_file_away,on=['game_id','away_team','season', 'week'],how='left')
games_df_merged = games_df_merged.drop_duplicates()

new_df=games_df_merged.copy()

missing_surface_dict=games_df[games_df["season"]==2024].groupby("home_team").agg('last')[["surface"]].to_dict()["surface"]

new_df["surface"]=new_df["surface"].fillna(new_df["home_team"].map(missing_surface_dict))

fields_df = define_field_surfaces(new_df, pbp_surface_repl)

new_df = pd.merge(
    new_df,
    fields_df.rename(columns={
        'team': 'home_team',
        'surface': 'home_surface'
    }),
    on=['home_team', 'season'],
    how='left'
)

new_df = pd.merge(
        new_df,
        fields_df.rename(columns={
            'team' : 'away_team',
            'surface' : 'away_surface'
        }),
        on=['away_team', 'season'],
        how='left'
    )

# games_df_merged['away_bye_mod'] = np.where(
#     games_df_merged['week'] == 1,
#     0,
#     np.where(
#         games_df_merged['week'] > games_df_merged['prev_week_away'] + 1,
#         games_df_merged['hfa_mod'] * -0.778,
#         0
#     )
# )
## surface ##

games_df_merged['home_surface_advantage'] = np.where(
    (new_df['home_surface'] != new_df['away_surface']) &
    (new_df['surface'].replace(pbp_surface_repl) == new_df['home_surface']),
    1,
    0
)

games_df_merged['surface_mod'] = np.where(
    games_df_merged['home_surface_advantage'] == 1,
    0.35 * games_df_merged['hfa_mod'],
    0
)

## calculates the amount to shift each team in the elo model ##


shift_calc_helper_config = {}
shift_calc_helper_config["market_resist_factor"]=1.5039
shift_calc_helper_config["k"] = 9.114
shift_calc_helper_config["b"] = 10
shift_calc_helper_config["reversion"] = 0.001

games_df_merged['home_time_advantage'] = define_time_advantages(
        games_df_merged, pbp_timezones, pbp_timezone_overrides)

time_advantage=0.179
games_df_merged['time_mod'] = (
        games_df_merged['hfa_mod'] *
        games_df_merged['home_time_advantage'] *
        time_advantage
    )

hfa_div=-0.451

playoff_boost = 0.1

games_df_merged.rename(columns={'div_game' : 'divisional_game'},inplace=True)

games_df_merged['div_mod'] = np.where(
        games_df_merged['divisional_game'] == 1,
        hfa_div * games_df_merged['hfa_mod'],
        0
    )

games_df_merged['is_playoffs'] = np.where(games_df_merged['game_type'] == 'post',1,0)

dist_df = pd.read_csv(
    'D:/NFL/NFL Beyond/data/margin_distributions.csv'
)

home_bye_week = 0.287,
away_bye_week = -0.778

## home ##
games_df_merged['home_bye_mod'] = np.where(
    games_df_merged['week'] == 1,
    0,
    np.where(
        games_df_merged['week'] > games_df_merged['prev_week_home'] + 1,
        games_df_merged['hfa_mod'] * home_bye_week,
        0
    )
)
## away ##
games_df_merged['away_bye_mod'] = np.where(
    games_df_merged['week'] == 1,
    0,
    np.where(
        games_df_merged['week'] > games_df_merged['prev_week_away'] + 1,
        games_df_merged['hfa_mod'] * away_bye_week,
        0
    ))

games_df_merged['type'] = np.where(
        games_df_merged['season'] > 2020,
        np.where(
            games_df_merged['week'] >= 19,
            'post',
            'reg'
        ),
        np.where(
            games_df_merged['week'] >= 18,
            'post',
            'reg'
        )
    )





#### MARKET STUFF #####

market_df = pd.read_excel("D:/NFL/NFL Beyond/data/Market/2019-2024-pff-with-sbr-spreads.xlsx").iloc[:,1:][["GAME-ID","away_open_line","home_open_line","away_close_line","home_close_line"]]
market_df.tail(10)

market_df.rename(columns={"GAME-ID":"game_id"},inplace=True)
market_df.rename(columns={"away_open_line":"away_line_open", "home_open_line":"home_line_open",
                         "home_close_line":"home_line_close", "away_close_line":"away_line_close"},inplace=True)

# List of renamed columns
columns_to_round = ["away_line_open", "home_line_open", "home_line_close", "away_line_close"]

# Convert and round each column in the list
for column in columns_to_round:
    market_df[column] = market_df[column].astype(float).round(1)


market_df['game_id'] = market_df['game_id'].str.replace('-', '_')
market_df['game_id'] = market_df['game_id'].str.replace('WC', '19')
market_df['game_id'] = market_df['game_id'].str.replace('DP', '20')
market_df['game_id'] = market_df['game_id'].str.replace('CC', '21')
market_df['game_id'] = market_df['game_id'].str.replace('SB', '22')

# Split the game_id column into parts
market_df[['year', 'week', 'team1', 'team2']] = market_df['game_id'].str.split('_', expand=True)

# Format the week number to always have two digits
market_df['week'] = market_df['week'].apply(lambda x: f"{int(x):02d}")



market_df['team1'] = market_df['team1'].replace({"JAC":"JAX","LV":"OAK"})
market_df['team2'] = market_df['team2'].replace({"JAC":"JAX","LV":"OAK"})

# Rejoin the game_id column
market_df['game_id'] = market_df[['year', 'week', 'team1', 'team2']].apply('_'.join, axis=1)
market_df = market_df.drop(['year', 'week', 'team1', 'team2'], axis=1)

games_df_merged=pd.merge(left=games_df_merged,right=market_df, on='game_id',how='left')



market_current_df = pd.read_csv("D:/NFL/NFL Beyond/market_prices_current.csv")[["map", "home spread", "total line"]]

market_current_df.columns=["game_id", "home_spread_last", "total_line_last"]


games_df_merged=pd.merge(left=games_df_merged,right=market_current_df, on='game_id',how='left')



games_df_merged["home_margin"] = games_df_merged["home_score"]-games_df_merged["away_score"]


games_df_merged_2024 = games_df_merged[games_df_merged["season"]==2024]

pre_dict = pd.read_csv(
    'D:/NFL/NFL Beyond/data_sources/nfelo/probability_spread_multiples.csv'.format(

    )
)
pre_dict['win_prob'] = pre_dict['win_prob'].round(3)
pre_dict['implied_spread'] = pre_dict['implied_spread'].round(3)

spread_mult_dict = dict(zip(pre_dict['win_prob'], pre_dict['implied_spread']))

## pull out prob dict ##
# spread_dict['spread'] = spread_dict['spread'].round(3)
# spread_dict['implied_win_prob'] = spread_dict['implied_win_prob'].round(3)
# spread_translation_dict = dict(zip(spread_dict['spread'],spread_dict['implied_win_prob']))

elo_dict = create_data_struc(games_df_merged_2024)

spread2prob_df=pd.read_csv("D:/NFL/NFL Beyond/NFeloMargins.csv")
spread_dict = (
    spread2prob_df[spread2prob_df['result'] > 0]
    .groupby('spread_line')['normalized_modeled_prob']
    .apply(lambda x: sum(float(prob[:-1]) for p in x for prob in p.split('%') if prob))
    .reset_index(name='sum_prob')
)

spread_dict.columns=["spread","implied_win_prob"]

spread_translation_dict = dict(zip(spread_dict['spread'],spread_dict['implied_win_prob']))

home_keys = [-55, -48, -45, -41, -38, -35, -31, -28, -24, -21, -17, -14, -10, -7, -3, 3, 7, 10, 14, 17, 21, 24, 28, 31,
             35, 38, 41, 45, 48, 55]


spread_values = np.round(np.arange(-30, 37.1, 0.1), 1)  # Include 37.1 to make 37.0 in the range
# win_prob_dict = {spread: calculate_win_prob(spread) for spread in spread_values}
# win_prob_dict = {key * -1: value for key, value in win_prob_dict.items()}

with open('D:/NFL/NFL Beyond/data/spread_to_prob.pkl', 'rb') as f:
    win_prob_dict = pickle.load(f)

yearly_elos = {}

starting_nflbyond_home_list = []
starting_nflbyond_away_list = []
elo_diff_pre_market_list = []
nfelo_home_probability_pre_market_list = []
nfelo_home_line_close_pre_market_list = []
starting_nfelo_adj_home_list = []
starting_nfelo_adj_away_list = []

market_home_probability_list = []
market_home_probability_open_list = []
nfelo_home_probability_pre_regression_list = []
nfelo_home_line_close_pre_regression_list = []

starting_market_se_home_list = []
starting_market_se_away_list = []
starting_model_se_home_list = []
starting_model_se_away_list = []

nfelo_home_probability_list = []
nfelo_home_probability_open_list = []
nfelo_home_line_close_list = []
nfelo_home_line_close_rounded_list = []
nfelo_home_line_open_list = []
nfelo_home_line_open_rounded_list = []

nfelo_regressed_line_open_list=[]


away_loss_prob_list = []
away_push_prob_list = []
away_cover_prob_list = []
away_ev_list = []
home_loss_prob_list = []
home_push_prob_list = []
home_cover_prob_list = []
home_ev_list = []
away_loss_prob_open_list = []
away_push_prob_open_list = []
away_cover_prob_open_list = []
away_ev_open_list = []
home_loss_prob_open_list = []
home_push_prob_open_list = []
home_cover_prob_open_list = []
home_ev_open_list = []
away_loss_prob_unregressed_list = []
away_push_prob_unregressed_list = []
away_cover_prob_unregressed_list = []
away_ev_unregressed_list = []
home_loss_prob_unregressed_list = []
home_push_prob_unregressed_list = []
home_cover_prob_unregressed_list = []
home_ev_unregressed_list = []
ending_nfelo_home_list = []
ending_nfelo_away_list = []
home_net_wepa_point_margin_list = []
away_net_wepa_point_margin_list = []
home_net_pff_point_margin_list = []
away_net_pff_point_margin_list = []

se_market_list = []
se_model_list = []

## add accuracy ##
ending_market_se_home_list = []
ending_market_se_away_list = []
ending_model_se_home_list = []
ending_model_se_away_list = []

is_hook_list = []
is_long_list = []
spread_delta_open_list = []
all_in_mr_factor_list = []
avg_market_se_list = []
avg_rolling_nfelo_adj_list = []
avg_qb_adj_list = []
net_qb_adj_list = []

weighted_shift_home_list = []
weighted_shift_away_list = []




### Empirical mod lists
ini_elo_diff_list = []
hfa_mod_list = []
home_bye_mod_list=[]
away_bye_mod_list=[]
surface_mod_list=[]
time_mod_list=[]
div_mod_list=[]
home_538_qb_adj_list=[]
away_538_qb_adj_list=[]





# ending_nflbyond_away_list=[]
qb_weight = 1
elo_z = 401.62
spread_delta_base = 1.1
long_line_inflator = 0.5355
playoff_boost = 0.1
hook_certainty = 0
rmse_base = 2.993
min_mr = 0.1298
market_regression = 0.90
nfelo_span = 4
se_span = 8.346


# weight_vector = [0.17518729, 0.46879127, 0.35602144]
weight_vector = [0.74, 0.15,0.11]

margin_weight = weight_vector[0]
wepa_weight = weight_vector[1]
pff_weight = weight_vector[2]

regression_factor_used_list = []


pff_coefs={
        "overall_grade": 0.7034888234075101,
        "opponent_overall_grade": -0.7623521437565135,
        "intercept": 4.201930775595308}

games_df_merged.to_csv("games_df_merged.csv")

market_home_line_most_recent_list=[]

games_df_current_year = games_df_merged[(games_df_merged["week"] <=current_week)&(games_df_merged["season"] == current_season)]



# list_removed_low_motivation_games = ["2023_18_DAL_WAS", "2023_18_KC_LAC", "2023_18_PIT_BAL", "2023_18_LAR_SF"]

# games_df_current_year=games_df_current_year[~games_df_current_year["game_id"].isin(list_removed_low_motivation_games)].reset_index(drop=True)
previous_week_num = current_week-1
games_sameweek_notplayed = games_df_current_year[(games_df_current_year["week"]==previous_week_num)&(games_df_current_year["home_margin"].isnull())]["game_id"]


games_df_current_year=games_df_current_year[~games_df_current_year["game_id"].isin(games_sameweek_notplayed)].reset_index(drop=True)

games_df_current_year['hfa_mod'] = np.where(
            games_df_current_year['neutral'] == 1,
            0,
            games_df_current_year['hfa_mod']
        )


for index, row in games_df_current_year.iterrows():

    home_spread_last = row['home_spread_last'] if not pd.isnull(row['home_spread_last']) else -3






    if row["week"] == 1:

        starting_nflbyond_home = \
        preseason2024ratings_df[preseason2024ratings_df["Team"] == row["home_team"]]["Elo"].iloc[0]
        starting_nflbyond_away = \
        preseason2024ratings_df[preseason2024ratings_df["Team"] == row["away_team"]]["Elo"].iloc[0]

        starting_nflbyond_home_list.append(starting_nflbyond_home)
        starting_nflbyond_away_list.append(starting_nflbyond_away)

        elo_dict[row['away_team']][row['game_id']]['starting'] = starting_nflbyond_away
        elo_dict[row['home_team']][row['game_id']]['starting'] = starting_nflbyond_home


    else:

        starting_nflbyond_home_list.append(elo_dict[row['home_team']][row['prev_game_id_home']]['ending'])
        starting_nflbyond_away_list.append(elo_dict[row['away_team']][row['prev_game_id_away']]['ending'])



    elo_diff_pre_market = (
        ## base elo difference ##
            starting_nflbyond_home_list[-1] - starting_nflbyond_away_list[-1] +
            ## empirical mods ##
            row['hfa_mod'] + row['home_bye_mod'] + row['away_bye_mod'] +
            row['surface_mod'] + row['time_mod'] + row['div_mod'] +
            ## QB adjustment ##
            qb_weight * (row['home_538_qb_adj'] - row['away_538_qb_adj'])
    )

    ini_elo_diff_list.append(starting_nflbyond_home_list[-1] - starting_nflbyond_away_list[-1])
    hfa_mod_list.append(row['hfa_mod'])
    home_bye_mod_list.append(row['home_bye_mod'])
    away_bye_mod_list.append(row['away_bye_mod'])
    surface_mod_list.append(row['surface_mod'])
    time_mod_list.append(row['time_mod'])
    div_mod_list.append(row['div_mod'])
    home_538_qb_adj_list.append(row['home_538_qb_adj'])
    away_538_qb_adj_list.append(row['away_538_qb_adj'])


    if row['is_playoffs'] == 1:
        elo_diff_pre_market = elo_diff_pre_market * (1 + playoff_boost)
    else:
        pass

    home_net_wepa_point_margin_list.append(row['wepa_net_home'])
    away_net_wepa_point_margin_list.append(row['wepa_net_away'])

    home_net_pff_point_margin_list.append(pff_coefs["intercept"] + pff_coefs["overall_grade"]*row['OVER_home'] + pff_coefs["opponent_overall_grade"]*row['OVER_away'])

    away_net_pff_point_margin_list.append(pff_coefs["intercept"] + pff_coefs["opponent_overall_grade"]*row['OVER_home'] + pff_coefs["overall_grade"]*row['OVER_away'])

    elo_diff_pre_market_list.append(elo_diff_pre_market)
    nfelo_home_probability_pre_market = 1.0 / (math.pow(10.0, (-elo_diff_pre_market / elo_z)) + 1.0)
    nfelo_home_line_close_pre_market = spread_mult_dict[round(nfelo_home_probability_pre_market, 3)]
    nfelo_home_probability_pre_market_list.append(nfelo_home_probability_pre_market)
    nfelo_home_line_close_pre_market_list.append(nfelo_home_line_close_pre_market)

    ## Update market adjustments ##
    ## Pull starting rolling nfelo adjustments ##
    try:
        starting_nfelo_adj_home = elo_dict[row['home_team']][row['prev_game_id_home']].get('rolling_nfelo_adj_end', 0)
        starting_nfelo_adj_away = elo_dict[row['away_team']][row['prev_game_id_away']].get('rolling_nfelo_adj_end', 0)
        starting_nfelo_adj_home_list.append(starting_nfelo_adj_home)
        starting_nfelo_adj_away_list.append(starting_nfelo_adj_away)
    except:
        starting_nfelo_adj_home = 0
        starting_nfelo_adj_away = 0
        starting_nfelo_adj_home_list.append(starting_nfelo_adj_home)
        starting_nfelo_adj_away_list.append(starting_nfelo_adj_away)



    ## save a pre regression probability ##

    # If 'home_line_close' is nan, append win_prob_dict(1); otherwise, use the provided value.
    market_home_probability_list.append(
        win_prob_dict[1] if np.isnan(row['home_line_close']) else win_prob_dict[row['home_line_close']])

    market_home_line_most_recent_list.append(home_spread_last)

    # If 'home_line_open' is nan, append win_prob_dict(1); otherwise, use the provided value.
    market_home_probability_open_list.append(
        win_prob_dict[1] if np.isnan(home_spread_last) else win_prob_dict[home_spread_last])


    nfelo_home_probability_pre_regression_list.append(1.0 / (math.pow(10.0, (-elo_diff_pre_market / elo_z)) + 1.0))
    nfelo_home_line_close_pre_regression_list.append(nfelo_home_line_close_pre_market)
    #
    try:
        starting_market_se_home_list.append(
            elo_dict[row['home_team']][row['prev_game_id_home']].get('rolling_market_se_end', 0))
        starting_market_se_away_list.append(
            elo_dict[row['away_team']][row['prev_game_id_away']].get('rolling_market_se_end', 0))
        starting_model_se_home_list.append(
            elo_dict[row['home_team']][row['prev_game_id_home']].get('rolling_model_se_end', 0))
        starting_model_se_away_list.append(
            elo_dict[row['away_team']][row['prev_game_id_away']].get('rolling_model_se_end', 0))
    except:
        starting_market_se_home_list.append(0)
        starting_market_se_away_list.append(0)
        starting_model_se_home_list.append(0)
        starting_model_se_away_list.append(0)
    ## then calc the scaling factor ##
    rmse_dif = (
            ((starting_model_se_home_list[-1] ** (1 / 2) + starting_model_se_away_list[-1] ** (1 / 2)) / 2) -
            ((starting_market_se_home_list[-1] ** (1 / 2) + starting_market_se_away_list[-1] ** (1 / 2)) / 2)
    )

    spread_delta_open = abs(nfelo_home_line_close_pre_regression_list[-1] - home_spread_last)

    mr_deflator_factor = (
            4 / (
            1 +
            (spread_delta_base * spread_delta_open ** 2)
    ) +
            spread_delta_open / 14
    )
    mr_factor = mr_deflator_factor

    is_long = 0
    if home_spread_last < -7.5 and nfelo_home_line_close_pre_regression_list[-1] > home_spread_last:
        is_long = 1
    long_inflator = 1 + (is_long * long_line_inflator)
    mr_factor = mr_factor * long_inflator

    is_hook = 1
    if home_spread_last == round(home_spread_last):
        is_hook = 0
    hook_inflator = 1 + (is_hook * hook_certainty)
    mr_factor = mr_factor * hook_inflator

    if spread_delta_open > 1:
        mr_factor = mr_factor * (1 + rmse_dif / rmse_base)
    else:
        pass

    mr_mult = max(min_mr, min(1, market_regression * mr_factor))
    #
    regression_factor_used_list.append(mr_mult)




    market_elo_dif_close = (
            (-1 * elo_z) *
            math.log10(
                (1 / market_home_probability_list[-1]) -
                1
            )
    )
    market_elo_dif_open = (
            (-1 * elo_z) *
            math.log10(
                (1 / market_home_probability_open_list[-1]) -
                1
            )
    )
    # elo_diff_open = elo_dif + mr_mult * (market_elo_dif_open - elo_diff_pre_market)

    regressed_elo_diff_open = elo_diff_pre_market + mr_mult * (market_elo_dif_open - elo_diff_pre_market)


    # elo_dif = elo_dif + mr_mult * (market_elo_dif_close - elo_dif)

    nfelo_home_probability_list.append(1.0 / (math.pow(10.0, (-market_elo_dif_open / elo_z)) + 1.0))
    nfelo_home_probability_open_list.append(1.0 / (math.pow(10.0, (-elo_diff_pre_market / elo_z)) + 1.0))

    regressed_prob=elo_to_prob(regressed_elo_diff_open)



    nfelo_regressed_line_open_list.append(spread_mult_dict[round(regressed_prob, 3)])

    nfelo_home_line_close_list.append((
        ## the unrounded line is a simple calc with a fixed multiplier of -16 ##
            -16 *
            ## multiply by a s
            math.log10(nfelo_home_probability_list[-1] / max(1 - nfelo_home_probability_list[-1], .001)))
    )
    # ## the rounded line uses a win prob to spread translation derived from actual moneyline probs and spreads ##
    nfelo_home_line_close_rounded_list.append(spread_mult_dict[round(nfelo_home_probability_list[-1], 3)])
    nfelo_home_line_open_list.append(
        -16 *
        ## multiply by a s
        math.log10(nfelo_home_probability_open_list[-1] / max(1 - nfelo_home_probability_open_list[-1], .001))
    )
    nfelo_home_line_open_rounded_list.append(spread_mult_dict[round(nfelo_home_probability_open_list[-1], 3)])

    # ## calc cover probs ##
    if nfelo_home_line_close_rounded_list[-1] <= 0:
        home_probs = calc_probs_favorite(nfelo_home_line_close_rounded_list[-1], row['home_line_close'], dist_df)
    else:
        home_probs = calc_probs_dog(nfelo_home_line_close_rounded_list[-1], row['home_line_close'], dist_df)
    if nfelo_home_line_open_rounded_list[-1] <= 0:
        home_probs_open = calc_probs_favorite(nfelo_home_line_open_rounded_list[-1], home_spread_last, dist_df)
    else:
        home_probs_open = calc_probs_dog(nfelo_home_line_open_rounded_list[-1], home_spread_last, dist_df)
    if nfelo_home_line_close_pre_market_list[-1] <= 0:
        home_probs_unregressed = calc_probs_favorite(nfelo_home_line_close_pre_market_list[-1], row['home_line_close'],
                                                     dist_df)
    else:
        home_probs_unregressed = calc_probs_dog(nfelo_home_line_close_pre_market_list[-1], row['home_line_close'],
                                                dist_df)

    away_loss_prob_list.append(home_probs[2])
    away_push_prob_list.append(home_probs[1])
    away_cover_prob_list.append(home_probs[0])
    away_ev_list.append((away_cover_prob_list[-1] - 1.1 * away_push_prob_list[-1]) / 1.1)
    home_loss_prob_list.append(home_probs[0])
    home_push_prob_list.append(home_probs[1])
    home_cover_prob_list.append(home_probs[2])
    home_ev_list.append((home_probs[2] - 1.1 * home_probs[0]) / 1.1)
    away_loss_prob_open_list.append(home_probs[2])
    away_push_prob_open_list.append(home_probs_open[1])
    away_cover_prob_open_list.append(home_probs_open[0])
    away_ev_open_list.append((away_cover_prob_open_list[-1] - 1.1 * away_loss_prob_open_list[-1]) / 1.1)

    home_loss_prob_open_list.append(home_probs_open[0])
    home_push_prob_open_list.append(home_probs_open[1])
    home_cover_prob_open_list.append(home_probs_open[2])
    home_ev_open_list.append((home_probs_open[2] - 1.1 * home_probs_open[0]) / 1.1)
    away_loss_prob_unregressed_list.append(home_probs_unregressed[2])
    away_push_prob_unregressed_list.append(home_probs_unregressed[1])
    away_cover_prob_unregressed_list.append(home_probs_unregressed[0])

    away_ev_unregressed_list.append(
        (away_cover_prob_unregressed_list[-1] - 1.1 * away_loss_prob_unregressed_list[-1]) / 1.1)
    home_loss_prob_unregressed_list.append(home_probs_unregressed[0])
    home_push_prob_unregressed_list.append(home_probs_unregressed[1])
    home_cover_prob_unregressed_list.append(home_probs_unregressed[2])
    home_ev_unregressed_list.append((home_probs_unregressed[2] - 1.1 * home_probs_unregressed[0]) / 1.1)

    ## calculate shifts ##
    ## margin ##
    margin_shift_home = shift_calc_helper(
        row['home_margin'], nfelo_home_line_close_pre_regression_list[-1],
        row['home_line_close'], shift_calc_helper_config, True
    )
    margin_shift_away = shift_calc_helper(
        -1 * row['home_margin'], nfelo_home_line_close_pre_regression_list[-1],
        row['home_line_close'], shift_calc_helper_config, False
    )
    ## wepa ##
    wepa_shift_home = shift_calc_helper(
        home_net_wepa_point_margin_list[-1], nfelo_home_line_close_pre_regression_list[-1],
        row['home_line_close'], shift_calc_helper_config, True
    )
    wepa_shift_away = shift_calc_helper(
        away_net_wepa_point_margin_list[-1], nfelo_home_line_close_pre_regression_list[-1],
        row['home_line_close'], shift_calc_helper_config, False
    )
    ## pff ##
    pff_shift_home = shift_calc_helper(
        home_net_pff_point_margin_list[-1], nfelo_home_line_close_pre_regression_list[-1],
        row['home_line_close'], shift_calc_helper_config, True
    )
    pff_shift_away = shift_calc_helper(
        away_net_pff_point_margin_list[-1], nfelo_home_line_close_pre_regression_list[-1],
        row['home_line_close'], shift_calc_helper_config, False
    )
    ## apply weighted average shift ##
    weighted_shift_home = (
        ## straight margin shift ##
            (margin_shift_home * margin_weight) +
            ## wepa ##
            (wepa_shift_home * wepa_weight) +
            ## pff ##
            (pff_shift_home * pff_weight)
    )
    weighted_shift_away = (
        ## straight margin shift ##
            (margin_shift_away * margin_weight) +
            ## wepa ##
            (wepa_shift_away * wepa_weight) +
            ## pff ##
            (pff_shift_away * pff_weight)
    )

    weighted_shift_home_list.append(weighted_shift_home)
    weighted_shift_away_list.append(weighted_shift_away)

    ## apply weighted shift ##
    ending_nfelo_home_list.append(starting_nflbyond_home_list[-1] + (weighted_shift_home))
    ending_nfelo_away_list.append(starting_nflbyond_away_list[-1] + (weighted_shift_away))
    ## add new elo back to dictionary ##
    nfelo_alpha = 2 / (1 + nfelo_span)
    ## away ##
    elo_dict[row['away_team']][row['game_id']]['starting'] = starting_nflbyond_away
    elo_dict[row['away_team']][row['game_id']]['ending'] = ending_nfelo_away_list[-1]
    elo_dict[row['away_team']][row['game_id']]['week'] = row['week']
    elo_dict[row['away_team']][row['game_id']]['rolling_nfelo_adj_start'] = starting_nfelo_adj_away
    elo_dict[row['away_team']][row['game_id']]['rolling_nfelo_adj_end'] = (
            starting_nfelo_adj_away * (1 - nfelo_alpha) +
            abs(weighted_shift_away) * (nfelo_alpha)
    )
    ## home ##
    elo_dict[row['home_team']][row['game_id']]['starting'] = starting_nflbyond_home
    elo_dict[row['home_team']][row['game_id']]['ending'] = ending_nfelo_home_list[-1]
    elo_dict[row['home_team']][row['game_id']]['week'] = row['week']
    elo_dict[row['home_team']][row['game_id']]['rolling_nfelo_adj_start'] = starting_nfelo_adj_home
    elo_dict[row['home_team']][row['game_id']]['rolling_nfelo_adj_end'] = (
            starting_nfelo_adj_home * (1 - nfelo_alpha) +
            abs(weighted_shift_home) * (nfelo_alpha))

    se_market_list.append((row['home_margin'] + row['home_line_close']) ** 2)
    se_model_list.append((row['home_margin'] + nfelo_home_line_close_pre_regression_list[-1]) ** 2)
    se_ema_alpha = 2 / (1 + se_span)
    #
    ## away ##
    elo_dict[row['away_team']][row['game_id']]['rolling_market_se_end'] = (
            starting_market_se_away_list[-1] * (1 - se_ema_alpha) +
            se_market_list[-1] * (se_ema_alpha)
    )
    elo_dict[row['away_team']][row['game_id']]['rolling_model_se_end'] = (
            starting_model_se_away_list[-1] * (1 - se_ema_alpha) +
            se_model_list[-1] * (se_ema_alpha)
    )
    elo_dict[row['home_team']][row['game_id']]['rolling_market_se_end'] = (
            starting_market_se_home_list[-1] * (1 - se_ema_alpha) +
            se_market_list[-1] * (se_ema_alpha)
    )
    elo_dict[row['home_team']][row['game_id']]['rolling_model_se_end'] = (
            starting_model_se_home_list[-1] * (1 - se_ema_alpha) +
            se_model_list[-1] * (se_ema_alpha)
    )
    #
    # ## add accuracy ##
    ending_market_se_home_list.append((
            starting_market_se_home_list[-1] * (1 - se_ema_alpha) +
            se_market_list[-1] * (se_ema_alpha)
    ))
    ending_market_se_away_list.append((
            starting_market_se_away_list[-1] * (1 - se_ema_alpha) +
            se_market_list[-1] * (se_ema_alpha)
    ))
    ending_model_se_home_list.append((
            starting_model_se_home_list[-1] * (1 - se_ema_alpha) +
            se_model_list[-1] * (se_ema_alpha)
    ))
    ending_model_se_away_list.append((
            starting_model_se_away_list[-1] * (1 - se_ema_alpha) +
            se_model_list[-1] * (se_ema_alpha)
    ))


df_elo_final = pd.DataFrame({
    'game_id': games_df_current_year["game_id"], 'home_team': games_df_current_year["home_team"],
    'date': games_df_current_year["gameday"],
    'away_team': games_df_current_year["away_team"]

    ,'home_margin': games_df_current_year["home_margin"]
    ,'home_elo_before': starting_nflbyond_home_list
    ,'home_elo_after': ending_nfelo_home_list,
    'away_elo_before': starting_nflbyond_away_list
    ,'away_elo_after': ending_nfelo_away_list
    # ,
    ,'home_wepa_margin': home_net_wepa_point_margin_list,
     'away_wepa_margin': away_net_wepa_point_margin_list,
    'home_pff': games_df_current_year["OVER_home"],
    'away_pff':games_df_current_year["OVER_away"],

    'home_model_win_prob_premarket': nfelo_home_probability_pre_market_list,
    'home_model_line_premarket': nfelo_home_line_close_pre_regression_list,
    'home_model_line_postmarket':nfelo_regressed_line_open_list,

    # ,
    'home_line_close': games_df_current_year['home_line_close'],
    'weighted_shift_home': weighted_shift_home_list, 'weighted_shift_away': weighted_shift_away_list
    # 'NFLByond_homewinprob': nfelo_home_probability_list,
    # 'NFLByond_homewinprob_open': nfelo_home_probability_open_list
    , 'hfa_mod': hfa_mod_list
    , 'home_bye_mod': home_bye_mod_list
    , 'away_bye_mod': away_bye_mod_list
    , 'surface_mod': surface_mod_list
    , 'time_mod': time_mod_list
    , 'div_mod': div_mod_list
    , 'home_538_qb_adj': home_538_qb_adj_list
    , 'away_538_qb_adj': away_538_qb_adj_list
    , 'initial_elo_diff': ini_elo_diff_list
    ,'nfelo_regressed_line_open':nfelo_regressed_line_open_list,
    'market_home_line_recent':market_home_line_most_recent_list,
    'home_pff_margin':home_net_pff_point_margin_list,
    'away_pff_margin':away_net_pff_point_margin_list

})




with open('D:/NFL/NFL Beyond/current_starters.pkl', 'rb') as f:
    dict_current_starters = pickle.load(f)

df_current_starters=pd.DataFrame.from_dict(dict_current_starters, orient='index')

df_current_starters = df_current_starters.reset_index()
df_current_starters.columns = ['team', 'weird_id', 'qb_id']

df_current_starters_home = df_current_starters.copy()
df_current_starters_away = df_current_starters.copy()

df_current_starters_home.columns = ['home_team','weird_id' ,'home_qb_id']
df_current_starters_away.columns = ['away_team','weird_id' ,'away_qb_id']

df_current_starters_home.drop(columns=['weird_id'], inplace=True)
df_current_starters_away.drop(columns=['weird_id'], inplace=True)


# df_elo_final["home_team_win"] = df_elo_final["home_margin"] > 0


df_elo_final = pd.merge(left=df_elo_final,right=df_current_starters_home, how='left', left_on='home_team', right_on='home_team')
df_elo_final = pd.merge(left=df_elo_final,right=df_current_starters_away, how='left', left_on='away_team', right_on='away_team')

df_airtable_qbs = pd.read_csv("D:/NFL/dashboard/airtable_qb_selections.csv")[["player_id", "player_name"]]
df_airtable_qbs_home = df_airtable_qbs.copy()
df_airtable_qbs_home.columns = ['home_qb_id', 'home_qb_name']
df_airtable_qbs_away = df_airtable_qbs.copy()
df_airtable_qbs_away.columns = ['away_qb_id', 'away_qb_name']

df_elo_final = pd.merge(left=df_elo_final,right=df_airtable_qbs_home, how='left', left_on='home_qb_id', right_on='home_qb_id')
df_elo_final = pd.merge(left=df_elo_final,right=df_airtable_qbs_away, how='left', left_on='away_qb_id', right_on='away_qb_id')

df_totals = pd.read_csv("D:/NFL/Totals/NFL_predicted_totals.csv")

df_elo_final = pd.merge(left=df_elo_final, right=df_totals, how='left', on='game_id')

df_elo_final.set_index('game_id', inplace=True)
df_elo_final.to_csv("df_elo_final.csv")


