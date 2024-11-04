import pandas as pd
import numpy
import statsmodels.api as sm
from scipy.optimize import minimize
import time
import nfl_data_py as nfl

best_guesses_all_time = [

    0.3593187177967391,    ## tos_three_quarters_int ##
    0.15254178441536528,    ## special_teams_all ##
    0.5550039450628008,    ## plays_all_boosted ##
]

## define weight names for reference ##
weight_names_list = [

    'tos_three_quarters_int',
    'special_teams_all',
    'd_plays_all_boosted'

]


## define function for calculating wepa given a dictionary of weights ##
def wepa_grade(weight_list, test_df):
    ## define weights ##
    ## use vectorized mapping to look up weights from a dictionary ##
    ## play style ##

    test_df['tos_three_quarters_int_weight'] =numpy.where(
        test_df['fumble_lost'] == 1,
        weight_list[0],
        numpy.where(
            test_df['interception'] == 1,
            .75 * weight_list[0],
            0
        )
    )

    mask = (numpy.isin(
        test_df['play_type'],
        ['kickoff','punt','field_goal','extra_point']
    ))
    
    test_df['special_teams_all_weight'] = numpy.where(mask, weight_list[1], 0)

    test_df['d_plays_all_boosted_weight'] = numpy.where(
        True,
        numpy.where(
            (test_df['interception'] == 1) |
            (test_df['fumble_lost'] == 1),
            weight_list[2],
            .5 * weight_list[2]
        ),
        0
    )
    

    ## add weights to list to build out headers and loops ##
    weight_names = [
        'tos_three_quarters_int_weight',
        'special_teams_all_weight']
    
    d_weight_names = [
        'd_plays_all_boosted_weight']
    
    ## create a second list for referencing the specifc weights ##
    weight_values = []
    for weight in weight_names:
        weight_values.append('{0}'.format(weight))
    ## defense ##
    d_weight_values = []
    for weight in d_weight_names:
        d_weight_values.append('{0}'.format(weight))
    ## create structures for aggregation ##
    aggregation_dict = {
        'margin' : 'max', ## game level margin added to each play, so take max to get 1 ##
        'wepa' : 'sum',
        'd_wepa' : 'sum',
        'epa' : 'sum',
    }
    headers = [
        'game_id',
        'posteam',
        'defteam',
        'season',
        'game_number',
        'margin',
        'wepa',
        'd_wepa',
        'epa'
    ]
  
    ## disctionary to join oppoenets epa to net out ##
    rename_opponent_dict = {
        'margin' : 'margin_against',
        'wepa' : 'wepa_against',
        'd_wepa' : 'd_wepa_against',
        'epa' : 'epa_against',
    }
    ## create wepa ##
    test_df['wepa'] = test_df['epa']
    
    for weight in weight_values:
        test_df['wepa'] = 0* test_df[weight] + test_df['wepa']*(1- test_df[weight])

    test_df['d_wepa'] = test_df['epa']

    for weight in d_weight_values:
        test_df['d_wepa'] = 0* test_df[weight] + test_df['d_wepa']*(1- test_df[weight])






        
    ## bound wepa to prevent extreme values from introducing volatility ##
    test_df['wepa'] = numpy.where(test_df['wepa'] > 10, 10, test_df['wepa'])
    test_df['wepa'] = numpy.where(test_df['wepa'] < -10, -10, test_df['wepa'])
    ## defense ##
    test_df['d_wepa'] = numpy.where(test_df['d_wepa'] > 10, 10, test_df['d_wepa'])
    test_df['d_wepa'] = numpy.where(test_df['d_wepa'] < -10, -10, test_df['d_wepa'])
    ## aggregate from pbp to game level ##
    game_level_df = test_df.groupby(['posteam','defteam','season','game_id','game_number']).agg(aggregation_dict).reset_index()
    game_level_df = game_level_df.sort_values(by=['posteam','game_id'])
    game_level_df = game_level_df[headers]
    ## add net epa ##
    ## create an opponent data frame ##
    game_level_opponent_df = game_level_df.copy()
    game_level_opponent_df['posteam'] = game_level_opponent_df['defteam']
    game_level_opponent_df = game_level_opponent_df.drop(columns=['defteam','season','game_number'])
    game_level_opponent_df = game_level_opponent_df.rename(columns=rename_opponent_dict)
    ## merge to main game file ##
    game_level_df = pd.merge(
        game_level_df,
        game_level_opponent_df,
        on=['posteam', 'game_id'],
        how='left'
    )
    ## calculate net wepa and apply defensive adjustment ##
    game_level_df['wepa_net'] = game_level_df['wepa'] - game_level_df['d_wepa_against']
    ## rename ##
    game_level_df = game_level_df.rename(columns={'posteam' : 'team', 'defteam' : 'opponent'})
    ## rejoin oppoenent net wepa ##
    game_level_df_opponent = game_level_df.copy()
    game_level_df_opponent = game_level_df_opponent[['opponent', 'game_id', 'wepa_net']].rename(columns={
        'opponent' : 'team',
        'wepa_net' : 'wepa_net_opponent',
    })
    game_level_df = pd.merge(
        game_level_df,
        game_level_df_opponent,
        on=['team', 'game_id'],
        how='left'
    )
    return game_level_df

aggregation_dict = {
        'margin' : 'max', ## game level margin added to each play, so take max to get 1 ##
        'wepa' : 'sum',
        'd_wepa' : 'sum',
        'epa' : 'sum',
    }

headers = [
        "game_id",
        "team",
        "opponent",
        "season",
        "game_number",
        "margin",
        "margin_against",
        "epa",
        "epa_against",
        "epa_net",
        "epa_net_opponent",
        "wepa",
        "d_wepa_against",
        "wepa_net",
        "wepa_against",
        "d_wepa",
        "wepa_net_opponent"
      ]

pbp_team_standard_dict = {

    'ARI' : 'ARI',
    'ATL' : 'ATL',
    'BAL' : 'BAL',
    'BUF' : 'BUF',
    'CAR' : 'CAR',
    'CHI' : 'CHI',
    'CIN' : 'CIN',
    'CLE' : 'CLE',
    'DAL' : 'DAL',
    'DEN' : 'DEN',
    'DET' : 'DET',
    'GB'  : 'GB',
    'HOU' : 'HOU',
    'IND' : 'IND',
    'JAC' : 'JAX',
    'JAX' : 'JAX',
    'KC'  : 'KC',
    'LA'  : 'LAR',
    'LAC' : 'LAC',
    'LV'  : 'OAK',
    'MIA' : 'MIA',
    'MIN' : 'MIN',
    'NE'  : 'NE',
    'NO'  : 'NO',
    'NYG' : 'NYG',
    'NYJ' : 'NYJ',
    'OAK' : 'OAK',
    'PHI' : 'PHI',
    'PIT' : 'PIT',
    'SD'  : 'LAC',
    'SEA' : 'SEA',
    'SF'  : 'SF',
    'STL' : 'LAR',
    'TB'  : 'TB',
    'TEN' : 'TEN',
    'WAS' : 'WAS',

}

pbp_df=nfl.import_pbp_data([2024],downcast=True, cache=False, alt_path=None,include_participation=False)

#pbp_df = pd.read_csv("C:/Users/NolanNicholls/Documents/NFL/Data/pbp_import/pbp-data-2024.csv")

pbp_df['posteam'] = pbp_df['posteam'].replace(pbp_team_standard_dict)
pbp_df['defteam'] = pbp_df['defteam'].replace(pbp_team_standard_dict)
pbp_df['penalty_team'] = pbp_df['penalty_team'].replace(pbp_team_standard_dict)
pbp_df['home_team'] = pbp_df['home_team'].replace(pbp_team_standard_dict)
pbp_df['away_team'] = pbp_df['away_team'].replace(pbp_team_standard_dict)

## replace game_id using standardized franchise names ##
pbp_df['game_id'] = (
    pbp_df['season'].astype('str') +
    '_' +
    pbp_df['week'].astype('str').str.zfill(2) +
    '_' +
    pbp_df['away_team'] +
    '_' +
    pbp_df['home_team']
)

## fix some data formatting issues ##
pbp_df['yards_after_catch'] = pd.to_numeric(pbp_df['yards_after_catch'], errors='coerce')

## denote pass or run ##
## seperate offensive and defensive penalties ##
pbp_df['off_penalty'] = numpy.where(pbp_df['penalty_team'] == pbp_df['posteam'], 1, 0)
pbp_df['def_penalty'] = numpy.where(pbp_df['penalty_team'] == pbp_df['defteam'], 1, 0)

## pandas wont group nans so must fill with a value ##
pbp_df['penalty_type'] = pbp_df['penalty_type'].fillna('No Penalty')

## accepted pentalites on no plays need additional detail to determine if they were a pass or run ##
## infer pass plays from the play description ##
pbp_df['desc_based_dropback'] = numpy.where(
    (
        (pbp_df['desc'].str.contains(' pass ', regex=False)) |
        (pbp_df['desc'].str.contains(' sacked', regex=False)) |
        (pbp_df['desc'].str.contains(' scramble', regex=False))
    ),
    1,
    0
)

## infer run plays from the play description ##
pbp_df['desc_based_run'] = numpy.where(
    (
        (~pbp_df['desc'].str.contains(' pass ', regex=False, na=False)) &
        (~pbp_df['desc'].str.contains(' sacked', regex=False, na=False)) &
        (~pbp_df['desc'].str.contains(' scramble', regex=False, na=False)) &
        (~pbp_df['desc'].str.contains(' kicks ', regex=False, na=False)) &
        (~pbp_df['desc'].str.contains(' punts ', regex=False, na=False)) &
        (~pbp_df['desc'].str.contains(' field goal ', regex=False, na=False)) &
        (pbp_df['desc'].str.contains(' to ', regex=False)) &
        (pbp_df['desc'].str.contains(' for ', regex=False))
    ),
    1,
    0
)








## coalesce coded and infered drop backs ##
pbp_df['qb_dropback'] = pbp_df[['qb_dropback', 'desc_based_dropback']].max(axis=1)

## coalesce coaded and infered rush attemps ##
pbp_df['rush_attempt'] = pbp_df[['rush_attempt', 'desc_based_run']].max(axis=1)


## create a specific field for play call ##
pbp_df['play_call'] = numpy.where(
                            pbp_df['qb_dropback'] == 1,
                            'Pass',
                            numpy.where(
                                pbp_df['rush_attempt'] == 1,
                                'Run',
                                numpy.nan
                            )
)

## Structure game file to attach to PBP data ##
## calc margin ##

game_file_df = nfl.import_schedules([2024,2025])

game_file_df['home_team'] = game_file_df['home_team'].replace(pbp_team_standard_dict)
game_file_df['away_team'] = game_file_df['away_team'].replace(pbp_team_standard_dict)


game_file_df['game_id'] = (
    game_file_df['season'].astype('str') +
    '_' +
    game_file_df['week'].astype('str').str.zfill(2) +
    '_' +
    game_file_df['away_team'] +
    '_' +
    game_file_df['home_team']
)

game_file_df['home_margin'] = game_file_df['home_score'] - game_file_df['away_score']
game_file_df['away_margin'] = game_file_df['away_score'] - game_file_df['home_score']

## flatten file to attach to single team
game_home_df = game_file_df.copy()[['game_id', 'week', 'season', 'home_team', 'home_margin']].rename(columns={
    'home_team' : 'posteam',
    'home_margin' : 'margin',
})
game_away_df = game_file_df.copy()[['game_id', 'week', 'season', 'away_team', 'away_margin']].rename(columns={
    'away_team' : 'posteam',
    'away_margin' : 'margin',
})

flat_game_df = pd.concat([game_home_df,game_away_df], ignore_index=True).sort_values(by=['game_id'])

## calculate game number to split in regressions ##
flat_game_df['game_number'] = flat_game_df.groupby(['posteam', 'season']).cumcount() + 1

## merge to pbp now, so you don't have to merge on every loop ##



pbp_df = pd.merge(
    pbp_df,
    flat_game_df[['posteam','game_id','margin', 'game_number']],
    on=['posteam','game_id'],
    how='left'
)

pbp_df = pbp_df[pbp_df['game_number'] < 23]


wepa_df = wepa_grade(best_guesses_all_time, pbp_df.copy())

## export ##
wepa_df['epa_net'] = wepa_df['epa'] - wepa_df['epa_against']
wepa_df['epa_net_opponent'] = wepa_df['epa_against'] - wepa_df['epa']
wepa_df = wepa_df[[
    'game_id', 'team', 'opponent', 'season', 'game_number', 'margin', 'margin_against',
    'epa', 'epa_against', 'epa_net', 'epa_net_opponent',
    'wepa', 'd_wepa_against', 'wepa_net',
    'wepa_against', 'd_wepa', 'wepa_net_opponent'
]]


wepa_df.to_csv('C:/Users/NolanNicholls/Documents/NFL/Data/wepa_game_flat_file.csv')
