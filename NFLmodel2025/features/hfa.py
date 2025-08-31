# features/hfa.py

import pandas as pd
import statsmodels.api as sm

def calc_rolling_hfa(current_df, level_weeks, reg_weeks):

    hfa_df = current_df.copy()
    hfa_df['expected_result'] = hfa_df['home_rating'] - hfa_df['away_rating']
    hfa_df['home_margin_error'] = hfa_df['result'] - hfa_df['expected_result']

    hfa_df_temp = hfa_df[(hfa_df['season'] != 2020) & (hfa_df['game_type'] == 'REG') & (hfa_df['location'] != 'Neutral')].copy()
    hfa_df_temp = hfa_df_temp.groupby(['season', 'week']).agg(avg_error=('home_margin_error', 'mean')).reset_index()

    a = 2 / (level_weeks + 1)
    hfa_df_temp['level'] = 2.50
    hfa_df_temp['intercept_constant'] = 1

    for index, row in hfa_df_temp.iterrows():
        if index < level_weeks:
            continue
        window_start = max(index - reg_weeks, 0)
        trailing_window = hfa_df_temp.iloc[window_start:index].copy()
        trailing_window['week_num'] = range(1, len(trailing_window)+1)

        reg = sm.OLS(trailing_window['avg_error'], trailing_window[['week_num', 'intercept_constant']], hasconst=True).fit()
        update_val = reg.params.intercept_constant + reg.params.week_num * trailing_window['week_num'].max()
        prev_level = hfa_df_temp.iloc[index - 1]['level']
        a_ = a * min(index / reg_weeks, 1)
        hfa_df_temp.loc[index, 'level'] = a_ * update_val + (1 - a_) * prev_level

    hfa_df_temp['rolling_hfa'] = hfa_df_temp['level'].shift(1).fillna(2.50)
    hfa_df = pd.merge(hfa_df[['season', 'week']], hfa_df_temp[['season', 'week', 'rolling_hfa']], on=['season', 'week'], how='left')
    hfa_df['rolling_hfa'] = hfa_df['rolling_hfa'].fillna(2.50).ffill()
    hfa_df.loc[hfa_df['season'] == 2020, 'rolling_hfa'] = 0.25
    return hfa_df


