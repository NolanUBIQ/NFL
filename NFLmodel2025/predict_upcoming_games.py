import pandas as pd


def predict_upcoming_games(df, elo_dict, config, spread_dict, elo_to_prob_fn):
    """
    Generates predictions for unplayed games by computing model_home_line
    based on current Elo ratings and modifiers.

    Args:
        df (pd.DataFrame): DataFrame of schedule data.
        elo_dict (dict): Dictionary of current team Elo ratings.
        config (dict): Model configuration.
        spread_dict (dict): Mapping from win probabilities to spreads.
        elo_to_prob_fn (function): Converts Elo diff to win prob.

    Returns:
        df (pd.DataFrame): Input DataFrame with new model_home_line column.
    """
    df = df[df['home_score'].isnull() & df['away_score'].isnull()].copy()

    model_lines = []
    for _, row in df.iterrows():
        home = row['home_team']
        away = row['away_team']

        # Starting Elo
        starting_home_elo = elo_dict.get(home, {}).get('elo', 1500)
        starting_away_elo = elo_dict.get(away, {}).get('elo', 1500)

        # Elo diff and adjustments
        elo_diff = starting_home_elo - starting_away_elo
        elo_diff += row.get('hfa_mod', 0)
        elo_diff += row.get('surface_mod', 0)
        elo_diff += row.get('time_mod', 0)
        elo_diff += row.get('div_mod', 0)
        elo_diff += config['qb_weight'] * row.get('qb_adjustment', 0)

        if row.get('is_playoffs') == 1:
            elo_diff *= (1 + config['playoff_boost'])

        # Convert to win prob and model spread
        win_prob = elo_to_prob_fn(elo_diff, config['z'])
        model_spread = spread_dict.get(round(win_prob, 3), 0)

        # Store prediction values
        row['model_home_line'] = model_spread
        row['win_prob'] = win_prob
        row['elo_diff'] = elo_diff
        model_lines.append(row)

    return pd.DataFrame(model_lines)
