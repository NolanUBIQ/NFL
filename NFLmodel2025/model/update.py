import pandas as pd
from elo import shift_calc_helper
from elo import calc_weighted_shift

def update_elo_model(df, elo_dict, config, spread_dict, prob_to_elo_fn, elo_to_prob_fn):
    """
    Updates the Elo model for each row in the game dataframe.

    Args:
        df (pd.DataFrame): DataFrame containing one row per game.
        elo_dict (dict): Dictionary storing current Elo ratings per team.
        config (dict): Configuration values for Elo model behavior.
        spread_dict (dict): Mapping from win probabilities to point spreads.
        prob_to_elo_fn (function): Function to convert win prob to Elo diff.
        elo_to_prob_fn (function): Function to convert Elo diff to win prob.

    Returns:
        updated_df (pd.DataFrame): Updated game-level results.
        updated_elo_dict (dict): Dictionary of updated team Elos.
    """
    updated_elos = []

    df = df[~df['home_score'].isnull() & ~df['away_score'].isnull()].copy()

    for _, row in df.iterrows():
        home = row['home_team']
        away = row['away_team']
        game_id = row['game_id']

        # Starting Elo
        starting_home_elo = elo_dict.get(home, {}).get('elo', 1500)
        starting_away_elo = elo_dict.get(away, {}).get('elo', 1500)

        # Base Elo difference
        elo_diff = starting_home_elo - starting_away_elo
        elo_diff += row.get('hfa_mod', 0)
        elo_diff += row.get('surface_mod', 0)
        elo_diff += row.get('time_mod', 0)
        elo_diff += row.get('div_mod', 0)
        elo_diff += config['qb_weight'] * row.get('qb_adjustment', 0)

        if row.get('is_playoffs') == 1:
            elo_diff *= (1 + config['playoff_boost'])

        # Convert to win prob and spread
        win_prob = elo_to_prob_fn(elo_diff, config['z'])
        model_spread = spread_dict.get(round(win_prob, 3), 0)

        # Store model line in row for transparency
        row['model_home_line'] = model_spread
        row['market_home_line'] = row.get('spread_line', 0)

        # --- Elo Shift Calculation ---

        home_shift = calc_weighted_shift(
            margin_array=[
                (row.get('home_margin'), config['margin_weight']),
                (row.get('home_net_wepa_point_margin'), config['wepa_weight']),
                (row.get('home_pff_point_margin'), config['pff_weight'])
            ],
            model_line=model_spread,
            market_line=row['market_home_line'],
            k=config['k'],
            b=config['b'],
            market_resist_factor=config['market_resist_factor'],
            is_home=True
        )

        away_shift = calc_weighted_shift(
            margin_array=[
                (row.get('away_margin'), config['margin_weight']),
                (row.get('away_net_wepa_point_margin'), config['wepa_weight']),
                (row.get('away_pff_point_margin'), config['pff_weight'])
            ],
            model_line=model_spread,
            market_line=row['market_home_line'],
            k=config['k'],
            b=config['b'],
            market_resist_factor=config['market_resist_factor'],
            is_home=False
        )

        # --- Update Elo Dictionary ---
        new_home_elo = starting_home_elo + home_shift
        new_away_elo = starting_away_elo + away_shift
        elo_dict[home] = {"elo": new_home_elo}
        elo_dict[away] = {"elo": new_away_elo}

        # --- Record ---
        updated_elos.append({
            "game_id": game_id,
            "home_team": home,
            "away_team": away,
            "home_elo_pre": starting_home_elo,
            "away_elo_pre": starting_away_elo,
            "elo_diff": elo_diff,
            "home_shift": home_shift,
            "away_shift": away_shift,
            "home_elo_post": new_home_elo,
            "away_elo_post": new_away_elo,
            "win_prob": win_prob,
            "model_spread": model_spread,
            "market_spread": row['market_home_line']
        })

    return pd.DataFrame(updated_elos), elo_dict