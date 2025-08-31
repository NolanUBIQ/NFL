def compute_bye_week_modifiers(df, home_bye_week=0.287, away_bye_week=-0.778):
    """
    Computes bye week modifiers for home and away teams based on time since last game.

    Args:
        df (pd.DataFrame): Game-level DataFrame with prev_week columns.
        home_bye_week (float): Multiplier for home bye week effect.
        away_bye_week (float): Multiplier for away bye week effect.

    Returns:
        pd.DataFrame: Modified DataFrame with 'home_bye_mod' and 'away_bye_mod' columns.
    """
    df = df.copy()

    # Home team bye modifier
    df['home_bye_mod'] = np.where(
        df['week'] == 1,
        0,
        np.where(
            df['week'] > df['prev_week_home'] + 1,
            df['hfa_mod'] * home_bye_week,
            0
        )
    )

    # Away team bye modifier
    df['away_bye_mod'] = np.where(
        df['week'] == 1,
        0,
        np.where(
            df['week'] > df['prev_week_away'] + 1,
            df['hfa_mod'] * away_bye_week,
            0
        )
    )

    return df
