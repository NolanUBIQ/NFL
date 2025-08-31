def compute_regression_factor(
    model_spread, market_spread,
    model_se_home, model_se_away,
    market_se_home, market_se_away,
    config
):
    """
    Computes a dynamic regression weight for model-to-market adjustment.

    Returns:
        float: regression multiplier between 0 and 1
    """
    # Market and model RMSE difference
    model_rmse = (model_se_home ** 0.5 + model_se_away ** 0.5) / 2
    market_rmse = (market_se_home ** 0.5 + market_se_away ** 0.5) / 2
    rmse_dif = model_rmse - market_rmse

    spread_delta_open = abs(model_spread - market_spread)

    # Deflator baseline term
    mr_deflator_factor = (
        4 / (1 + (config["spread_delta_base"] * spread_delta_open ** 2)) +
        spread_delta_open / 14
    )
    mr_factor = mr_deflator_factor

    # Long line inflator
    if market_spread < -7.5 and model_spread > market_spread:
        mr_factor *= (1 + config["long_line_inflator"])

    # Hook detection (whole number spreads)
    if market_spread % 1 == 0:
        hook_inflator = 1 + config["hook_certainty"]
        mr_factor *= hook_inflator

    # RMSE-based inflator if market and model are far apart
    if spread_delta_open > 1:
        mr_factor *= (1 + rmse_dif / config["rmse_base"])

    # Final bounded multiplier
    return max(config["min_mr"], min(1, config["market_regression"] * mr_factor))
