# config/settings.py

team_standardization = {
    "ARI": "ARI", "ATL": "ATL", "BAL": "BAL", "BUF": "BUF", "CAR": "CAR",
    "CHI": "CHI", "CIN": "CIN", "CLE": "CLE", "DAL": "DAL", "DEN": "DEN",
    "DET": "DET", "GB": "GB", "HOU": "HOU", "IND": "IND", "JAC": "JAX",
    "JAX": "JAX", "KC": "KC", "LA": "LAR", "LAC": "LAC", "LV": "OAK",
    "MIA": "MIA", "MIN": "MIN", "NE": "NE", "NO": "NO", "NYG": "NYG",
    "NYJ": "NYJ", "OAK": "OAK", "PHI": "PHI", "PIT": "PIT", "SD": "LAC",
    "SEA": "SEA", "SF": "SF", "STL": "LAR", "TB": "TB", "TEN": "TEN", "WAS": "WAS"
}

surface_repl = {
    "fieldturf": "artificial", "matrixturf": "artificial", "sportturf": "artificial",
    "astroturf": "artificial", "astroplay": "artificial", "a_turf": "artificial",
    "fieldturf ": "artificial", "grass": "natural", "dessograss": "natural"
}
pbp_surface_repl = surface_repl.copy()

timezones = {
    "ARI": "MT", "ATL": "ET", "BAL": "ET", "BUF": "ET", "CAR": "ET",
    "CHI": "CT", "CIN": "ET", "CLE": "ET", "DAL": "CT", "DEN": "MT",
    "DET": "ET", "GB": "CT", "HOU": "CT", "IND": "ET", "JAX": "ET",
    "KC": "CT", "LAR": "PT", "LAC": "PT", "MIA": "ET", "MIN": "CT",
    "NE": "ET", "NO": "CT", "NYG": "ET", "NYJ": "ET", "OAK": "PT",
    "PHI": "ET", "PIT": "ET", "SEA": "PT", "SF": "PT", "TB": "ET",
    "TEN": "CT", "WAS": "ET"
}

timezone_overrides = {
    "LAR": {"season": 2015, "tz_override": "CT"}
}

elo_config = {
    "z": 401.62,
    "k": 9.114,
    "market_resist_factor": 1.5039,
    "b": 10,
    "qb_weight": 1.0,
    "playoff_boost": 0.1,
    "rmse_base": 2.993,
    "min_mr": 0.1298,
    "market_regression": 0.90,
    "nfelo_span": 4,
    "se_span": 8.346,
    "spread_delta_base": 1.1,
    "long_line_inflator": 0.5355,
    "hook_certainty": 0,
    "weight_vector": [0.74, 0.15, 0.11]  # margin, wepa, pff
}
