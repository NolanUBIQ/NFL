import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import ast
from scipy.stats import norm
import nfl_data_py as nfl

# Load 2024 game data
df_nfl_games = nfl.import_schedules([2024])
df_nfl_games["actual_total"] = df_nfl_games["home_score"] + df_nfl_games["away_score"]

# Regulation-only scores
df_nfl_games["home_score_reg"] = np.where(
    df_nfl_games.overtime == 1,
    df_nfl_games[["home_score", "away_score"]].min(axis=1),
    df_nfl_games["home_score"]
)
df_nfl_games["away_score_reg"] = np.where(
    df_nfl_games.overtime == 1,
    df_nfl_games[["home_score", "away_score"]].min(axis=1),
    df_nfl_games["away_score"]
)

# Team indexing
teams = sorted(df_nfl_games["home_team"].unique())
team_to_idx = {team: i for i, team in enumerate(teams)}
df_nfl_games["i_home"] = df_nfl_games["home_team"].map(team_to_idx)
df_nfl_games["i_away"] = df_nfl_games["away_team"].map(team_to_idx)

# Load prior file
priors_df = pd.read_csv("D:/NFL/Totals/NFL_priors.csv").iloc[:, 1:]

# ============ QB Adjustment Table ============
qb_adjustments = pd.DataFrame(columns=["week", "team", "qb_adjustment"])

def add_qb_adjustment(team, week, value):
    global qb_adjustments
    qb_adjustments = pd.concat([
        qb_adjustments,
        pd.DataFrame([{"week": week, "team": team, "qb_adjustment": value}])
    ], ignore_index=True)

def get_qb_adjustment(team, week):
    row = qb_adjustments[(qb_adjustments["team"] == team) & (qb_adjustments["week"] == week)]
    return row["qb_adjustment"].values[0] if not row.empty else 0.0


def mle_fit(data):
    return norm.fit(data)

def model_week(df_week, priors):
    home_scores = np.array(df_week["home_score_reg"])
    away_scores = np.array(df_week["away_score_reg"])
    home_team = np.array(df_week["i_home"])
    away_team = np.array(df_week["i_away"])
    home_qb_adjust = np.array(df_week["qb_adjust_home"])
    away_qb_adjust = np.array(df_week["qb_adjust_away"])

    with pm.Model() as model:
        offence_star = pm.Normal("offence_star", mu=ast.literal_eval(priors["off_mu"].iloc[0]),
                                 sigma=ast.literal_eval(priors["off_sd"].iloc[0]), shape=32)
        defence_star = pm.Normal("defence_star", mu=ast.literal_eval(priors["def_mu"].iloc[0]),
                                 sigma=ast.literal_eval(priors["def_sd"].iloc[0]), shape=32)

        delta_o = pm.Normal("delta_o", mu=0.0, sigma=0.5, shape=32)
        delta_d = pm.Normal("delta_d", mu=0.0, sigma=0.5, shape=32)

        team_off = pm.Deterministic("offence", offence_star + delta_o - pm.math.sum(offence_star + delta_o) / 32)
        team_def = pm.Deterministic("defence", defence_star + delta_d - pm.math.sum(defence_star + delta_d) / 32)

        avg_pts = 21.9
        home_field = 1.8

        mu_home = team_off[home_team] - team_def[away_team] + avg_pts + home_field
        mu_away = team_off[away_team] - team_def[home_team] + avg_pts

        pm.Normal("obs_home", mu=mu_home, sigma=7, observed=home_scores + home_qb_adjust)
        pm.Normal("obs_away", mu=mu_away, sigma=7, observed=away_scores + away_qb_adjust)

        trace = pm.sample(2000, tune=1000, chains=1, cores=1, progressbar=False)

        off_mu = [mle_fit(trace.posterior["offence"].values[..., i].flatten())[0] for i in range(32)]
        off_sd = [mle_fit(trace.posterior["offence"].values[..., i].flatten())[1] for i in range(32)]
        def_mu = [mle_fit(trace.posterior["defence"].values[..., i].flatten())[0] for i in range(32)]
        def_sd = [mle_fit(trace.posterior["defence"].values[..., i].flatten())[1] for i in range(32)]

        return {
            "off_mu": str(off_mu),
            "off_sd": str(off_sd),
            "def_mu": str(def_mu),
            "def_sd": str(def_sd)
        }

all_preds = []
max_week = df_nfl_games["week"].max()

for week in range(1, max_week + 1):
    train_df = df_nfl_games[df_nfl_games["week"] < week].copy()
    test_df = df_nfl_games[df_nfl_games["week"] == week].copy()

    # Apply QB adjustments
    for df in [train_df, test_df]:
        df["qb_adjust_home"] = df.apply(lambda row: get_qb_adjustment(row["home_team"], row["week"]), axis=1)
        df["qb_adjust_away"] = df.apply(lambda row: get_qb_adjustment(row["away_team"], row["week"]), axis=1)

    # Fit model and update priors
    if not train_df.empty:
        posteriors = model_week(train_df, priors_df[priors_df["week"] == week - 1])
        posteriors["week"] = week
        priors_df = pd.concat([priors_df, pd.DataFrame([posteriors])], ignore_index=True)

    # Make predictions
    if not test_df.empty:
        posterior = posteriors
        off_mu = ast.literal_eval(posterior["off_mu"])
        def_mu = ast.literal_eval(posterior["def_mu"])

        preds = []
        for _, row in test_df.iterrows():
            i_home = row["i_home"]
            i_away = row["i_away"]
            pred_total = (
                off_mu[i_home] - def_mu[i_away] +
                off_mu[i_away] - def_mu[i_home] +
                2 * 21.9 + 1.8 - row["qb_adjust_home"] - row["qb_adjust_away"]
            )
            preds.append(pred_total)

        test_df["predicted_total"] = preds
        all_preds.append(test_df[["game_id", "week", "actual_total", "predicted_total"]])

# Combine predictions and compute MAE
results_df = pd.concat(all_preds).dropna()
mae = np.mean(np.abs(results_df["predicted_total"] - results_df["actual_total"]))
print(f"Final MAE: {mae:.2f}")

# Save results and updated priors
results_df.to_csv("D:/NFL/Totals/NFL_predicted_totals_full_season.csv", index=False)
priors_df.to_csv("D:/NFL/Totals/NFL_priors.csv", index=False)
