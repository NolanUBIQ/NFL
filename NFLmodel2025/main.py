# main.py

from config.settings import (
    team_standardization, surface_repl, pbp_surface_repl,
    timezones, timezone_overrides, elo_config
)
from config.runtime import current_week, current_season

from data.load_data import load_schedules, load_qb_elo, load_wepa
from data.preprocess import standardize_teams, generate_result_column
from features.hfa import calc_rolling_hfa
from features.surface import define_field_surfaces
from features.time_advantage import define_time_advantages
from features.compute_bye_week_modifiers import compute_bye_week_modifiers
from features.qb_adjustment import compute_qb_adjustment
from model.elo import elo_to_prob, prob_to_elo, shift_calc_helper
from model.predict_upcoming_games import predict_upcoming_games
from model.rolling_stats import add_game_numbers, apply_rolling_epa
from output.write_results import save_dataframe
from utils.helpers import log_stage
from update import update_elo_model

def main():
    log_stage("Loading schedules...")
    schedule_df = load_schedules(seasons=list(range(1999, 2026)))


    log_stage("Loading QB and WEPA data...")
    qb_df = load_qb_elo("data_sources/qb_elos.csv")
    wepa_df = load_wepa("data_sources/wepa_game_flat_file.csv")

    log_stage("Calculating rolling HFA...")
    hfa_df = calc_rolling_hfa(schedule_df, level_weeks=10, reg_weeks=140)

    log_stage("Defining field surfaces...")
    fields_df = define_field_surfaces(schedule_df, pbp_surface_repl)

    log_stage("Computing timezone-based home advantage...")
    schedule_df['home_time_advantage'] = define_time_advantages(schedule_df, timezones, timezone_overrides)

    log_stage("Computing bye week adjustments...")
    schedule_df = compute_bye_week_modifiers(schedule_df)

    log_stage("Generating result column...")
    schedule_df = generate_result_column(schedule_df)

    log_stage("Filtering to current season/week...")
    current_df = schedule_df[
        (schedule_df["season"] == current_season) &
        (schedule_df["week"] <= current_week)
    ].copy()

    unplayed_df = schedule_df[
        (schedule_df["season"] == current_season) &
        (schedule_df["week"] == current_week) &
        (schedule_df["home_score"].isnull())
        ].copy()




    log_stage("Adding net QB adjustment...")
    current_df["qb_adjustment"] = current_df.apply(
        lambda row: compute_qb_adjustment(row, elo_config["qb_weight"]), axis=1
    )

    log_stage("Running Elo update loop...")
    initial_elos = {}  # could load from pickle
    spread_mult_dict = {}  # should load from CSV


    updated_df, updated_elos = update_elo_model(
        current_df, initial_elos, elo_config, spread_mult_dict, prob_to_elo, elo_to_prob
    )

    predictions_df = predict_upcoming_games(
        unplayed_df, updated_elos, elo_config, spread_mult_dict, elo_to_prob
    )



    log_stage("Saving updated Elo dataframe...")
    save_dataframe(updated_df, f"output/elo_results_week{current_week}.csv")
    log_stage("Saving prediction dataframe...")
    save_dataframe(predictions_df, f"output/predictions_week{current_week}.csv")

    log_stage("Pipeline complete.")


if __name__ == "__main__":
    main()