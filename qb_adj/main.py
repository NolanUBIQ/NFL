from data.loader import DataLoader
from model.qb_model import QBModel
from airtable.wrapper import AirtableWrapper
from model.game_context import GameContext
from model.elo_constructor import EloConstructor
import json
from config import config


def main():
    # Load config

    # Load data
    # Load data
    loader = DataLoader()
    model_df = loader.model_df
    games_df = loader.games

    # Initialize model
    model = QBModel(model_df.copy(), config)
    model.run_model()

    # Setup Airtable sync
    at_config_path = "D:/NFL/qb_adj/airtable/secrets.json"
    with open(at_config_path, "r") as f:
        at_config = json.load(f)

    at_wrapper = AirtableWrapper(model_df, at_config["airtable"], perform_starter_update=True)
    #at_wrapper.update_qb_table()
    #at_wrapper.update_qb_options()
    #at_wrapper.update_qb_table_games_started()
    #at_wrapper.update_starters()

    # Score model
    #record = model.score_model()
    #print("Model performance:", record)

    # Build new Elo file
    constructor = EloConstructor(
        games=games_df,
        qb_model=model,
        at_wrapper=at_wrapper,
        export_loc="D:/NFL/qb_adj/data/new_elo_file.csv"
    )
    constructor.determine_new_games()
    constructor.add_qbs_to_new_games()
    constructor.get_next_games()
    constructor.add_starters()
    constructor.add_team_values()

    print(constructor.next_games.columns)
    constructor.merge_new_and_next()

    # Save new Elo file
    if constructor.new_file_games is not None:
        constructor.new_file_games.to_csv(constructor.export_loc, index=False)
        print(f"New Elo file exported to: {constructor.export_loc}")
    else:
        print("No new games to export.")


if __name__ == "__main__":
    main()