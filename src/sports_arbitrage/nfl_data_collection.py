"""
NFL Data Collection Module

This module collects NFL game data and play-by-play data using nfl_data_py library.
It aggregates play-by-play data to game-level statistics and matches with betting odds data.
"""

import pandas as pd
import numpy as np
import nfl_data_py as nfl
from typing import Dict, List, Tuple
import warnings

warnings.filterwarnings("ignore")


# Team name mapping from odds data to NFL data format
TEAM_NAME_MAPPING = {
    "Arizona Cardinals": "ARI",
    "Atlanta Falcons": "ATL",
    "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR",
    "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN",
    "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN",
    "Detroit Lions": "DET",
    "Green Bay Packers": "GB",
    "Houston Texans": "HOU",
    "Indianapolis Colts": "IND",
    "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC",
    "Las Vegas Raiders": "LV",
    "Oakland Raiders": "LV",  # Historical name
    "Los Angeles Chargers": "LAC",
    "San Diego Chargers": "LAC",  # Historical name
    "Los Angeles Rams": "LAR",
    "St. Louis Rams": "LAR",  # Historical name
    "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN",
    "New England Patriots": "NE",
    "New Orleans Saints": "NO",
    "New York Giants": "NYG",
    "New York Jets": "NYJ",
    "Philadelphia Eagles": "PHI",
    "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF",
    "Seattle Seahawks": "SEA",
    "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN",
    "Washington Commanders": "WAS",
    "Washington Football Team": "WAS",  # Historical name
    "Washington Redskins": "WAS",  # Historical name
}


def fetch_nfl_game_data(years: List[int]) -> pd.DataFrame:
    """
    Fetch NFL game-level data for specified years.

    Args:
        years: List of years to fetch data for (e.g., [2020, 2021, 2022])

    Returns:
        DataFrame with game-level data
    """
    print(f"Fetching NFL schedule data for years: {years}")
    schedules = nfl.import_schedules(years)

    # Keep relevant columns
    game_cols = [
        "game_id",
        "season",
        "game_type",
        "week",
        "gameday",
        "weekday",
        "gametime",
        "away_team",
        "home_team",
        "away_score",
        "home_score",
        "away_rest",
        "home_rest",
        "away_moneyline",
        "home_moneyline",
        "spread_line",
        "away_spread_odds",
        "home_spread_odds",
        "total_line",
        "under_odds",
        "over_odds",
        "div_game",
        "roof",
        "surface",
        "temp",
        "wind",
        "home_coach",
        "away_coach",
        "stadium",
    ]

    # Filter to only include columns that exist
    available_cols = [col for col in game_cols if col in schedules.columns]
    schedules = schedules[available_cols].copy()

    # Filter to only regular season and playoff games
    schedules = schedules[schedules["game_type"].isin(["REG", "WC", "DIV", "CON", "SB"])]

    print(f"Fetched {len(schedules)} games")
    return schedules


def fetch_nfl_pbp_data(years: List[int]) -> pd.DataFrame:
    """
    Fetch NFL play-by-play data for specified years.

    Args:
        years: List of years to fetch data for

    Returns:
        DataFrame with play-by-play data
    """
    print(f"Fetching NFL play-by-play data for years: {years}")
    print("This may take a few minutes...")

    pbp_data = nfl.import_pbp_data(years)

    print(f"Fetched play-by-play data: {len(pbp_data)} plays")
    return pbp_data


def aggregate_pbp_to_game_stats(pbp_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate play-by-play data to game-level statistics.

    Calculates offensive and defensive statistics for each team in each game.

    Args:
        pbp_df: DataFrame with play-by-play data

    Returns:
        DataFrame with aggregated game-level statistics
    """
    print("Aggregating play-by-play data to game-level statistics...")

    # Helper function to safely sum a column
    def safe_sum(df, col):
        if col not in df.columns:
            return 0
        try:
            # Convert to numeric, coercing errors to NaN, then sum (NaN values are ignored)
            return pd.to_numeric(df[col], errors="coerce").sum()
        except:
            return 0

    # Helper function to safely calculate mean
    def safe_mean(df, col):
        if col not in df.columns or len(df) == 0:
            return 0
        try:
            # Convert to numeric, coercing errors to NaN, then calculate mean
            return pd.to_numeric(df[col], errors="coerce").mean()
        except:
            return 0

    # Filter to only plays that count (remove timeouts, kickoffs, etc.)
    pbp_df = pbp_df[
        pbp_df["play_type"].isin(["pass", "run", "punt", "field_goal", "extra_point"])
    ].copy()

    game_stats = []

    for game_id in pbp_df["game_id"].unique():
        game_plays = pbp_df[pbp_df["game_id"] == game_id]

        if len(game_plays) == 0:
            continue

        home_team = game_plays["home_team"].iloc[0]
        away_team = game_plays["away_team"].iloc[0]

        # Initialize stats dictionary
        stats = {
            "game_id": game_id,
            "home_team": home_team,
            "away_team": away_team,
        }

        # Home team offensive stats
        home_off_plays = game_plays[game_plays["posteam"] == home_team]
        stats["home_total_plays"] = len(home_off_plays)
        stats["home_pass_attempts"] = len(home_off_plays[home_off_plays["play_type"] == "pass"])
        stats["home_rush_attempts"] = len(home_off_plays[home_off_plays["play_type"] == "run"])
        stats["home_total_yards"] = safe_sum(home_off_plays, "yards_gained")
        stats["home_yards_per_play"] = safe_mean(home_off_plays, "yards_gained")
        stats["home_turnovers"] = safe_sum(home_off_plays, "turnover")
        stats["home_first_downs"] = safe_sum(home_off_plays, "first_down")
        stats["home_third_down_conv"] = safe_sum(home_off_plays, "third_down_converted")
        stats["home_third_down_att"] = safe_sum(home_off_plays, "third_down_failed") + safe_sum(
            home_off_plays, "third_down_converted"
        )
        stats["home_fourth_down_conv"] = safe_sum(home_off_plays, "fourth_down_converted")
        stats["home_fourth_down_att"] = safe_sum(home_off_plays, "fourth_down_failed") + safe_sum(
            home_off_plays, "fourth_down_converted"
        )

        # Home team defensive stats (away team offense)
        away_off_plays = game_plays[game_plays["posteam"] == away_team]
        stats["home_def_yards_allowed"] = safe_sum(away_off_plays, "yards_gained")
        stats["home_def_yards_per_play"] = safe_mean(away_off_plays, "yards_gained")
        stats["home_turnovers_forced"] = safe_sum(away_off_plays, "turnover")

        # Away team offensive stats
        stats["away_total_plays"] = len(away_off_plays)
        stats["away_pass_attempts"] = len(away_off_plays[away_off_plays["play_type"] == "pass"])
        stats["away_rush_attempts"] = len(away_off_plays[away_off_plays["play_type"] == "run"])
        stats["away_total_yards"] = safe_sum(away_off_plays, "yards_gained")
        stats["away_yards_per_play"] = safe_mean(away_off_plays, "yards_gained")
        stats["away_turnovers"] = safe_sum(away_off_plays, "turnover")
        stats["away_first_downs"] = safe_sum(away_off_plays, "first_down")
        stats["away_third_down_conv"] = safe_sum(away_off_plays, "third_down_converted")
        stats["away_third_down_att"] = safe_sum(away_off_plays, "third_down_failed") + safe_sum(
            away_off_plays, "third_down_converted"
        )
        stats["away_fourth_down_conv"] = safe_sum(away_off_plays, "fourth_down_converted")
        stats["away_fourth_down_att"] = safe_sum(away_off_plays, "fourth_down_failed") + safe_sum(
            away_off_plays, "fourth_down_converted"
        )

        # Away team defensive stats (home team offense)
        stats["away_def_yards_allowed"] = safe_sum(home_off_plays, "yards_gained")
        stats["away_def_yards_per_play"] = safe_mean(home_off_plays, "yards_gained")
        stats["away_turnovers_forced"] = safe_sum(home_off_plays, "turnover")

        # Time of possession (in seconds)
        stats["home_time_of_possession"] = safe_sum(home_off_plays, "drive_time_of_possession")
        stats["away_time_of_possession"] = safe_sum(away_off_plays, "drive_time_of_possession")

        # Red zone efficiency
        if "yardline_100" in home_off_plays.columns:
            home_rz = home_off_plays[home_off_plays["yardline_100"] <= 20]
            away_rz = away_off_plays[away_off_plays["yardline_100"] <= 20]
            stats["home_red_zone_plays"] = len(home_rz)
            stats["away_red_zone_plays"] = len(away_rz)
        else:
            stats["home_red_zone_plays"] = 0
            stats["away_red_zone_plays"] = 0

        # Expected Points Added (EPA)
        stats["home_epa_total"] = safe_sum(home_off_plays, "epa")
        stats["home_epa_per_play"] = safe_mean(home_off_plays, "epa")
        stats["away_epa_total"] = safe_sum(away_off_plays, "epa")
        stats["away_epa_per_play"] = safe_mean(away_off_plays, "epa")

        # Win Probability Added (WPA)
        stats["home_wpa_total"] = safe_sum(home_off_plays, "wpa")
        stats["away_wpa_total"] = safe_sum(away_off_plays, "wpa")

        game_stats.append(stats)

    stats_df = pd.DataFrame(game_stats)
    print(f"Aggregated statistics for {len(stats_df)} games")

    return stats_df


def match_with_odds_data(
    nfl_games: pd.DataFrame, nfl_pbp_stats: pd.DataFrame, odds_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Match NFL data with betting odds data.

    Args:
        nfl_games: DataFrame with NFL game metadata
        nfl_pbp_stats: DataFrame with aggregated play-by-play statistics
        odds_df: DataFrame with betting odds

    Returns:
        Combined DataFrame with NFL data and odds
    """
    print("Matching NFL data with betting odds...")

    # Parse game dates
    nfl_games["gameday"] = pd.to_datetime(nfl_games["gameday"])

    # Create reverse mapping (abbreviation to full name)
    abbr_to_full = {v: k for k, v in TEAM_NAME_MAPPING.items()}

    # Convert team abbreviations to full names
    nfl_games["home_team_full"] = nfl_games["home_team"].map(abbr_to_full)
    nfl_games["away_team_full"] = nfl_games["away_team"].map(abbr_to_full)

    # Merge NFL games with play-by-play stats
    nfl_combined = nfl_games.merge(
        nfl_pbp_stats, on=["game_id", "home_team", "away_team"], how="left"
    )

    # Get unique games from odds data
    odds_games = (
        odds_df.groupby(["game_id", "home_team", "away_team", "commence_time"])
        .first()
        .reset_index()
    )
    odds_games["commence_date"] = pd.to_datetime(odds_games["commence_time"]).dt.date

    # Match on team names and dates
    matched_games = []

    for _, odds_row in odds_games.iterrows():
        odds_date = odds_row["commence_date"]
        home_team = odds_row["home_team"]
        away_team = odds_row["away_team"]

        # Find matching NFL game
        nfl_match = nfl_combined[
            (nfl_combined["gameday"].dt.date == odds_date)
            & (nfl_combined["home_team_full"] == home_team)
            & (nfl_combined["away_team_full"] == away_team)
        ]

        if len(nfl_match) > 0:
            # Found a match!
            nfl_data = nfl_match.iloc[0].to_dict()
            odds_data = odds_row.to_dict()

            # Combine the data
            combined = {**odds_data, **nfl_data}
            matched_games.append(combined)

    matched_df = pd.DataFrame(matched_games)
    print(f"Successfully matched {len(matched_df)} games out of {len(odds_games)} odds records")

    return matched_df


def collect_and_save_nfl_data(
    odds_filepath: str, output_filepath: str, years: List[int] = [2020, 2021, 2022, 2023, 2024]
):
    """
    Main function to collect NFL data, aggregate statistics, and save to CSV.

    Args:
        odds_filepath: Path to the odds CSV file
        output_filepath: Path to save the combined data
        years: List of years to collect data for
    """
    print("=" * 80)
    print("NFL DATA COLLECTION")
    print("=" * 80)
    print()

    # Load odds data
    print("Loading odds data...")
    odds_df = pd.read_csv(odds_filepath)
    odds_df["commence_time"] = pd.to_datetime(odds_df["commence_time"])
    print(f"Loaded {len(odds_df)} odds records")
    print()

    # Fetch NFL game data
    nfl_games = fetch_nfl_game_data(years)
    print()

    # Fetch play-by-play data
    nfl_pbp = fetch_nfl_pbp_data(years)
    print()

    # Aggregate play-by-play to game stats
    nfl_pbp_stats = aggregate_pbp_to_game_stats(nfl_pbp)
    print()

    # Match with odds data
    combined_df = match_with_odds_data(nfl_games, nfl_pbp_stats, odds_df)
    print()

    # Save to CSV
    print(f"Saving combined data to {output_filepath}")
    combined_df.to_csv(output_filepath, index=False)
    print(f"Saved {len(combined_df)} game records with {len(combined_df.columns)} features")
    print()

    # Print summary statistics
    print("Summary of NFL features added:")
    print("-" * 80)

    nfl_feature_cols = [
        col
        for col in combined_df.columns
        if col.startswith(("home_", "away_"))
        and col not in ["home_team", "away_team", "home_avg_odds", "away_avg_odds"]
    ]

    print(f"Total NFL features: {len(nfl_feature_cols)}")
    print("\nSample features:")
    for col in nfl_feature_cols[:20]:
        non_null = combined_df[col].notna().sum()
        print(f"  - {col}: {non_null}/{len(combined_df)} non-null values")

    print()
    print("=" * 80)
    print("DATA COLLECTION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    # Example usage
    odds_file = "../../data/odds_2020_2024_combined.csv"
    output_file = "../../data/nfl_games_with_stats.csv"

    collect_and_save_nfl_data(odds_file, output_file, years=[2020, 2021, 2022, 2023, 2024])
