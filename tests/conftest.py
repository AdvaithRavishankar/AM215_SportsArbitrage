"""
Pytest configuration and shared fixtures for sports betting arbitrage tests.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


@pytest.fixture
def sample_games_df():
    """Create a sample games DataFrame for testing."""
    rng = np.random.default_rng(42)
    n_games = 100

    teams = ["Team_A", "Team_B", "Team_C", "Team_D", "Team_E"]

    games = []
    base_date = datetime(2023, 1, 1)

    for i in range(n_games):
        home_team = rng.choice(teams)
        away_team = rng.choice([t for t in teams if t != home_team])

        games.append(
            {
                "game_id": f"game_{i}",
                "home_team": home_team,
                "away_team": away_team,
                "commence_time": base_date + timedelta(days=i),
                "home_won": rng.choice([True, False]),
                "home_avg_odds": rng.choice([-200, -150, -110, 100, 150, 200]),
                "away_avg_odds": rng.choice([-200, -150, -110, 100, 150, 200]),
            }
        )

    return pd.DataFrame(games)


@pytest.fixture
def sample_odds_df():
    """Create a sample odds DataFrame for testing."""
    rng = np.random.default_rng(42)

    games = []
    teams = ["Team_A", "Team_B"]
    sportsbooks = ["DraftKings", "FanDuel", "BetMGM"]

    for i in range(10):
        game_id = f"game_{i}"
        for team in teams:
            for sportsbook in sportsbooks:
                games.append(
                    {
                        "game_id": game_id,
                        "home_team": teams[0],
                        "away_team": teams[1],
                        "team": team,
                        "sportsbook": sportsbook,
                        "odds": rng.choice([-200, -150, -110, 100, 150, 200]),
                        "commence_time": datetime(2023, 1, 1) + timedelta(days=i),
                    }
                )

    return pd.DataFrame(games)


@pytest.fixture
def sample_predictions():
    """Create sample model predictions for testing."""
    rng = np.random.default_rng(42)
    return rng.uniform(0.2, 0.8, 50)


@pytest.fixture
def sample_actuals():
    """Create sample actual outcomes for testing."""
    rng = np.random.default_rng(42)
    return rng.choice([0, 1], 50)


@pytest.fixture
def sample_roi_results():
    """Create sample ROI results for testing."""
    return {
        "ELO": {"roi": 5.0, "profit": 500, "total_bet": 10000, "num_bets": 100},
        "XGBoost": {"roi": 3.0, "profit": 300, "total_bet": 10000, "num_bets": 100},
        "Random Forest": {"roi": 4.0, "profit": 400, "total_bet": 10000, "num_bets": 100},
        "Rank Centrality": {"roi": -2.0, "profit": -200, "total_bet": 10000, "num_bets": 100},
    }


@pytest.fixture
def trained_elo_model(sample_games_df):
    """Create a trained ELO model for testing."""
    from sports_arbitrage.models.elo import ELOModel

    model = ELOModel()
    model.fit(sample_games_df)
    return model


@pytest.fixture
def trained_xgboost_model(sample_games_df):
    """Create a trained XGBoost model for testing."""
    from sports_arbitrage.models.xgboost_model import XGBoostModel

    model = XGBoostModel(n_estimators=10)  # Use fewer estimators for testing speed
    model.fit(sample_games_df)
    return model


@pytest.fixture
def trained_random_forest_model(sample_games_df):
    """Create a trained Random Forest model for testing."""
    from sports_arbitrage.models.random_forest import RandomForestModel

    model = RandomForestModel(n_estimators=10)  # Use fewer estimators for testing speed
    model.fit(sample_games_df)
    return model


@pytest.fixture
def trained_rank_centrality_model(sample_games_df):
    """Create a trained Rank Centrality model for testing."""
    from sports_arbitrage.models.rank_centrality import RankCentralityModel

    model = RankCentralityModel()
    model.fit(sample_games_df)
    return model
