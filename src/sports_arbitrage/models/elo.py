"""
ELO Rating System for NFL Game Predictions

The ELO rating system is a method for calculating the relative skill levels of players
or teams in competitive games. It updates ratings based on game outcomes.
"""

from typing import Dict, Tuple

import numpy as np
import pandas as pd


class ELOModel:
    """
    ELO rating system for predicting NFL game outcomes.

    Attributes:
        k_factor (float): The maximum rating change per game
        initial_rating (float): Starting ELO rating for new teams
        home_advantage (float): Points added to home team's rating
        ratings (Dict[str, float]): Current ELO ratings for each team
    """

    def __init__(
        self, k_factor: float = 20, initial_rating: float = 1500, home_advantage: float = 50
    ):
        """
        Initialize ELO model.

        Args:
            k_factor: Maximum rating change per game (default: 20)
            initial_rating: Starting rating for new teams (default: 1500)
            home_advantage: Points added to home team (default: 50)
        """
        self.k_factor = k_factor
        self.initial_rating = initial_rating
        self.home_advantage = home_advantage
        self.ratings: Dict[str, float] = {}
        self.rating_history: Dict[str, list] = {}

    def _get_rating(self, team: str) -> float:
        """Get current ELO rating for a team."""
        if team not in self.ratings:
            self.ratings[team] = self.initial_rating
            self.rating_history[team] = [self.initial_rating]
        return self.ratings[team]

    def _expected_score(self, rating_a: float, rating_b: float) -> float:
        """
        Calculate expected score for team A against team B.

        Args:
            rating_a: ELO rating of team A
            rating_b: ELO rating of team B

        Returns:
            Expected win probability for team A (0 to 1)
        """
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

    def predict_game(self, home_team: str, away_team: str) -> Tuple[float, float]:
        """
        Predict win probabilities for a game.

        Args:
            home_team: Name of home team
            away_team: Name of away team

        Returns:
            Tuple of (home_win_prob, away_win_prob)
        """
        home_rating = self._get_rating(home_team) + self.home_advantage
        away_rating = self._get_rating(away_team)

        home_win_prob = self._expected_score(home_rating, away_rating)
        away_win_prob = 1 - home_win_prob

        return home_win_prob, away_win_prob

    def update_ratings(self, home_team: str, away_team: str, home_won: bool):
        """
        Update ELO ratings after a game result.

        Args:
            home_team: Name of home team
            away_team: Name of away team
            home_won: True if home team won, False otherwise
        """
        home_rating = self._get_rating(home_team)
        away_rating = self._get_rating(away_team)

        # Calculate expected scores with home advantage for accurate expectations
        home_rating_with_advantage = home_rating + self.home_advantage
        home_expected = self._expected_score(home_rating_with_advantage, away_rating)
        away_expected = 1 - home_expected

        # Actual scores
        home_actual = 1.0 if home_won else 0.0
        away_actual = 0.0 if home_won else 1.0

        # Update ratings
        self.ratings[home_team] = home_rating + self.k_factor * (home_actual - home_expected)
        self.ratings[away_team] = away_rating + self.k_factor * (away_actual - away_expected)

        # Store history
        self.rating_history[home_team].append(self.ratings[home_team])
        self.rating_history[away_team].append(self.ratings[away_team])

    def fit(self, games_df: pd.DataFrame):
        """
        Train the ELO model on historical games.

        Args:
            games_df: DataFrame with columns ['home_team', 'away_team', 'home_won']
        """
        for _, game in games_df.iterrows():
            self.update_ratings(game["home_team"], game["away_team"], game["home_won"])

    def predict(self, games_df: pd.DataFrame) -> np.ndarray:
        """
        Predict outcomes for multiple games.

        Args:
            games_df: DataFrame with columns ['home_team', 'away_team']

        Returns:
            Array of home team win probabilities
        """
        predictions = []
        for _, game in games_df.iterrows():
            home_prob, _ = self.predict_game(game["home_team"], game["away_team"])
            predictions.append(home_prob)
        return np.array(predictions)

    def get_ratings_df(self) -> pd.DataFrame:
        """
        Get current ratings as a DataFrame.

        Returns:
            DataFrame with team names and their ELO ratings
        """
        return pd.DataFrame(
            [
                {"team": team, "elo_rating": rating}
                for team, rating in sorted(self.ratings.items(), key=lambda x: x[1], reverse=True)
            ]
        )

    def reset(self):
        """Reset all ratings to initial values."""
        self.ratings = {}
        self.rating_history = {}
