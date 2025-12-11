"""
Random Forest Model for NFL Game Predictions

Random Forest is an ensemble learning method that operates by constructing
multiple decision trees and outputting the mode of their predictions.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from typing import Dict, Tuple


class RandomForestModel:
    """
    Random Forest model for predicting NFL game outcomes.

    Uses ensemble of decision trees with engineered features including:
    - Team win rates
    - Recent form
    - Head-to-head history
    - Home/away performance

    Attributes:
        model (RandomForestClassifier): The Random Forest classifier
        feature_names (list): Names of features used
        team_stats (Dict): Historical statistics for each team
    """

    def __init__(self, n_estimators: int = 100, max_depth: int = 10,
                 min_samples_split: int = 5, random_state: int = 42):
        """
        Initialize Random Forest model.

        Args:
            n_estimators: Number of trees in the forest
            max_depth: Maximum tree depth
            min_samples_split: Minimum samples required to split
            random_state: Random seed for reproducibility
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state,
            class_weight='balanced'
        )
        self.feature_names = []
        self.team_stats: Dict = {}

    def _calculate_team_stats(self, games_df: pd.DataFrame) -> Dict:
        """
        Calculate team statistics from historical games.

        Args:
            games_df: DataFrame with game results

        Returns:
            Dictionary of team statistics
        """
        stats = {}

        for team in pd.concat([games_df['home_team'], games_df['away_team']]).unique():
            # Home games
            home_games = games_df[games_df['home_team'] == team]
            home_wins = home_games['home_won'].sum() if len(home_games) > 0 else 0
            home_total = len(home_games)

            # Away games
            away_games = games_df[games_df['away_team'] == team]
            away_wins = (~away_games['home_won']).sum() if len(away_games) > 0 else 0
            away_total = len(away_games)

            # Overall stats
            total_games = home_total + away_total
            total_wins = home_wins + away_wins

            # Recent form (last 5 games)
            team_games = pd.concat([
                home_games.assign(won=home_games['home_won'], is_home=True),
                away_games.assign(won=~away_games['home_won'], is_home=False)
            ]).sort_values('commence_time')

            recent_games = team_games.tail(5)
            recent_wins = recent_games['won'].sum() if len(recent_games) > 0 else 0

            stats[team] = {
                'win_rate': total_wins / total_games if total_games > 0 else 0.5,
                'home_win_rate': home_wins / home_total if home_total > 0 else 0.5,
                'away_win_rate': away_wins / away_total if away_total > 0 else 0.5,
                'recent_form': recent_wins / len(recent_games) if len(recent_games) > 0 else 0.5,
                'total_games': total_games,
            }

        return stats

    def _extract_features(self, row: pd.Series, stats: Dict) -> np.ndarray:
        """
        Extract features for a single game.

        Args:
            row: Game data row
            stats: Team statistics dictionary

        Returns:
            Feature vector as numpy array
        """
        home_team = row['home_team']
        away_team = row['away_team']

        home_stats = stats.get(home_team, {
            'win_rate': 0.5, 'home_win_rate': 0.5,
            'away_win_rate': 0.5, 'recent_form': 0.5, 'total_games': 0
        })
        away_stats = stats.get(away_team, {
            'win_rate': 0.5, 'home_win_rate': 0.5,
            'away_win_rate': 0.5, 'recent_form': 0.5, 'total_games': 0
        })

        # Extract temporal features from game date
        game_time = pd.to_datetime(row['commence_time'])

        features = [
            home_stats['win_rate'],
            away_stats['win_rate'],
            home_stats['home_win_rate'],
            away_stats['away_win_rate'],
            home_stats['recent_form'],
            away_stats['recent_form'],
            home_stats['total_games'],
            away_stats['total_games'],
            game_time.year,           # Year (captures roster changes)
            game_time.month,          # Month (early vs late season)
            game_time.dayofweek,      # Day of week (rest patterns, 0=Monday, 6=Sunday)
        ]

        return np.array(features)

    def fit(self, games_df: pd.DataFrame):
        """
        Train the Random Forest model on historical games.

        Args:
            games_df: DataFrame with columns ['home_team', 'away_team', 'home_won']
        """
        # Calculate statistics on training data
        self.team_stats = self._calculate_team_stats(games_df)

        # Extract features
        X = np.array([
            self._extract_features(row, self.team_stats)
            for _, row in games_df.iterrows()
        ])

        y = games_df['home_won'].values.astype(int)

        # Feature names for interpretability
        self.feature_names = [
            'home_win_rate', 'away_win_rate',
            'home_home_win_rate', 'away_away_win_rate',
            'home_recent_form', 'away_recent_form',
            'home_total_games', 'away_total_games',
            'year', 'month', 'day_of_week'
        ]

        # Train model
        self.model.fit(X, y)

    def predict(self, games_df: pd.DataFrame) -> np.ndarray:
        """
        Predict outcomes for multiple games.

        Args:
            games_df: DataFrame with columns ['home_team', 'away_team']

        Returns:
            Array of home team win probabilities
        """
        X = np.array([
            self._extract_features(row, self.team_stats)
            for _, row in games_df.iterrows()
        ])

        # Predict probabilities
        probabilities = self.model.predict_proba(X)[:, 1]
        return probabilities

    def predict_game(self, home_team: str, away_team: str) -> Tuple[float, float]:
        """
        Predict win probabilities for a single game.

        Args:
            home_team: Name of home team
            away_team: Name of away team

        Returns:
            Tuple of (home_win_prob, away_win_prob)
        """
        game_df = pd.DataFrame([{
            'home_team': home_team,
            'away_team': away_team
        }])

        home_prob = self.predict(game_df)[0]
        return home_prob, 1 - home_prob

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance scores.

        Returns:
            DataFrame with feature names and importance scores
        """
        importance = self.model.feature_importances_
        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)

    def reset(self):
        """Reset the model."""
        self.team_stats = {}
        self.model = RandomForestClassifier(
            n_estimators=self.model.n_estimators,
            max_depth=self.model.max_depth,
            min_samples_split=self.model.min_samples_split,
            random_state=self.model.random_state,
            class_weight='balanced'
        )
