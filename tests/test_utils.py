"""
Tests for utility functions (data processing, metrics, odds conversion).
"""

import numpy as np
import pandas as pd
import pytest

from sports_arbitrage.utils import (
    american_to_probability,
    calculate_kelly_criterion,
    calculate_metrics,
    prepare_games_data,
    probability_to_american,
)


class TestOddsConversion:
    """Tests for odds conversion functions."""

    def test_american_to_probability_negative(self):
        """Test conversion of negative American odds to probability."""
        # -200 implies 200/(200+100) = 0.667
        prob = american_to_probability(-200)
        assert abs(prob - 0.6667) < 0.01

    def test_american_to_probability_positive(self):
        """Test conversion of positive American odds to probability."""
        # +200 implies 100/(200+100) = 0.333
        prob = american_to_probability(200)
        assert abs(prob - 0.3333) < 0.01

    def test_american_to_probability_even(self):
        """Test conversion of even odds."""
        # +100 implies 100/(100+100) = 0.5
        prob = american_to_probability(100)
        assert abs(prob - 0.5) < 0.01

    def test_american_to_probability_invalid_zero(self):
        """Test that zero odds raise ValueError."""
        with pytest.raises(ValueError, match="cannot be zero"):
            american_to_probability(0)

    def test_probability_to_american_favorite(self):
        """Test conversion of probability to negative American odds (favorite)."""
        # 0.667 probability should give approximately -200
        odds = probability_to_american(0.6667)
        assert odds < 0
        assert abs(odds - (-200)) < 10

    def test_probability_to_american_underdog(self):
        """Test conversion of probability to positive American odds (underdog)."""
        # 0.333 probability should give approximately +200
        odds = probability_to_american(0.3333)
        assert odds > 0
        assert abs(odds - 200) < 10

    def test_probability_to_american_invalid_range(self):
        """Test that probabilities outside (0, 1) raise ValueError."""
        with pytest.raises(ValueError, match="Must be between 0 and 1"):
            probability_to_american(1.5)

        with pytest.raises(ValueError, match="Must be between 0 and 1"):
            probability_to_american(0.0)

    def test_round_trip_conversion(self):
        """Test that odds -> probability -> odds is consistent."""
        test_odds = [-200, -150, -110, 100, 150, 200]

        for odds in test_odds:
            prob = american_to_probability(odds)
            back_to_odds = probability_to_american(prob)
            back_prob = american_to_probability(back_to_odds)
            assert abs(prob - back_prob) < 1e-6, f"Round trip probability failed for odds {odds}"


class TestKellyCriterion:
    """Tests for Kelly Criterion calculation."""

    def test_kelly_criterion_positive_edge(self):
        """Test Kelly with positive edge (good bet)."""
        # Win prob: 0.6, Odds: +100 (implied prob 0.5) -> positive edge
        kelly_fraction = calculate_kelly_criterion(0.6, 100, fraction=1.0)
        assert kelly_fraction > 0, "Should recommend bet with positive edge"

    def test_kelly_criterion_negative_edge(self):
        """Test Kelly with negative edge (bad bet)."""
        # Win prob: 0.4, Odds: +100 (implied prob 0.5) -> negative edge
        kelly_fraction = calculate_kelly_criterion(0.4, 100, fraction=1.0)
        assert kelly_fraction == 0, "Should not recommend bet with negative edge"

    def test_kelly_criterion_fractional(self):
        """Test fractional Kelly (quarter Kelly)."""
        kelly_full = calculate_kelly_criterion(0.6, 100, fraction=1.0)
        kelly_quarter = calculate_kelly_criterion(0.6, 100, fraction=0.25)

        assert kelly_quarter == kelly_full * 0.25

    def test_kelly_criterion_invalid_probability(self):
        """Test that invalid probabilities raise ValueError."""
        with pytest.raises(ValueError, match="Invalid win probability"):
            calculate_kelly_criterion(1.5, 100)

        with pytest.raises(ValueError, match="Invalid win probability"):
            calculate_kelly_criterion(0.0, 100)

    def test_kelly_criterion_invalid_odds(self):
        """Test that invalid odds raise ValueError."""
        with pytest.raises(ValueError, match="Invalid American odds"):
            calculate_kelly_criterion(0.6, 0)

    def test_kelly_criterion_invalid_fraction(self):
        """Test that invalid fraction raises ValueError."""
        with pytest.raises(ValueError, match="Invalid Kelly fraction"):
            calculate_kelly_criterion(0.6, 100, fraction=1.5)

        with pytest.raises(ValueError, match="Invalid Kelly fraction"):
            calculate_kelly_criterion(0.6, 100, fraction=0.0)


class TestMetricsCalculation:
    """Tests for performance metrics calculation."""

    def test_calculate_metrics_perfect_predictions(self):
        """Test metrics with perfect predictions."""
        y_true = np.array([1, 1, 0, 0, 1, 0])
        y_pred = np.array([1.0, 1.0, 0.0, 0.0, 1.0, 0.0])

        metrics = calculate_metrics(y_true, y_pred)

        assert metrics["accuracy"] == 1.0
        assert metrics["roc_auc"] == 1.0
        assert metrics["log_loss"] < 0.1  # Very small log loss
        assert metrics["brier_score"] < 0.1  # Very small Brier score

    def test_calculate_metrics_random_predictions(self):
        """Test metrics with random predictions."""
        np.random.seed(42)
        y_true = np.random.choice([0, 1], 100)
        y_pred = np.random.uniform(0, 1, 100)

        metrics = calculate_metrics(y_true, y_pred)

        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["roc_auc"] <= 1
        assert metrics["log_loss"] >= 0
        assert 0 <= metrics["brier_score"] <= 1

    def test_calculate_metrics_consistent_predictions(self):
        """Test that better predictions yield better metrics."""
        y_true = np.array([1, 1, 0, 0, 1, 0, 1, 0])

        # Good predictions
        good_pred = np.array([0.9, 0.85, 0.1, 0.15, 0.95, 0.05, 0.9, 0.2])
        # Bad predictions (opposite)
        bad_pred = 1 - good_pred

        metrics_good = calculate_metrics(y_true, good_pred)
        metrics_bad = calculate_metrics(y_true, bad_pred)

        assert metrics_good["accuracy"] > metrics_bad["accuracy"]
        assert metrics_good["roc_auc"] > metrics_bad["roc_auc"]
        assert metrics_good["log_loss"] < metrics_bad["log_loss"]
        assert metrics_good["brier_score"] < metrics_bad["brier_score"]


class TestDataPreparation:
    """Tests for data preparation functions."""

    def test_prepare_games_data(self, sample_odds_df):
        """Test game data preparation from odds."""
        games_df = prepare_games_data(sample_odds_df)

        assert isinstance(games_df, pd.DataFrame)
        assert "game_id" in games_df.columns
        assert "home_team" in games_df.columns
        assert "away_team" in games_df.columns
        assert "home_avg_odds" in games_df.columns
        assert "away_avg_odds" in games_df.columns

        # Check that odds are averaged across sportsbooks
        assert len(games_df) > 0
        assert games_df["home_avg_odds"].notna().any()

    def test_prepare_games_data_unique_games(self, sample_odds_df):
        """Test that games are deduplicated properly."""
        games_df = prepare_games_data(sample_odds_df)

        # Each game_id should appear only once
        assert len(games_df) == games_df["game_id"].nunique()
