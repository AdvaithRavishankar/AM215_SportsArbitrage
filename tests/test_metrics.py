"""
Tests for ROI calculations and betting strategies.
"""

import pytest
import numpy as np
import pandas as pd
from sports_arbitrage.utils import (
    calculate_roi,
    calculate_kelly_roi,
    optimize_markowitz_portfolio,
)


class TestROICalculation:
    """Tests for fixed betting ROI calculation."""

    def test_calculate_roi_all_wins(self):
        """Test ROI when all predictions win."""
        # Create simple test data
        games_df = pd.DataFrame({
            'game_id': ['g1', 'g2', 'g3'],
            'home_team': ['A', 'B', 'C'],
            'away_team': ['X', 'Y', 'Z'],
            'home_won': [True, True, True],
            'home_avg_odds': [100, 100, 100],  # Even odds
        })

        predictions = np.array([0.6, 0.7, 0.8])  # All > 0.5, so bet on all

        roi_result = calculate_roi(
            predictions=predictions,
            odds=games_df['home_avg_odds'].values,
            actual_wins=games_df['home_won'].astype(int).values,
            bet_amount=100
        )

        assert 'roi' in roi_result
        assert 'profit' in roi_result
        assert 'total_bet' in roi_result
        assert 'num_bets' in roi_result

        # All wins at even odds should yield positive ROI
        assert roi_result['roi'] > 0
        assert roi_result['profit'] > 0
        assert roi_result['num_bets'] == 3

    def test_calculate_roi_all_losses(self):
        """Test ROI when all predictions lose."""
        games_df = pd.DataFrame({
            'game_id': ['g1', 'g2', 'g3'],
            'home_team': ['A', 'B', 'C'],
            'away_team': ['X', 'Y', 'Z'],
            'home_won': [False, False, False],
            'home_avg_odds': [100, 100, 100],
        })

        predictions = np.array([0.6, 0.7, 0.8])  # Bet on all

        roi_result = calculate_roi(
            predictions=predictions,
            odds=games_df['home_avg_odds'].values,
            actual_wins=games_df['home_won'].astype(int).values,
            bet_amount=100
        )

        # All losses should yield negative ROI
        assert roi_result['roi'] < 0
        assert roi_result['profit'] < 0
        assert roi_result['num_bets'] == 3

    def test_calculate_roi_no_bets(self):
        """Test ROI when no bets are placed."""
        games_df = pd.DataFrame({
            'game_id': ['g1', 'g2'],
            'home_team': ['A', 'B'],
            'away_team': ['X', 'Y'],
            'home_won': [True, False],
            'home_avg_odds': [100, 100],
        })

        predictions = np.array([0.3, 0.4])  # All < 0.5, no bets

        roi_result = calculate_roi(
            predictions=predictions,
            odds=games_df['home_avg_odds'].values,
            actual_wins=games_df['home_won'].astype(int).values,
        )

        assert roi_result['num_bets'] == 0
        assert roi_result['total_bet'] == 0
        assert roi_result['profit'] == 0

    def test_calculate_roi_mixed_results(self):
        """Test ROI with mixed wins and losses."""
        games_df = pd.DataFrame({
            'game_id': ['g1', 'g2', 'g3', 'g4'],
            'home_team': ['A', 'B', 'C', 'D'],
            'away_team': ['W', 'X', 'Y', 'Z'],
            'home_won': [True, False, True, False],
            'home_avg_odds': [100, 100, 100, 100],
        })

        predictions = np.array([0.6, 0.6, 0.6, 0.6])  # Bet on all

        roi_result = calculate_roi(
            predictions=predictions,
            odds=games_df['home_avg_odds'].values,
            actual_wins=games_df['home_won'].astype(int).values,
            bet_amount=100
        )

        assert roi_result['num_bets'] == 4
        assert roi_result['total_bet'] == 400
        # 2 wins, 2 losses at even odds = break even
        assert abs(roi_result['profit']) < 50  # Should be close to 0


class TestKellyROI:
    """Tests for Kelly Criterion betting strategy."""

    def test_kelly_roi_positive_edge(self):
        """Test Kelly strategy with positive edge."""
        games_df = pd.DataFrame({
            'game_id': ['g1', 'g2'],
            'home_team': ['A', 'B'],
            'away_team': ['X', 'Y'],
            'home_won': [True, True],
            'home_avg_odds': [100, 100],  # Implied prob 0.5
            'commence_time': pd.to_datetime(['2023-01-01', '2023-01-02']),
        })

        predictions = np.array([0.65, 0.70])  # Edge over implied prob

        kelly_result = calculate_kelly_roi(
            predictions=predictions,
            odds=games_df['home_avg_odds'].values,
            actual_wins=games_df['home_won'].astype(int).values,
            bankroll=10000,
            fraction=0.25
        )

        assert 'roi' in kelly_result
        assert 'profit' in kelly_result
        assert 'final_bankroll' in kelly_result
        assert kelly_result['num_bets'] == 2

    def test_kelly_roi_bankroll_growth(self):
        """Test that bankroll grows with Kelly strategy."""
        games_df = pd.DataFrame({
            'game_id': ['g1'] * 5,
            'home_team': ['A'] * 5,
            'away_team': ['X'] * 5,
            'home_won': [True] * 5,  # All wins
            'home_avg_odds': [100] * 5,
            'commence_time': pd.to_datetime(['2023-01-01'] * 5),
        })

        predictions = np.array([0.7] * 5)  # Good predictions

        initial = 10000
        kelly_result = calculate_kelly_roi(
            predictions=predictions,
            odds=games_df['home_avg_odds'].values,
            actual_wins=games_df['home_won'].astype(int).values,
            bankroll=initial,
            fraction=0.25
        )

        # With positive edge and wins, final bankroll should grow
        assert kelly_result['final_bankroll'] > initial

    def test_kelly_roi_no_bets_negative_edge(self):
        """Test Kelly doesn't bet with negative edge."""
        games_df = pd.DataFrame({
            'game_id': ['g1', 'g2'],
            'home_team': ['A', 'B'],
            'away_team': ['X', 'Y'],
            'home_won': [True, True],
            'home_avg_odds': [100, 100],  # Implied prob 0.5
            'commence_time': pd.to_datetime(['2023-01-01', '2023-01-02']),
        })

        predictions = np.array([0.3, 0.4])  # Negative edge

        kelly_result = calculate_kelly_roi(
            predictions=predictions,
            odds=games_df['home_avg_odds'].values,
            actual_wins=games_df['home_won'].astype(int).values,
            bankroll=10000,
            fraction=0.25
        )

        # Should make minimal or no bets
        assert kelly_result['num_bets'] <= 2


class TestMarkowitzOptimization:
    """Tests for Markowitz portfolio optimization."""

    def test_optimize_markowitz_portfolio_basic(self):
        """Test basic Markowitz optimization."""
        np.random.seed(42)

        # Create simple expected returns and covariance
        n = 5
        expected_returns = np.random.uniform(0.01, 0.10, n)
        covariance_matrix = np.eye(n) * 0.01  # Diagonal covariance

        weights = optimize_markowitz_portfolio(
            expected_returns,
            covariance_matrix,
            risk_aversion=2.0,
            max_position=0.3
        )

        assert len(weights) == n
        assert all(w >= 0 for w in weights), "Weights must be non-negative"
        assert all(w <= 0.3 for w in weights), "Weights must respect max position"
        assert abs(np.sum(weights) - 1.0) < 1e-4, "Weights must sum to 1"

    def test_optimize_markowitz_portfolio_high_return_asset(self):
        """Test that Markowitz allocates more to higher return assets."""
        n = 3
        expected_returns = np.array([0.01, 0.10, 0.02])  # Second asset has high return
        covariance_matrix = np.eye(n) * 0.01

        weights = optimize_markowitz_portfolio(
            expected_returns,
            covariance_matrix,
            risk_aversion=1.0,
            max_position=1.0
        )

        # Asset 1 (index 1) should get significant allocation
        assert weights[1] > weights[0]
        assert weights[1] > weights[2]

    def test_optimize_markowitz_portfolio_empty_input(self):
        """Test handling of empty inputs."""
        with pytest.raises(ValueError, match="cannot be empty"):
            optimize_markowitz_portfolio(
                np.array([]),
                np.array([[]]),
                risk_aversion=2.0
            )

    def test_optimize_markowitz_portfolio_mismatched_dimensions(self):
        """Test handling of mismatched dimensions."""
        expected_returns = np.array([0.01, 0.02, 0.03])
        bad_covariance = np.eye(2)  # Wrong size

        with pytest.raises(ValueError, match="must match"):
            optimize_markowitz_portfolio(
                expected_returns,
                bad_covariance,
                risk_aversion=2.0
            )

    def test_optimize_markowitz_portfolio_invalid_risk_aversion(self):
        """Test handling of invalid risk aversion."""
        expected_returns = np.array([0.01, 0.02])
        covariance_matrix = np.eye(2) * 0.01

        with pytest.raises(ValueError, match="risk_aversion must be positive"):
            optimize_markowitz_portfolio(
                expected_returns,
                covariance_matrix,
                risk_aversion=-1.0
            )
