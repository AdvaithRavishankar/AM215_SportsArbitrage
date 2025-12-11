"""
Regression tests using golden files to ensure consistent model outputs.

These tests compare current model outputs against saved 'golden' reference outputs
to catch any unintended changes in model behavior.
"""

import pytest
import numpy as np
import pandas as pd
import os
import json
from pathlib import Path


GOLDEN_DIR = Path(__file__).parent / "golden"


@pytest.fixture
def golden_predictions_path():
    """Path to golden predictions file."""
    return GOLDEN_DIR / "elo_predictions.npy"


@pytest.fixture
def golden_roi_path():
    """Path to golden ROI results file."""
    return GOLDEN_DIR / "roi_results.json"


class TestRegressionELO:
    """Regression tests for ELO model."""

    def test_elo_predictions_match_golden(self, sample_games_df, golden_predictions_path):
        """Test that ELO predictions match golden reference."""
        from sports_arbitrage.models.elo import ELOModel

        # Train model
        model = ELOModel(initial_rating=1500, k_factor=32, home_advantage=100)
        model.fit(sample_games_df.iloc[:80])

        # Make predictions on test set
        predictions = model.predict(sample_games_df.iloc[80:])

        # Load golden predictions
        if golden_predictions_path.exists():
            golden_predictions = np.load(golden_predictions_path)

            # Compare with tolerance
            np.testing.assert_allclose(
                predictions,
                golden_predictions,
                rtol=1e-5,
                atol=1e-8,
                err_msg="ELO predictions have changed from golden reference"
            )
        else:
            # Create golden file for first run
            os.makedirs(GOLDEN_DIR, exist_ok=True)
            np.save(golden_predictions_path, predictions)
            pytest.skip("Golden file created - rerun test to validate")


class TestRegressionROI:
    """Regression tests for ROI calculations."""

    def test_roi_results_match_golden(self, sample_games_df, golden_roi_path):
        """Test that ROI calculations match golden reference."""
        from sports_arbitrage.models.elo import ELOModel
        from sports_arbitrage.utils import calculate_roi

        # Train model
        model = ELOModel()
        model.fit(sample_games_df.iloc[:80])

        # Calculate ROI on test set
        test_df = sample_games_df.iloc[80:]
        predictions = model.predict(test_df)
        roi_result = calculate_roi(
            predictions=predictions,
            odds=test_df['home_avg_odds'].values,
            actual_wins=test_df['home_won'].astype(int).values,
            bet_amount=100
        )

        # Load golden ROI
        if golden_roi_path.exists():
            with open(golden_roi_path, 'r') as f:
                golden_roi = json.load(f)

            # Compare key metrics
            assert abs(roi_result['roi'] - golden_roi['roi']) < 0.01, \
                "ROI has changed from golden reference"
            assert abs(roi_result['profit'] - golden_roi['profit']) < 1.0, \
                "Profit has changed from golden reference"
            assert roi_result['num_bets'] == golden_roi['num_bets'], \
                "Number of bets has changed from golden reference"
        else:
            # Create golden file for first run
            os.makedirs(GOLDEN_DIR, exist_ok=True)
            with open(golden_roi_path, 'w') as f:
                json.dump(roi_result, f, indent=2)
            pytest.skip("Golden file created - rerun test to validate")


class TestModelConsistency:
    """Tests to ensure model outputs are consistent across runs."""

    def test_elo_model_deterministic(self, sample_games_df):
        """Test that ELO model produces same results with same seed."""
        from sports_arbitrage.models.elo import ELOModel

        # Train two models with same data
        model1 = ELOModel()
        model1.fit(sample_games_df)
        pred1 = model1.predict(sample_games_df.head(10))

        model2 = ELOModel()
        model2.fit(sample_games_df)
        pred2 = model2.predict(sample_games_df.head(10))

        # Predictions should be identical
        np.testing.assert_array_equal(pred1, pred2)

    def test_xgboost_model_deterministic(self, sample_games_df):
        """Test that XGBoost model produces same results with same seed."""
        from sports_arbitrage.models.xgboost_model import XGBoostModel

        model1 = XGBoostModel(random_state=42, n_estimators=10)
        model1.fit(sample_games_df)
        pred1 = model1.predict(sample_games_df.head(10))

        model2 = XGBoostModel(random_state=42, n_estimators=10)
        model2.fit(sample_games_df)
        pred2 = model2.predict(sample_games_df.head(10))

        # Predictions should be very close (allowing for floating point differences)
        np.testing.assert_allclose(pred1, pred2, rtol=1e-5)

    def test_random_forest_model_deterministic(self, sample_games_df):
        """Test that Random Forest model produces same results with same seed."""
        from sports_arbitrage.models.random_forest import RandomForestModel

        model1 = RandomForestModel(random_state=42, n_estimators=10)
        model1.fit(sample_games_df)
        pred1 = model1.predict(sample_games_df.head(10))

        model2 = RandomForestModel(random_state=42, n_estimators=10)
        model2.fit(sample_games_df)
        pred2 = model2.predict(sample_games_df.head(10))

        # Predictions should be identical
        np.testing.assert_array_equal(pred1, pred2)


def create_golden_files():
    """
    Utility function to create golden reference files.

    Run this after verifying current model outputs are correct:
        python -m pytest tests/test_regression.py --create-golden
    """
    pass  # Placeholder for golden file creation script
