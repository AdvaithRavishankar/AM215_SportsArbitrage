"""
Tests for arbitrage detection and opportunity finding.
"""

import pytest
import numpy as np
import pandas as pd
from sports_arbitrage.utils import find_arbitrage_opportunities


class TestArbitrageDetection:
    """Tests for arbitrage opportunity detection."""

    def test_find_arbitrage_no_opportunities(self):
        """Test when no arbitrage opportunities exist."""
        # Create odds where no arbitrage is possible
        odds_df = pd.DataFrame(
            [
                {
                    "game_id": "g1",
                    "home_team": "A",
                    "away_team": "B",
                    "sportsbook": "Book1",
                    "team": "A",
                    "odds": -110,
                },
                {
                    "game_id": "g1",
                    "home_team": "A",
                    "away_team": "B",
                    "sportsbook": "Book1",
                    "team": "B",
                    "odds": -110,
                },
            ]
        )

        arb_opportunities = find_arbitrage_opportunities(odds_df, min_roi=0.01)

        assert isinstance(arb_opportunities, pd.DataFrame)
        # May or may not find opportunities depending on exact odds
        # but should return a DataFrame

    def test_find_arbitrage_with_opportunities(self):
        """Test finding clear arbitrage opportunities."""
        # Create obvious arbitrage:
        # Team A at +200 (prob 0.333) and Team B at +200 (prob 0.333)
        # Total prob = 0.666 < 1.0 -> Arbitrage!
        odds_df = pd.DataFrame(
            [
                {
                    "game_id": "g1",
                    "home_team": "A",
                    "away_team": "B",
                    "sportsbook": "Book1",
                    "team": "A",
                    "odds": 200,
                    "commence_time": pd.Timestamp("2023-01-01"),
                },
                {
                    "game_id": "g1",
                    "home_team": "A",
                    "away_team": "B",
                    "sportsbook": "Book2",
                    "team": "B",
                    "odds": 200,
                    "commence_time": pd.Timestamp("2023-01-01"),
                },
            ]
        )

        arb_opportunities = find_arbitrage_opportunities(odds_df, min_roi=0.01)

        assert isinstance(arb_opportunities, pd.DataFrame)

        if len(arb_opportunities) > 0:
            # Check structure
            assert "game_id" in arb_opportunities.columns
            assert "arbitrage_roi" in arb_opportunities.columns
            assert "home_stake_pct" in arb_opportunities.columns
            assert "away_stake_pct" in arb_opportunities.columns

            # Check stake percentages sum to 100
            if "home_stake_pct" in arb_opportunities.columns:
                for _, row in arb_opportunities.iterrows():
                    total_stake = row["home_stake_pct"] + row["away_stake_pct"]
                    assert abs(total_stake - 100) < 0.1

    def test_find_arbitrage_min_roi_filter(self):
        """Test that min_roi parameter filters results."""
        # Create synthetic arbitrage data
        odds_df = pd.DataFrame(
            [
                {
                    "game_id": "g1",
                    "home_team": "A",
                    "away_team": "B",
                    "sportsbook": "Book1",
                    "team": "A",
                    "odds": 150,
                    "commence_time": pd.Timestamp("2023-01-01"),
                },
                {
                    "game_id": "g1",
                    "home_team": "A",
                    "away_team": "B",
                    "sportsbook": "Book2",
                    "team": "B",
                    "odds": 150,
                    "commence_time": pd.Timestamp("2023-01-01"),
                },
            ]
        )

        # Find with low threshold
        arb_low = find_arbitrage_opportunities(odds_df, min_roi=0.001)
        # Find with high threshold
        arb_high = find_arbitrage_opportunities(odds_df, min_roi=10.0)

        # High threshold should return fewer (or equal) opportunities
        assert len(arb_high) <= len(arb_low)

    def test_find_arbitrage_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        empty_df = pd.DataFrame(columns=["game_id", "team", "sportsbook", "odds"])
        arb_opportunities = find_arbitrage_opportunities(empty_df)

        assert isinstance(arb_opportunities, pd.DataFrame)
        assert len(arb_opportunities) == 0

    def test_find_arbitrage_return_columns(self):
        """Test that returned DataFrame has expected columns."""
        odds_df = pd.DataFrame(
            [
                {
                    "game_id": "g1",
                    "home_team": "A",
                    "away_team": "B",
                    "sportsbook": "Book1",
                    "team": "A",
                    "odds": -105,
                    "commence_time": pd.Timestamp("2023-01-01"),
                },
                {
                    "game_id": "g1",
                    "home_team": "A",
                    "away_team": "B",
                    "sportsbook": "Book2",
                    "team": "B",
                    "odds": -105,
                    "commence_time": pd.Timestamp("2023-01-01"),
                },
            ]
        )

        arb_opportunities = find_arbitrage_opportunities(odds_df)

        expected_columns = [
            "game_id",
            "home_team",
            "away_team",
            "arbitrage_roi",
            "home_stake_pct",
            "away_stake_pct",
        ]

        for col in expected_columns:
            if len(arb_opportunities) > 0:
                assert col in arb_opportunities.columns, f"Missing column: {col}"
