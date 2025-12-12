"""
Tests for prediction models (ELO, XGBoost, Random Forest, Rank Centrality).
"""

import pandas as pd

from sports_arbitrage.models.elo import ELOModel
from sports_arbitrage.models.random_forest import RandomForestModel
from sports_arbitrage.models.rank_centrality import RankCentralityModel
from sports_arbitrage.models.xgboost_model import XGBoostModel


class TestELOModel:
    """Tests for ELO rating model."""

    def test_initialization(self):
        """Test ELO model initialization."""
        model = ELOModel(initial_rating=1500, k_factor=32, home_advantage=100)
        assert model.initial_rating == 1500
        assert model.k_factor == 32
        assert model.home_advantage == 100
        assert model.ratings == {}

    def test_fit(self, sample_games_df):
        """Test ELO model training."""
        model = ELOModel()
        model.fit(sample_games_df)

        assert len(model.ratings) > 0
        assert all(isinstance(rating, float) for rating in model.ratings.values())
        assert all(rating > 0 for rating in model.ratings.values())

    def test_predict(self, trained_elo_model, sample_games_df):
        """Test ELO model predictions."""
        predictions = trained_elo_model.predict(sample_games_df.head(10))

        assert len(predictions) == 10
        assert all(0 <= p <= 1 for p in predictions), "Predictions must be probabilities [0, 1]"

    def test_predict_game(self, trained_elo_model):
        """Test single game prediction."""
        home_prob, away_prob = trained_elo_model.predict_game("Team_A", "Team_B")

        assert 0 <= home_prob <= 1
        assert 0 <= away_prob <= 1
        assert abs(home_prob + away_prob - 1.0) < 1e-6, "Probabilities should sum to 1"

    def test_reset(self, trained_elo_model):
        """Test model reset."""
        assert len(trained_elo_model.ratings) > 0

        trained_elo_model.reset()
        assert len(trained_elo_model.ratings) == 0


class TestXGBoostModel:
    """Tests for XGBoost model."""

    def test_initialization(self):
        """Test XGBoost model initialization."""
        model = XGBoostModel(n_estimators=50, max_depth=3, learning_rate=0.1)
        assert model.model.n_estimators == 50
        assert model.model.max_depth == 3
        assert model.team_stats == {}

    def test_fit(self, sample_games_df):
        """Test XGBoost model training."""
        model = XGBoostModel(n_estimators=10)
        model.fit(sample_games_df)

        assert len(model.team_stats) > 0
        assert len(model.feature_names) > 0
        assert model.model.n_features_in_ > 0

    def test_predict(self, trained_xgboost_model, sample_games_df):
        """Test XGBoost model predictions."""
        predictions = trained_xgboost_model.predict(sample_games_df.head(10))

        assert len(predictions) == 10
        assert all(0 <= p <= 1 for p in predictions), "Predictions must be probabilities [0, 1]"

    def test_predict_game(self, trained_xgboost_model):
        """Test single game prediction."""
        home_prob, away_prob = trained_xgboost_model.predict_game("Team_A", "Team_B")

        assert 0 <= home_prob <= 1
        assert 0 <= away_prob <= 1
        assert abs(home_prob + away_prob - 1.0) < 1e-6

    def test_feature_importance(self, trained_xgboost_model):
        """Test feature importance extraction."""
        importance_df = trained_xgboost_model.get_feature_importance()

        assert isinstance(importance_df, pd.DataFrame)
        assert "feature" in importance_df.columns
        assert "importance" in importance_df.columns
        assert len(importance_df) == len(trained_xgboost_model.feature_names)


class TestRandomForestModel:
    """Tests for Random Forest model."""

    def test_initialization(self):
        """Test Random Forest model initialization."""
        model = RandomForestModel(n_estimators=50, max_depth=5, min_samples_split=10)
        assert model.model.n_estimators == 50
        assert model.model.max_depth == 5
        assert model.team_stats == {}

    def test_fit(self, sample_games_df):
        """Test Random Forest model training."""
        model = RandomForestModel(n_estimators=10)
        model.fit(sample_games_df)

        assert len(model.team_stats) > 0
        assert len(model.feature_names) > 0
        assert model.model.n_features_in_ > 0

    def test_predict(self, trained_random_forest_model, sample_games_df):
        """Test Random Forest model predictions."""
        predictions = trained_random_forest_model.predict(sample_games_df.head(10))

        assert len(predictions) == 10
        assert all(0 <= p <= 1 for p in predictions), "Predictions must be probabilities [0, 1]"

    def test_predict_game(self, trained_random_forest_model):
        """Test single game prediction."""
        home_prob, away_prob = trained_random_forest_model.predict_game("Team_A", "Team_B")

        assert 0 <= home_prob <= 1
        assert 0 <= away_prob <= 1
        assert abs(home_prob + away_prob - 1.0) < 1e-6

    def test_feature_importance(self, trained_random_forest_model):
        """Test feature importance extraction."""
        importance_df = trained_random_forest_model.get_feature_importance()

        assert isinstance(importance_df, pd.DataFrame)
        assert "feature" in importance_df.columns
        assert "importance" in importance_df.columns


class TestRankCentralityModel:
    """Tests for Rank Centrality model."""

    def test_initialization(self):
        """Test Rank Centrality model initialization."""
        model = RankCentralityModel(method="pagerank", damping_factor=0.85)
        assert model.method == "pagerank"
        assert model.damping_factor == 0.85
        assert model.rankings == {}

    def test_fit(self, sample_games_df):
        """Test Rank Centrality model training."""
        model = RankCentralityModel()
        model.fit(sample_games_df)

        assert len(model.rankings) > 0
        assert all(isinstance(rank, float) for rank in model.rankings.values())
        assert all(0 <= rank <= 1 for rank in model.rankings.values())

    def test_predict(self, trained_rank_centrality_model, sample_games_df):
        """Test Rank Centrality model predictions."""
        predictions = trained_rank_centrality_model.predict(sample_games_df.head(10))

        assert len(predictions) == 10
        assert all(0 <= p <= 1 for p in predictions), "Predictions must be probabilities [0, 1]"

    def test_predict_game(self, trained_rank_centrality_model):
        """Test single game prediction."""
        home_prob, away_prob = trained_rank_centrality_model.predict_game("Team_A", "Team_B")

        assert 0 <= home_prob <= 1
        assert 0 <= away_prob <= 1
        assert abs(home_prob + away_prob - 1.0) < 1e-6

    def test_get_rankings(self, trained_rank_centrality_model):
        """Test rankings retrieval."""
        rankings_df = trained_rank_centrality_model.get_rankings_df()

        assert isinstance(rankings_df, pd.DataFrame)
        assert "team" in rankings_df.columns
        assert "rank_centrality" in rankings_df.columns
