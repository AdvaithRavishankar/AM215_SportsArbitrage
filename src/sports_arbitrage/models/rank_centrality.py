"""
Rank Centrality Model for NFL Game Predictions

Rank Centrality uses network-based ranking to determine team strength based on
the outcomes of games played. It constructs a graph of teams and uses eigenvector
centrality or similar metrics to rank teams.
"""

import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, Tuple


class RankCentralityModel:
    """
    Rank Centrality model for predicting NFL game outcomes.

    Uses graph-based ranking where teams are nodes and game results create
    weighted edges. Strength is determined by centrality measures.

    Attributes:
        method (str): Centrality method ('pagerank', 'eigenvector', 'katz')
        damping_factor (float): Damping factor for PageRank
        home_advantage (float): Advantage multiplier for home team
        rankings (Dict[str, float]): Current team rankings
    """

    def __init__(self, method: str = 'pagerank', damping_factor: float = 0.85,
                 home_advantage: float = 0.05):
        """
        Initialize Rank Centrality model.

        Args:
            method: Centrality method ('pagerank', 'eigenvector', 'katz')
            damping_factor: Damping factor for PageRank (default: 0.85)
            home_advantage: Home advantage boost (default: 0.05)
        """
        self.method = method
        self.damping_factor = damping_factor
        self.home_advantage = home_advantage
        self.rankings: Dict[str, float] = {}
        self.graph = nx.DiGraph()

    def _build_graph(self, games_df: pd.DataFrame):
        """
        Build directed graph from game results.

        Args:
            games_df: DataFrame with columns ['home_team', 'away_team', 'home_won']
        """
        self.graph.clear()

        for _, game in games_df.iterrows():
            home_team = game['home_team']
            away_team = game['away_team']
            home_won = game['home_won']

            # Add nodes
            if home_team not in self.graph:
                self.graph.add_node(home_team)
            if away_team not in self.graph:
                self.graph.add_node(away_team)

            # Add edges: loser -> winner (standard PageRank: winners receive in-edges)
            if home_won:
                if self.graph.has_edge(away_team, home_team):
                    self.graph[away_team][home_team]['weight'] += 1
                else:
                    self.graph.add_edge(away_team, home_team, weight=1)
            else:
                if self.graph.has_edge(home_team, away_team):
                    self.graph[home_team][away_team]['weight'] += 1
                else:
                    self.graph.add_edge(home_team, away_team, weight=1)

    def _calculate_centrality(self) -> Dict[str, float]:
        """
        Calculate centrality scores for all teams.

        Returns:
            Dictionary mapping team names to centrality scores
        """
        if len(self.graph.nodes()) == 0:
            return {}

        try:
            if self.method == 'pagerank':
                centrality = nx.pagerank(
                    self.graph,
                    alpha=self.damping_factor,
                    weight='weight'
                )
            elif self.method == 'eigenvector':
                centrality = nx.eigenvector_centrality(
                    self.graph,
                    weight='weight',
                    max_iter=1000
                )
            elif self.method == 'katz':
                centrality = nx.katz_centrality(
                    self.graph,
                    alpha=0.1,
                    weight='weight',
                    max_iter=1000
                )
            else:
                raise ValueError(f"Unknown method: {self.method}")

        except (nx.PowerIterationFailedConvergence, nx.NetworkXError):
            # Fallback: use win rate (in-degree = wins)
            centrality = {}
            for node in self.graph.nodes():
                in_degree = self.graph.in_degree(node, weight='weight')
                out_degree = self.graph.out_degree(node, weight='weight')
                total = in_degree + out_degree
                centrality[node] = in_degree / total if total > 0 else 0.5

        return centrality

    def fit(self, games_df: pd.DataFrame):
        """
        Train the Rank Centrality model on historical games.

        Args:
            games_df: DataFrame with columns ['home_team', 'away_team', 'home_won']
        """
        self._build_graph(games_df)
        self.rankings = self._calculate_centrality()

    def predict_game(self, home_team: str, away_team: str) -> Tuple[float, float]:
        """
        Predict win probabilities for a game.

        Args:
            home_team: Name of home team
            away_team: Name of away team

        Returns:
            Tuple of (home_win_prob, away_win_prob)
        """
        # Get rankings (default to 0.5 for unseen teams)
        home_rank = self.rankings.get(home_team, 0.5)
        away_rank = self.rankings.get(away_team, 0.5)

        # Apply home advantage
        home_rank_adj = home_rank * (1 + self.home_advantage)

        # Normalize to probabilities
        total = home_rank_adj + away_rank
        if total > 0:
            home_win_prob = home_rank_adj / total
            away_win_prob = away_rank / total
        else:
            home_win_prob = 0.5
            away_win_prob = 0.5

        return home_win_prob, away_win_prob

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
            home_prob, _ = self.predict_game(game['home_team'], game['away_team'])
            predictions.append(home_prob)
        return np.array(predictions)

    def get_rankings_df(self) -> pd.DataFrame:
        """
        Get current rankings as a DataFrame.

        Returns:
            DataFrame with team names and their centrality rankings
        """
        return pd.DataFrame([
            {'team': team, 'rank_centrality': rank}
            for team, rank in sorted(self.rankings.items(),
                                    key=lambda x: x[1], reverse=True)
        ])

    def reset(self):
        """Reset the model."""
        self.rankings = {}
        self.graph.clear()
