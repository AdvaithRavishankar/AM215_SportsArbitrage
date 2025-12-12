"""
Utility functions for sports betting arbitrage system.

Includes data loading, preprocessing, evaluation metrics, and arbitrage calculations.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score


def load_odds_data(filepath: str) -> pd.DataFrame:
    """
    Load odds data from CSV file.

    Args:
        filepath: Path to the odds CSV file

    Returns:
        DataFrame with odds data
    """
    df = pd.read_csv(filepath)

    # Parse dates
    if "commence_time" in df.columns:
        df["commence_time"] = pd.to_datetime(df["commence_time"])
    if "snapshot_date" in df.columns:
        df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])

    return df


def american_to_probability(odds: float) -> float:
    """
    Convert American odds to implied probability.

    Args:
        odds: American odds (e.g., -110, +150)

    Returns:
        Implied probability (0 to 1)

    Raises:
        ValueError: If odds are invalid (e.g., between -100 and 100 exclusive, or 0)
    """
    if odds == 0:
        raise ValueError(f"Invalid American odds: {odds}. Odds cannot be zero")

    if odds < 0:
        return abs(odds) / (abs(odds) + 100)
    else:
        return 100 / (odds + 100)


def probability_to_american(prob: float) -> float:
    """
    Convert probability to American odds.

    Args:
        prob: Probability (0 to 1)

    Returns:
        American odds

    Raises:
        ValueError: If probability is not in valid range (0, 1)
    """
    if not (0 < prob < 1):
        raise ValueError(f"Invalid probability: {prob}. Must be between 0 and 1 (exclusive)")

    if prob >= 0.5:
        return -100 * prob / (1 - prob)
    else:
        return 100 * (1 - prob) / prob


def prepare_games_data(odds_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare game-level data from odds data.

    Args:
        odds_df: DataFrame with odds data

    Returns:
        DataFrame with one row per game
    """
    # Get unique games
    games = (
        odds_df.groupby(["game_id", "home_team", "away_team", "commence_time"])
        .first()
        .reset_index()
    )

    # Get average odds per team per game
    avg_odds = odds_df.groupby(["game_id", "team"])["odds"].mean().reset_index()

    # Merge home team odds
    home_odds = avg_odds.rename(columns={"team": "home_team", "odds": "home_avg_odds"})
    games_with_odds = games.merge(
        home_odds[["game_id", "home_team", "home_avg_odds"]],
        on=["game_id", "home_team"],
        how="left",
    )

    # Merge away team odds
    away_odds = avg_odds.rename(columns={"team": "away_team", "odds": "away_avg_odds"})
    games_with_odds = games_with_odds.merge(
        away_odds[["game_id", "away_team", "away_avg_odds"]],
        on=["game_id", "away_team"],
        how="left",
    )

    return games_with_odds


def add_game_results(
    games_df: pd.DataFrame,
    results_df: Optional[pd.DataFrame] = None,
    rng: Optional[np.random.Generator] = None,
) -> pd.DataFrame:
    """
    Add game results to games DataFrame.

    If results_df is provided, use it to add actual outcomes.
    Otherwise, simulate results based on odds.

    Args:
        games_df: DataFrame with game data
        results_df: Optional DataFrame with actual results

    Returns:
        DataFrame with 'home_won' column added
    """
    df = games_df.copy()
    rng = rng or np.random.default_rng()

    if results_df is not None:
        # Merge with actual results
        df = df.merge(results_df[["game_id", "home_won"]], on="game_id", how="left")
    else:
        # Simulate results based on odds (for demonstration)
        if "home_avg_odds" in df.columns and "away_avg_odds" in df.columns:
            df["home_prob"] = df["home_avg_odds"].apply(
                lambda x: american_to_probability(x) if pd.notna(x) else 0.5
            )
            df["home_won"] = rng.random(len(df)) < df["home_prob"]
        else:
            # Random if no odds available
            df["home_won"] = rng.random(len(df)) < 0.5

    return df


def create_rolling_windows(
    df: pd.DataFrame, n_folds: int = 3
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Create rolling window train/test splits for time series cross-validation.

    Args:
        df: DataFrame sorted by time
        n_folds: Number of folds

    Returns:
        List of (train_df, test_df) tuples
    """
    df = df.sort_values("commence_time").reset_index(drop=True)
    n = len(df)
    fold_size = n // (n_folds + 1)

    folds = []
    for i in range(n_folds):
        train_end = (i + 1) * fold_size
        test_start = train_end
        test_end = test_start + fold_size

        if test_end > n:
            test_end = n

        train_df = df.iloc[:train_end].copy()
        test_df = df.iloc[test_start:test_end].copy()

        if len(test_df) > 0:
            folds.append((train_df, test_df))

    return folds


def calculate_metrics(
    y_true: np.ndarray, y_pred_proba: np.ndarray, threshold: float = 0.5
) -> Dict[str, float]:
    """
    Calculate evaluation metrics for predictions.

    Args:
        y_true: True labels (0 or 1)
        y_pred_proba: Predicted probabilities
        threshold: Decision threshold for classification

    Returns:
        Dictionary of metric names and values
    """
    y_pred = (y_pred_proba >= threshold).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "log_loss": log_loss(y_true, y_pred_proba),
        "brier_score": brier_score_loss(y_true, y_pred_proba),
    }

    try:
        metrics["roc_auc"] = roc_auc_score(y_true, y_pred_proba)
    except ValueError:
        metrics["roc_auc"] = np.nan

    return metrics


def calculate_roi(
    predictions: np.ndarray, odds: np.ndarray, actual_wins: np.ndarray, bet_amount: float = 100
) -> Dict[str, float]:
    """
    Calculate return on investment for betting strategy.

    Args:
        predictions: Predicted win probabilities
        odds: American odds for bets
        actual_wins: Actual outcomes (1 if won, 0 if lost)
        bet_amount: Amount bet per game

    Returns:
        Dictionary with ROI metrics
    """
    total_bet = 0
    total_return = 0

    for pred, odd, won in zip(predictions, odds, actual_wins):
        # Bet if prediction is confident (you can adjust threshold)
        if pred > 0.5:
            total_bet += bet_amount

            if won:
                # Calculate payout from American odds
                profit = bet_amount * (100 / abs(odd)) if odd < 0 else bet_amount * (odd / 100)

                total_return += bet_amount + profit
            # else: lost bet, return is 0

    roi = ((total_return - total_bet) / total_bet * 100) if total_bet > 0 else 0
    profit = total_return - total_bet

    return {
        "total_bet": total_bet,
        "total_return": total_return,
        "profit": profit,
        "roi": roi,
        "num_bets": np.sum(predictions > 0.5),
    }


def find_arbitrage_opportunities(odds_df: pd.DataFrame, min_roi: float = 0.01) -> pd.DataFrame:
    """
    Find arbitrage opportunities across multiple sportsbooks.

    Args:
        odds_df: DataFrame with odds from multiple sportsbooks
        min_roi: Minimum ROI threshold for arbitrage (default: 1%)

    Returns:
        DataFrame with arbitrage opportunities
    """
    arbitrage_opps = []

    # Group by game
    for game_id, game_odds in odds_df.groupby("game_id"):
        # Get home and away teams
        home_team = game_odds["home_team"].iloc[0]
        away_team = game_odds["away_team"].iloc[0]

        # Get odds for each team across sportsbooks
        home_odds = game_odds[game_odds["team"] == home_team]
        away_odds = game_odds[game_odds["team"] == away_team]

        if len(home_odds) == 0 or len(away_odds) == 0:
            continue

        # Find best odds for each outcome
        best_home_odds = home_odds.loc[home_odds["odds"].idxmax()]
        best_away_odds = away_odds.loc[away_odds["odds"].idxmax()]

        # Calculate implied probabilities
        home_prob = american_to_probability(best_home_odds["odds"])
        away_prob = american_to_probability(best_away_odds["odds"])

        # Check for arbitrage (total probability < 1)
        total_prob = home_prob + away_prob

        if total_prob < 1:
            arbitrage_roi = (1 / total_prob - 1) * 100

            if arbitrage_roi >= min_roi:
                arbitrage_opps.append(
                    {
                        "game_id": game_id,
                        "home_team": home_team,
                        "away_team": away_team,
                        "home_sportsbook": best_home_odds["sportsbook"],
                        "home_odds": best_home_odds["odds"],
                        "away_sportsbook": best_away_odds["sportsbook"],
                        "away_odds": best_away_odds["odds"],
                        "arbitrage_roi": arbitrage_roi,
                        "home_stake_pct": home_prob / total_prob * 100,
                        "away_stake_pct": away_prob / total_prob * 100,
                    }
                )

    return pd.DataFrame(arbitrage_opps)


def calculate_kelly_criterion(win_prob: float, odds: float, fraction: float = 0.25) -> float:
    """
    Calculate optimal bet size using Kelly Criterion.

    Args:
        win_prob: Probability of winning (0 to 1)
        odds: American odds
        fraction: Fraction of Kelly to use (0.25 = quarter Kelly)

    Returns:
        Optimal bet size as fraction of bankroll

    Raises:
        ValueError: If win_prob not in (0, 1), odds invalid, or fraction not in (0, 1]
    """
    # Validate inputs
    if not (0 < win_prob < 1):
        raise ValueError(f"Invalid win probability: {win_prob}. Must be between 0 and 1")
    if odds == 0:
        raise ValueError(f"Invalid American odds: {odds}. Odds cannot be zero")
    if not (0 < fraction <= 1):
        raise ValueError(f"Invalid Kelly fraction: {fraction}. Must be between 0 and 1")

    # Convert American odds to decimal
    decimal_odds = 1 + (100 / abs(odds)) if odds < 0 else 1 + (odds / 100)

    # Kelly formula: (bp - q) / b
    # where b = decimal odds - 1, p = win prob, q = 1 - p
    b = decimal_odds - 1
    p = win_prob
    q = 1 - p

    kelly = (b * p - q) / b

    # Apply fractional Kelly
    kelly = max(0, kelly * fraction)

    return kelly


def estimate_game_variance(win_prob: float, odds: float) -> float:
    """
    Calculate variance of return for single game bet.

    Args:
        win_prob: Probability of winning (0 to 1)
        odds: American odds

    Returns:
        Variance of return
    """
    # Convert odds to payout ratio
    payout = 100 / abs(odds) if odds < 0 else odds / 100

    # Expected return
    expected_return = win_prob * payout - (1 - win_prob) * 1

    # Variance: E[X²] - E[X]²
    expected_return_squared = win_prob * (payout**2) + (1 - win_prob) * ((-1) ** 2)
    variance = expected_return_squared - (expected_return**2)

    return variance


def calculate_portfolio_metrics(
    predictions: np.ndarray, odds: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate expected returns and variances for portfolio optimization.

    Args:
        predictions: Predicted win probabilities for each game
        odds: American odds for each game

    Returns:
        Tuple of (expected_returns, variances)
    """
    n_games = len(predictions)
    expected_returns = np.zeros(n_games)
    variances = np.zeros(n_games)

    for i in range(n_games):
        # Payout ratio
        payout = 100 / abs(odds[i]) if odds[i] < 0 else odds[i] / 100

        # Expected return
        expected_returns[i] = predictions[i] * payout - (1 - predictions[i])

        # Variance
        variances[i] = estimate_game_variance(predictions[i], odds[i])

    return expected_returns, variances


def optimize_markowitz_portfolio(
    expected_returns: np.ndarray,
    covariance_matrix: np.ndarray,
    risk_aversion: float = 2.0,
    max_position: float = 0.3,
) -> np.ndarray:
    """
    Optimize portfolio weights using mean-variance framework.

    Maximizes: w^T μ - risk_aversion × w^T Σ w
    Subject to: w_i ≥ 0, w_i ≤ max_position, Σw_i = 1

    Args:
        expected_returns: Expected return for each game
        covariance_matrix: Covariance matrix of returns
        risk_aversion: Risk aversion coefficient (higher = more conservative)
        max_position: Maximum fraction of bankroll for single bet

    Returns:
        Optimal portfolio weights

    Raises:
        ValueError: If inputs have invalid shapes or values
    """
    # Validate inputs
    if not isinstance(expected_returns, np.ndarray) or not isinstance(
        covariance_matrix, np.ndarray
    ):
        raise ValueError("expected_returns and covariance_matrix must be numpy arrays")

    n = len(expected_returns)

    if n == 0:
        raise ValueError("expected_returns cannot be empty")
    if covariance_matrix.shape != (n, n):
        raise ValueError(
            f"covariance_matrix shape {covariance_matrix.shape} must match expected_returns length {n}"
        )
    if risk_aversion <= 0:
        raise ValueError(f"risk_aversion must be positive, got {risk_aversion}")
    if not (0 < max_position <= 1):
        raise ValueError(f"max_position must be in (0, 1], got {max_position}")

    # Objective: maximize risk-adjusted return
    def objective(w):
        return -(w @ expected_returns - risk_aversion * w @ covariance_matrix @ w)

    # Constraints: sum of weights must equal 1 (fully invest bankroll)
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

    # Bounds: 0 <= w_i <= max_position
    bounds = [(0, max_position) for _ in range(n)]

    # Initial guess
    w0 = np.ones(n) / n

    # Optimize
    result = minimize(
        objective,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-9, "maxiter": 1000},
    )

    if result.success:
        return result.x
    else:
        print(f"Optimization warning: {result.message}")
        return np.ones(n) / n  # Fallback to equal weights


def calculate_markowitz_roi(
    predictions: np.ndarray,
    odds: np.ndarray,
    actual_wins: np.ndarray,
    dates: np.ndarray,
    bankroll: float = 10000,
    risk_aversion: float = 2.0,
    max_position: float = 0.3,
) -> Dict[str, float]:
    """
    Calculate ROI using Markowitz portfolio optimization strategy.

    Groups games by date and optimizes bet allocation within each date.

    Args:
        predictions: Predicted win probabilities
        odds: American odds
        actual_wins: Actual outcomes (1 if won, 0 if lost)
        dates: Date for each game (for grouping)
        bankroll: Starting bankroll
        risk_aversion: Risk aversion parameter
        max_position: Maximum fraction per single bet

    Returns:
        Dictionary with ROI metrics
    """
    current_bankroll = bankroll
    total_bet = 0
    total_return = 0
    num_bets = 0

    # Group by date
    unique_dates = np.unique(dates)

    for date in unique_dates:
        date_mask = dates == date
        date_preds = predictions[date_mask]
        date_odds = odds[date_mask]
        date_wins = actual_wins[date_mask]

        # Filter games with edge (prediction > implied probability + 5%)
        edges = []
        for i, (pred, odd) in enumerate(zip(date_preds, date_odds)):
            # Convert odds to implied probability
            implied_prob = abs(odd) / (abs(odd) + 100) if odd < 0 else 100 / (odd + 100)

            if pred > implied_prob + 0.05:  # Require 5% edge
                edges.append(i)

        if len(edges) == 0:
            continue

        # Subset to games with edge
        edge_preds = date_preds[edges]
        edge_odds = date_odds[edges]
        edge_wins = date_wins[edges]

        # Calculate portfolio metrics
        exp_returns, variances = calculate_portfolio_metrics(edge_preds, edge_odds)

        # Covariance matrix (diagonal - assume independent games)
        cov_matrix = np.diag(variances)

        # Optimize weights
        weights = optimize_markowitz_portfolio(
            exp_returns, cov_matrix, risk_aversion=risk_aversion, max_position=max_position
        )

        # Place bets
        for _i, (weight, odd, won) in enumerate(zip(weights, edge_odds, edge_wins)):
            bet_amount = weight * current_bankroll

            if bet_amount < 1:  # Minimum bet $1
                continue

            total_bet += bet_amount
            num_bets += 1

            if won:
                # Calculate profit
                profit = bet_amount * (100 / abs(odd)) if odd < 0 else bet_amount * (odd / 100)

                payout = bet_amount + profit
                total_return += payout
                current_bankroll += profit
            else:
                # Lost bet
                current_bankroll -= bet_amount

    roi = ((total_return - total_bet) / total_bet * 100) if total_bet > 0 else 0
    profit = total_return - total_bet

    return {
        "total_bet": total_bet,
        "total_return": total_return,
        "profit": profit,
        "roi": roi,
        "num_bets": num_bets,
        "final_bankroll": current_bankroll,
    }


def calculate_kelly_roi(
    predictions: np.ndarray,
    odds: np.ndarray,
    actual_wins: np.ndarray,
    bankroll: float = 10000,
    fraction: float = 0.25,
) -> Dict[str, float]:
    """
    Calculate ROI using Kelly Criterion betting strategy.

    Args:
        predictions: Predicted win probabilities
        odds: American odds
        actual_wins: Actual outcomes (1 if won, 0 if lost)
        bankroll: Starting bankroll
        fraction: Fraction of Kelly to use (0.25 = quarter Kelly)

    Returns:
        Dictionary with ROI metrics
    """
    current_bankroll = bankroll
    total_bet = 0
    total_return = 0
    num_bets = 0

    for pred, odd, won in zip(predictions, odds, actual_wins):
        # Calculate Kelly bet size
        kelly_fraction = calculate_kelly_criterion(pred, odd, fraction=fraction)
        bet_amount = kelly_fraction * current_bankroll

        # Only bet if Kelly suggests betting and minimum bet met
        if bet_amount >= 1:
            total_bet += bet_amount
            num_bets += 1

            if won:
                # Calculate profit
                profit = bet_amount * (100 / abs(odd)) if odd < 0 else bet_amount * (odd / 100)

                payout = bet_amount + profit
                total_return += payout
                current_bankroll += profit
            else:
                # Lost bet
                current_bankroll -= bet_amount

    roi = ((total_return - total_bet) / total_bet * 100) if total_bet > 0 else 0
    profit = total_return - total_bet

    return {
        "total_bet": total_bet,
        "total_return": total_return,
        "profit": profit,
        "roi": roi,
        "num_bets": num_bets,
        "final_bankroll": current_bankroll,
    }


def combine_model_predictions(
    predictions_dict: Dict[str, np.ndarray], weights: Optional[Dict[str, float]] = None
) -> np.ndarray:
    """
    Combine predictions from multiple models.

    Args:
        predictions_dict: Dictionary mapping model names to prediction arrays
        weights: Optional weights for each model (must sum to 1)

    Returns:
        Combined predictions array
    """
    if weights is None:
        # Equal weights
        weights = {name: 1.0 / len(predictions_dict) for name in predictions_dict}

    # Validate weights
    assert abs(sum(weights.values()) - 1.0) < 1e-6, "Weights must sum to 1"

    # Combine
    combined = np.zeros_like(next(iter(predictions_dict.values())))

    for name, preds in predictions_dict.items():
        combined += preds * weights[name]

    return combined
