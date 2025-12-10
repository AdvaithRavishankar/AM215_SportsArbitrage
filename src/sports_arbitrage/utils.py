"""
Utility functions for sports betting arbitrage system.

Includes data loading, preprocessing, evaluation metrics, and arbitrage calculations.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, brier_score_loss


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
    if 'commence_time' in df.columns:
        df['commence_time'] = pd.to_datetime(df['commence_time'])
    if 'snapshot_date' in df.columns:
        df['snapshot_date'] = pd.to_datetime(df['snapshot_date'])

    return df


def american_to_probability(odds: float) -> float:
    """
    Convert American odds to implied probability.

    Args:
        odds: American odds (e.g., -110, +150)

    Returns:
        Implied probability (0 to 1)
    """
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
    """
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
    games = odds_df.groupby(['game_id', 'home_team', 'away_team', 'commence_time']).first().reset_index()

    # Get average odds per team per game
    avg_odds = odds_df.groupby(['game_id', 'team'])['odds'].mean().reset_index()

    # Pivot to get home and away odds
    games_with_odds = games.copy()
    games_with_odds['home_avg_odds'] = games_with_odds.apply(
        lambda row: avg_odds[
            (avg_odds['game_id'] == row['game_id']) &
            (avg_odds['team'] == row['home_team'])
        ]['odds'].values[0] if len(avg_odds[
            (avg_odds['game_id'] == row['game_id']) &
            (avg_odds['team'] == row['home_team'])
        ]) > 0 else None,
        axis=1
    )

    games_with_odds['away_avg_odds'] = games_with_odds.apply(
        lambda row: avg_odds[
            (avg_odds['game_id'] == row['game_id']) &
            (avg_odds['team'] == row['away_team'])
        ]['odds'].values[0] if len(avg_odds[
            (avg_odds['game_id'] == row['game_id']) &
            (avg_odds['team'] == row['away_team'])
        ]) > 0 else None,
        axis=1
    )

    return games_with_odds


def add_game_results(games_df: pd.DataFrame, results_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
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

    if results_df is not None:
        # Merge with actual results
        df = df.merge(
            results_df[['game_id', 'home_won']],
            on='game_id',
            how='left'
        )
    else:
        # Simulate results based on odds (for demonstration)
        if 'home_avg_odds' in df.columns and 'away_avg_odds' in df.columns:
            df['home_prob'] = df['home_avg_odds'].apply(
                lambda x: american_to_probability(x) if pd.notna(x) else 0.5
            )
            df['home_won'] = np.random.random(len(df)) < df['home_prob']
        else:
            # Random if no odds available
            df['home_won'] = np.random.random(len(df)) < 0.5

    return df


def create_rolling_windows(df: pd.DataFrame, n_folds: int = 3) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Create rolling window train/test splits for time series cross-validation.

    Args:
        df: DataFrame sorted by time
        n_folds: Number of folds

    Returns:
        List of (train_df, test_df) tuples
    """
    df = df.sort_values('commence_time').reset_index(drop=True)
    n = len(df)
    fold_size = n // (n_folds + 1)

    folds = []
    for i in range(n_folds):
        train_end = (i + 1) * fold_size + fold_size // 2
        test_start = train_end
        test_end = test_start + fold_size

        if test_end > n:
            test_end = n

        train_df = df.iloc[:train_end].copy()
        test_df = df.iloc[test_start:test_end].copy()

        if len(test_df) > 0:
            folds.append((train_df, test_df))

    return folds


def calculate_metrics(y_true: np.ndarray, y_pred_proba: np.ndarray,
                     threshold: float = 0.5) -> Dict[str, float]:
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
        'accuracy': accuracy_score(y_true, y_pred),
        'log_loss': log_loss(y_true, y_pred_proba),
        'brier_score': brier_score_loss(y_true, y_pred_proba),
    }

    try:
        metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
    except ValueError:
        metrics['roc_auc'] = np.nan

    return metrics


def calculate_roi(predictions: np.ndarray, odds: np.ndarray,
                  actual_wins: np.ndarray, bet_amount: float = 100) -> Dict[str, float]:
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
                if odd < 0:
                    profit = bet_amount * (100 / abs(odd))
                else:
                    profit = bet_amount * (odd / 100)

                total_return += bet_amount + profit
            # else: lost bet, return is 0

    roi = ((total_return - total_bet) / total_bet * 100) if total_bet > 0 else 0
    profit = total_return - total_bet

    return {
        'total_bet': total_bet,
        'total_return': total_return,
        'profit': profit,
        'roi': roi,
        'num_bets': np.sum(predictions > 0.5)
    }


def find_arbitrage_opportunities(odds_df: pd.DataFrame,
                                  min_roi: float = 0.01) -> pd.DataFrame:
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
    for game_id, game_odds in odds_df.groupby('game_id'):
        # Get home and away teams
        home_team = game_odds['home_team'].iloc[0]
        away_team = game_odds['away_team'].iloc[0]

        # Get odds for each team across sportsbooks
        home_odds = game_odds[game_odds['team'] == home_team]
        away_odds = game_odds[game_odds['team'] == away_team]

        if len(home_odds) == 0 or len(away_odds) == 0:
            continue

        # Find best odds for each outcome
        best_home_odds = home_odds.loc[home_odds['odds'].idxmax()]
        best_away_odds = away_odds.loc[away_odds['odds'].idxmax()]

        # Calculate implied probabilities
        home_prob = american_to_probability(best_home_odds['odds'])
        away_prob = american_to_probability(best_away_odds['odds'])

        # Check for arbitrage (total probability < 1)
        total_prob = home_prob + away_prob

        if total_prob < 1:
            arbitrage_roi = (1 / total_prob - 1) * 100

            if arbitrage_roi >= min_roi:
                arbitrage_opps.append({
                    'game_id': game_id,
                    'home_team': home_team,
                    'away_team': away_team,
                    'home_sportsbook': best_home_odds['sportsbook'],
                    'home_odds': best_home_odds['odds'],
                    'away_sportsbook': best_away_odds['sportsbook'],
                    'away_odds': best_away_odds['odds'],
                    'arbitrage_roi': arbitrage_roi,
                    'home_stake_pct': home_prob / total_prob * 100,
                    'away_stake_pct': away_prob / total_prob * 100,
                })

    return pd.DataFrame(arbitrage_opps)


def calculate_kelly_criterion(win_prob: float, odds: float,
                               fraction: float = 0.25) -> float:
    """
    Calculate optimal bet size using Kelly Criterion.

    Args:
        win_prob: Probability of winning
        odds: American odds
        fraction: Fraction of Kelly to use (0.25 = quarter Kelly)

    Returns:
        Optimal bet size as fraction of bankroll
    """
    # Convert American odds to decimal
    if odds < 0:
        decimal_odds = 1 + (100 / abs(odds))
    else:
        decimal_odds = 1 + (odds / 100)

    # Kelly formula: (bp - q) / b
    # where b = decimal odds - 1, p = win prob, q = 1 - p
    b = decimal_odds - 1
    p = win_prob
    q = 1 - p

    kelly = (b * p - q) / b

    # Apply fractional Kelly
    kelly = max(0, kelly * fraction)

    return kelly


def combine_model_predictions(predictions_dict: Dict[str, np.ndarray],
                              weights: Optional[Dict[str, float]] = None) -> np.ndarray:
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
