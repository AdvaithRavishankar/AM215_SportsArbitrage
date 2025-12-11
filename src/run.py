"""
Main script to run sports betting arbitrage analysis.

This script:
1. Loads odds data
2. Prepares game data with results
3. Performs 3-fold rolling window cross-validation
4. Trains and evaluates all models (ELO, Rank Centrality, XGBoost, Random Forest)
5. Calculates ROI and arbitrage opportunities
6. Generates visualizations
7. Uses last 20% of data as final test set
"""

import os
import numpy as np
import pandas as pd
import warnings
from datetime import datetime

from sports_arbitrage.models.elo import ELOModel
from sports_arbitrage.models.rank_centrality import RankCentralityModel
from sports_arbitrage.models.xgboost_model import XGBoostModel
from sports_arbitrage.models.random_forest import RandomForestModel

from sports_arbitrage.utils import (
    load_odds_data,
    prepare_games_data,
    add_game_results,
    create_rolling_windows,
    calculate_metrics,
    calculate_roi,
    calculate_kelly_roi,
    calculate_markowitz_roi,
    find_arbitrage_opportunities,
    american_to_probability
)

from sports_arbitrage.plotting import (
    plot_model_comparison,
    plot_roi_comparison,
    plot_calibration_curve,
    plot_feature_importance,
    plot_prediction_distribution,
    plot_arbitrage_opportunities,
    plot_metrics_heatmap,
    plot_strategy_roi_comparison,
    plot_strategy_profit_comparison,
    plot_strategy_bet_frequency,
    plot_strategy_best_strategy_counts
)

warnings.filterwarnings('ignore')


def simulate_game_results(games_df: pd.DataFrame) -> pd.DataFrame:
    """
    Simulate game results based on average odds.
    In production, you would use actual game results.

    Args:
        games_df: DataFrame with game data and odds

    Returns:
        DataFrame with simulated results
    """
    df = games_df.copy()

    # Use average odds to determine implied probability
    if 'home_avg_odds' in df.columns and 'away_avg_odds' in df.columns:
        df['home_implied_prob'] = df['home_avg_odds'].apply(
            lambda x: american_to_probability(x) if pd.notna(x) else 0.5
        )

        # Add some randomness to make it realistic
        df['home_true_prob'] = df['home_implied_prob'] + np.random.normal(0, 0.05, len(df))
        df['home_true_prob'] = df['home_true_prob'].clip(0.1, 0.9)

        # Simulate results
        df['home_won'] = np.random.random(len(df)) < df['home_true_prob']
    else:
        # If no odds, use 50/50
        df['home_won'] = np.random.random(len(df)) < 0.5

    return df


def main():
    """Main execution function."""

    print("=" * 80)
    print("SPORTS BETTING ARBITRAGE ANALYSIS")
    print("=" * 80)
    print()

    # Create output directory
    os.makedirs('../results', exist_ok=True)
    os.makedirs('../results/figures', exist_ok=True)

    # =========================================================================
    # 1. LOAD AND PREPARE DATA
    # =========================================================================
    print("1. Loading and preparing data...")
    print("-" * 80)

    # Load odds data
    odds_file = '../data/odds_2020_2024_combined.csv'
    if not os.path.exists(odds_file):
        print(f"ERROR: {odds_file} not found!")
        print("Please ensure the data file exists in the data/ directory.")
        return

    odds_df = load_odds_data(odds_file)
    print(f"   Loaded {len(odds_df):,} odds records")
    print(f"   Date range: {odds_df['commence_time'].min()} to {odds_df['commence_time'].max()}")

    # Prepare game-level data
    games_df = prepare_games_data(odds_df)
    print(f"   Prepared {len(games_df):,} unique games")

    # Load actual game results
    results_file = '../data/nfl_games_with_stats.csv'
    if os.path.exists(results_file):
        print(f"   Loading actual game results from {results_file}...")
        results_df = pd.read_csv(results_file, parse_dates=['commence_time'])

        # Create home_won column from scores
        results_df['home_won'] = (results_df['home_score'] > results_df['away_score']).astype(int)

        # Create matching key based on full team names and date
        # Use home_team_full and away_team_full to match with odds file
        results_df['match_key'] = (
            results_df['home_team_full'] + '_' +
            results_df['away_team_full'] + '_' +
            results_df['commence_time'].dt.strftime('%Y-%m-%d')
        )

        # Keep only necessary columns and drop duplicates
        results_df = results_df[['match_key', 'home_won']].drop_duplicates(subset=['match_key'])

        # Create matching key in games_df
        games_df['match_key'] = (
            games_df['home_team'] + '_' +
            games_df['away_team'] + '_' +
            games_df['commence_time'].dt.strftime('%Y-%m-%d')
        )

        # Merge results
        games_df = games_df.merge(results_df, on='match_key', how='left')
        games_df = games_df.drop(columns=['match_key'])

        matched_games = games_df['home_won'].notna().sum()
        print(f"   Matched actual results for {matched_games:,} games")

        # Filter out games without actual results to ensure reproducibility
        unmatched_count = games_df['home_won'].isna().sum()
        if unmatched_count > 0:
            print(f"   Filtering out {unmatched_count:,} games without actual results (likely future games)...")
            games_df = games_df[games_df['home_won'].notna()].copy()
            print(f"   Using {len(games_df):,} games with actual historical results")

        # Ensure home_won is boolean type (not float)
        games_df['home_won'] = games_df['home_won'].astype(bool)
    else:
        # Fallback to simulation if results file not found
        print(f"   WARNING: {results_file} not found, simulating results...")
        games_df = simulate_game_results(games_df)
        games_df['home_won'] = games_df['home_won'].astype(bool)
    print()

    # =========================================================================
    # 2. SPLIT DATA
    # =========================================================================
    print("2. Splitting data...")
    print("-" * 80)

    # Sort by date
    games_df = games_df.sort_values('commence_time').reset_index(drop=True)

    # Split: first 80% for CV, last 20% for final test
    split_idx = int(len(games_df) * 0.8)
    cv_data = games_df.iloc[:split_idx].copy()
    test_data = games_df.iloc[split_idx:].copy()

    print(f"   Cross-validation data: {len(cv_data):,} games")
    print(f"   Final test data: {len(test_data):,} games")
    print()

    # =========================================================================
    # 3. CREATE ROLLING WINDOW FOLDS
    # =========================================================================
    print("3. Creating rolling window folds for cross-validation...")
    print("-" * 80)

    folds = create_rolling_windows(cv_data, n_folds=3)
    print(f"   Created {len(folds)} folds")

    for i, (train_df, val_df) in enumerate(folds):
        print(f"   Fold {i+1}: Train={len(train_df):,}, Val={len(val_df):,}")
    print()

    # =========================================================================
    # 4. TRAIN AND EVALUATE MODELS
    # =========================================================================
    print("4. Training and evaluating models...")
    print("-" * 80)

    models = {
        'ELO': ELOModel(k_factor=20, initial_rating=1500, home_advantage=50),
        'Rank Centrality': RankCentralityModel(method='pagerank', damping_factor=0.85),
        'XGBoost': XGBoostModel(n_estimators=100, max_depth=6, learning_rate=0.1),
        'Random Forest': RandomForestModel(n_estimators=100, max_depth=10)
    }

    cv_results = []
    all_predictions = {name: [] for name in models.keys()}
    all_actuals = []

    for fold_idx, (train_df, val_df) in enumerate(folds):
        print(f"\n   Fold {fold_idx + 1}/{len(folds)}")
        print(f"   {'-' * 40}")

        for model_name, model in models.items():
            # Reset model
            model.reset()

            # Train
            print(f"      Training {model_name}...", end=" ")
            model.fit(train_df)

            # Predict
            predictions = model.predict(val_df)
            actuals = val_df['home_won'].values.astype(int)

            # Calculate metrics
            metrics = calculate_metrics(actuals, predictions)

            # Store results
            cv_results.append({
                'model': model_name,
                'fold': fold_idx + 1,
                **metrics
            })

            # Store predictions for final analysis
            all_predictions[model_name].extend(predictions)

            print(f"Accuracy: {metrics['accuracy']:.3f}, Log Loss: {metrics['log_loss']:.3f}")

        # Store actuals once per fold
        all_actuals.extend(val_df['home_won'].values.astype(int))

    # Convert to DataFrame
    cv_results_df = pd.DataFrame(cv_results)

    print("\n   Cross-Validation Results Summary:")
    print("   " + "=" * 40)
    summary = cv_results_df.groupby('model')[['accuracy', 'log_loss', 'brier_score', 'roc_auc']].mean()
    print(summary.to_string())
    print()

    # =========================================================================
    # 5. FINAL TEST SET EVALUATION
    # =========================================================================
    print("5. Evaluating on final test set (last 20%)...")
    print("-" * 80)

    test_results = []
    test_predictions = {}

    for model_name, model in models.items():
        # Reset and train on all CV data
        model.reset()
        model.fit(cv_data)

        # Predict on test set
        predictions = model.predict(test_data)
        actuals = test_data['home_won'].values.astype(int)

        # Calculate metrics
        metrics = calculate_metrics(actuals, predictions)

        test_results.append({
            'model': model_name,
            **metrics
        })

        test_predictions[model_name] = predictions

        print(f"   {model_name:20s} - Accuracy: {metrics['accuracy']:.3f}, "
              f"Log Loss: {metrics['log_loss']:.3f}, ROC AUC: {metrics['roc_auc']:.3f}")

    test_results_df = pd.DataFrame(test_results)
    print()

    # =========================================================================
    # 6. CALCULATE ROI
    # =========================================================================
    print("6. Calculating Return on Investment (ROI)...")
    print("-" * 80)

    roi_results = {}

    for model_name in models.keys():
        predictions = test_predictions[model_name]
        actuals = test_data['home_won'].values.astype(int)

        # Use home team odds for ROI calculation
        if 'home_avg_odds' in test_data.columns:
            odds = test_data['home_avg_odds'].fillna(-110).values

            roi_metrics = calculate_roi(predictions, odds, actuals, bet_amount=100)
            roi_results[model_name] = roi_metrics

            print(f"   {model_name:20s} - ROI: {roi_metrics['roi']:>7.2f}%, "
                  f"Profit: ${roi_metrics['profit']:>8.2f}, "
                  f"Bets: {int(roi_metrics['num_bets'])}")

    print()

    # =========================================================================
    # 6a. CALCULATE KELLY CRITERION ROI
    # =========================================================================
    print("6a. Calculating Kelly Criterion ROI...")
    print("-" * 80)

    kelly_roi_results = {}

    for model_name in models.keys():
        predictions = test_predictions[model_name]
        actuals = test_data['home_won'].values.astype(int)

        if 'home_avg_odds' in test_data.columns:
            odds = test_data['home_avg_odds'].fillna(-110).values

            kelly_metrics = calculate_kelly_roi(
                predictions,
                odds,
                actuals,
                bankroll=10000,
                fraction=0.25
            )
            kelly_roi_results[model_name] = kelly_metrics

            print(f"   {model_name:20s} - ROI: {kelly_metrics['roi']:>7.2f}%, "
                  f"Profit: ${kelly_metrics['profit']:>8.2f}, "
                  f"Bets: {int(kelly_metrics['num_bets'])}")

    print()

    # =========================================================================
    # 6b. CALCULATE MARKOWITZ PORTFOLIO STRATEGY ROI
    # =========================================================================
    print("6b. Calculating Markowitz Portfolio ROI...")
    print("-" * 80)

    markowitz_roi_results = {}

    for model_name in models.keys():
        predictions = test_predictions[model_name]
        actuals = test_data['home_won'].values.astype(int)
        dates = test_data['commence_time'].dt.date.values

        if 'home_avg_odds' in test_data.columns:
            odds = test_data['home_avg_odds'].fillna(-110).values

            markowitz_metrics = calculate_markowitz_roi(
                predictions,
                odds,
                actuals,
                dates,
                bankroll=10000,
                risk_aversion=2.0,
                max_position=0.3
            )
            markowitz_roi_results[model_name] = markowitz_metrics

            print(f"   {model_name:20s} - ROI: {markowitz_metrics['roi']:>7.2f}%, "
                  f"Profit: ${markowitz_metrics['profit']:>8.2f}, "
                  f"Bets: {int(markowitz_metrics['num_bets'])}")

    print()

    # Print comparison
    print("Strategy Comparison: Fixed vs Kelly vs Markowitz")
    print("-" * 100)
    print(f"{'Model':<20} {'Fixed ROI':>12} {'Kelly ROI':>12} {'Markowitz ROI':>15} {'Best Strategy':>20}")
    print("-" * 100)
    for model_name in models.keys():
        fixed_roi = roi_results[model_name]['roi']
        kelly_roi = kelly_roi_results[model_name]['roi']
        mark_roi = markowitz_roi_results[model_name]['roi']

        # Determine best strategy
        best_roi = max(fixed_roi, kelly_roi, mark_roi)
        if best_roi == fixed_roi:
            best = "Fixed"
        elif best_roi == kelly_roi:
            best = "Kelly"
        else:
            best = "Markowitz"

        print(f"{model_name:<20} {fixed_roi:>11.2f}% {kelly_roi:>11.2f}% {mark_roi:>14.2f}% {best:>20}")
    print()

    # =========================================================================
    # 7. FIND ARBITRAGE OPPORTUNITIES
    # =========================================================================
    print("7. Finding arbitrage opportunities...")
    print("-" * 80)

    # Find arbitrage in test set
    test_game_ids = test_data['game_id'].unique()
    test_odds = odds_df[odds_df['game_id'].isin(test_game_ids)]

    arb_opportunities = find_arbitrage_opportunities(test_odds, min_roi=0.01)

    print(f"   Found {len(arb_opportunities)} arbitrage opportunities")

    if len(arb_opportunities) > 0:
        print(f"   Average arbitrage ROI: {arb_opportunities['arbitrage_roi'].mean():.2f}%")
        print(f"   Max arbitrage ROI: {arb_opportunities['arbitrage_roi'].max():.2f}%")

        print("\n   Top 5 Arbitrage Opportunities:")
        print("   " + "=" * 78)
        top_5 = arb_opportunities.nlargest(5, 'arbitrage_roi')

        for idx, row in top_5.iterrows():
            print(f"   {row['away_team']} @ {row['home_team']}")
            print(f"      Home: {row['home_sportsbook']} ({row['home_odds']:+.0f}) - "
                  f"Stake: {row['home_stake_pct']:.1f}%")
            print(f"      Away: {row['away_sportsbook']} ({row['away_odds']:+.0f}) - "
                  f"Stake: {row['away_stake_pct']:.1f}%")
            print(f"      ROI: {row['arbitrage_roi']:.2f}%")
            print()

    # =========================================================================
    # 8. GENERATE VISUALIZATIONS
    # =========================================================================
    print("8. Generating visualizations...")
    print("-" * 80)

    # Model comparison
    print("   Creating model comparison plots...")
    plot_model_comparison(cv_results_df, metric='accuracy',
                         save_path='../results/figures/model_comparison_accuracy.png')
    plot_model_comparison(cv_results_df, metric='log_loss',
                         save_path='../results/figures/model_comparison_logloss.png')

    # ROI comparison
    if roi_results:
        print("   Creating ROI comparison plot...")
        plot_roi_comparison(roi_results,
                           save_path='../results/figures/roi_comparison.png')

    # Prediction distributions
    print("   Creating prediction distribution plots...")
    plot_prediction_distribution(test_predictions,
                                save_path='../results/figures/prediction_distributions.png')

    # Feature importance for tree-based models
    print("   Creating feature importance plots...")
    for model_name, model in models.items():
        if hasattr(model, 'get_feature_importance'):
            importance_df = model.get_feature_importance()
            plot_feature_importance(importance_df, model_name=model_name,
                                   save_path=f'../results/figures/feature_importance_{model_name.replace(" ", "_").lower()}.png')

    # Calibration curves
    print("   Creating calibration curves...")
    for model_name, preds in test_predictions.items():
        plot_calibration_curve(
            test_data['home_won'].values.astype(int),
            preds,
            model_name=model_name,
            save_path=f'../results/figures/calibration_{model_name.replace(" ", "_").lower()}.png'
        )

    # Arbitrage opportunities
    if len(arb_opportunities) > 0:
        print("   Creating arbitrage opportunities plot...")
        plot_arbitrage_opportunities(arb_opportunities,
                                    save_path='../results/figures/arbitrage_opportunities.png')

    # Metrics heatmap
    print("   Creating metrics heatmap...")
    plot_metrics_heatmap(cv_results_df,
                        save_path='../results/figures/metrics_heatmap.png')

    # Strategy comparison
    if roi_results and kelly_roi_results and markowitz_roi_results:
        print("   Creating strategy comparison plots...")
        plot_strategy_roi_comparison(
            roi_results,
            kelly_roi_results,
            markowitz_roi_results,
            save_path='../results/figures/strategy_roi_comparison.png'
        )
        plot_strategy_profit_comparison(
            roi_results,
            kelly_roi_results,
            markowitz_roi_results,
            save_path='../results/figures/strategy_profit_comparison.png'
        )
        plot_strategy_bet_frequency(
            roi_results,
            kelly_roi_results,
            markowitz_roi_results,
            save_path='../results/figures/strategy_bet_frequency.png'
        )
        plot_strategy_best_strategy_counts(
            roi_results,
            kelly_roi_results,
            markowitz_roi_results,
            save_path='../results/figures/strategy_best_strategy_counts.png'
        )

    print()

    # =========================================================================
    # 9. SAVE RESULTS
    # =========================================================================
    print("9. Saving results...")
    print("-" * 80)

    # Save CV results
    cv_results_df.to_csv('../results/cv_results.csv', index=False)
    print("   Saved cross-validation results to results/cv_results.csv")

    # Save test results
    test_results_df.to_csv('../results/test_results.csv', index=False)
    print("   Saved test results to results/test_results.csv")

    # Save ROI results
    if roi_results:
        roi_df = pd.DataFrame(roi_results).T
        roi_df.to_csv('../results/roi_results.csv')
        print("   Saved ROI results to results/roi_results.csv")

    # Save arbitrage opportunities
    if len(arb_opportunities) > 0:
        arb_opportunities.to_csv('../results/arbitrage_opportunities.csv', index=False)
        print("   Saved arbitrage opportunities to results/arbitrage_opportunities.csv")

    print()

    # =========================================================================
    # 10. SUMMARY
    # =========================================================================
    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print()
    print("Summary:")
    print(f"   Best model (Accuracy): {test_results_df.loc[test_results_df['accuracy'].idxmax(), 'model']}")
    print(f"   Best model (ROC AUC): {test_results_df.loc[test_results_df['roc_auc'].idxmax(), 'model']}")

    if roi_results:
        best_roi_model = max(roi_results.items(), key=lambda x: x[1]['roi'])[0]
        print(f"   Best model (ROI): {best_roi_model} ({roi_results[best_roi_model]['roi']:.2f}%)")

    print()
    print("All results and figures have been saved to the 'results/' directory.")
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
