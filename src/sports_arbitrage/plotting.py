"""
Plotting and visualization functions for sports betting analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
from matplotlib.figure import Figure


# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def plot_model_comparison(results_df: pd.DataFrame, metric: str = 'accuracy',
                          save_path: Optional[str] = None) -> Figure:
    """
    Plot comparison of model performance across folds.

    Args:
        results_df: DataFrame with columns ['model', 'fold', metric]
        metric: Metric to plot
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Create box plot
    models = results_df['model'].unique()
    data_to_plot = [results_df[results_df['model'] == model][metric].values
                    for model in models]

    bp = ax.boxplot(data_to_plot, labels=models, patch_artist=True)

    # Color the boxes
    colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']
    for patch, color in zip(bp['boxes'], colors[:len(models)]):
        patch.set_facecolor(color)

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
    ax.set_title(f'Model Comparison - {metric.replace("_", " ").title()}', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Add mean line
    means = [results_df[results_df['model'] == model][metric].mean()
             for model in models]
    ax.plot(range(1, len(models) + 1), means, 'ro-', label='Mean', linewidth=2)
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_roi_comparison(roi_results: Dict[str, Dict], save_path: Optional[str] = None) -> Figure:
    """
    Plot ROI comparison across models.

    Args:
        roi_results: Dict mapping model names to ROI metrics
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    models = list(roi_results.keys())
    roi_values = [roi_results[model]['roi'] for model in models]
    profit_values = [roi_results[model]['profit'] for model in models]

    # ROI bar plot
    colors = ['green' if roi > 0 else 'red' for roi in roi_values]
    ax1.bar(models, roi_values, color=colors, alpha=0.7, edgecolor='black')
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.set_xlabel('Model', fontsize=12)
    ax1.set_ylabel('ROI (%)', fontsize=12)
    ax1.set_title('Return on Investment by Model', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)

    # Add value labels
    for i, (model, roi) in enumerate(zip(models, roi_values)):
        ax1.text(i, roi + (1 if roi > 0 else -1), f'{roi:.2f}%',
                ha='center', va='bottom' if roi > 0 else 'top', fontsize=10)

    # Profit bar plot
    colors = ['green' if profit > 0 else 'red' for profit in profit_values]
    ax2.bar(models, profit_values, color=colors, alpha=0.7, edgecolor='black')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('Model', fontsize=12)
    ax2.set_ylabel('Profit ($)', fontsize=12)
    ax2.set_title('Total Profit by Model', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    # Add value labels
    for i, (model, profit) in enumerate(zip(models, profit_values)):
        ax2.text(i, profit + (50 if profit > 0 else -50), f'${profit:.2f}',
                ha='center', va='bottom' if profit > 0 else 'top', fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_calibration_curve(y_true: np.ndarray, y_pred: np.ndarray,
                           model_name: str = 'Model', n_bins: int = 10,
                           save_path: Optional[str] = None) -> Figure:
    """
    Plot calibration curve to assess prediction reliability.

    Args:
        y_true: True labels
        y_pred: Predicted probabilities
        model_name: Name of the model
        n_bins: Number of bins for calibration
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # Calculate calibration
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_freq = np.zeros(n_bins)
    bin_true = np.zeros(n_bins)

    for i in range(n_bins):
        mask = (y_pred >= bins[i]) & (y_pred < bins[i + 1])
        if mask.sum() > 0:
            bin_freq[i] = mask.sum()
            bin_true[i] = y_true[mask].mean()
        else:
            bin_true[i] = np.nan

    # Plot
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    ax.plot(bin_centers, bin_true, 'o-', label=model_name, linewidth=2, markersize=8)

    ax.set_xlabel('Predicted Probability', fontsize=12)
    ax.set_ylabel('True Frequency', fontsize=12)
    ax.set_title(f'Calibration Curve - {model_name}', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_feature_importance(importance_df: pd.DataFrame, model_name: str = 'Model',
                            top_n: int = 10, save_path: Optional[str] = None) -> Figure:
    """
    Plot feature importance.

    Args:
        importance_df: DataFrame with columns ['feature', 'importance']
        model_name: Name of the model
        top_n: Number of top features to show
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Get top N features
    top_features = importance_df.head(top_n)

    # Plot
    ax.barh(range(len(top_features)), top_features['importance'].values,
            color='steelblue', alpha=0.7, edgecolor='black')
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'].values)
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title(f'Feature Importance - {model_name}', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    # Invert y-axis to have most important at top
    ax.invert_yaxis()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_prediction_distribution(predictions: Dict[str, np.ndarray],
                                 save_path: Optional[str] = None) -> Figure:
    """
    Plot distribution of predictions for each model.

    Args:
        predictions: Dict mapping model names to prediction arrays
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, (model_name, preds) in enumerate(predictions.items()):
        if idx >= len(axes):
            break

        ax = axes[idx]
        ax.hist(preds, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
        ax.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Decision Threshold')
        ax.set_xlabel('Predicted Probability', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title(f'{model_name} - Prediction Distribution', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)

    # Hide unused subplots
    for idx in range(len(predictions), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_arbitrage_opportunities(arb_df: pd.DataFrame, save_path: Optional[str] = None) -> Figure:
    """
    Plot arbitrage opportunities.

    Args:
        arb_df: DataFrame with arbitrage opportunities
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    if len(arb_df) == 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No Arbitrage Opportunities Found',
               ha='center', va='center', fontsize=16)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.axis('off')
        return fig

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # ROI distribution
    ax1.hist(arb_df['arbitrage_roi'], bins=20, alpha=0.7, color='green', edgecolor='black')
    ax1.set_xlabel('Arbitrage ROI (%)', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Distribution of Arbitrage ROI', fontsize=14, fontweight='bold')
    ax1.grid(alpha=0.3)

    # Top opportunities
    top_arb = arb_df.nlargest(10, 'arbitrage_roi')
    game_labels = [f"{row['away_team'][:10]} @ {row['home_team'][:10]}"
                   for _, row in top_arb.iterrows()]

    ax2.barh(range(len(top_arb)), top_arb['arbitrage_roi'].values,
            color='green', alpha=0.7, edgecolor='black')
    ax2.set_yticks(range(len(top_arb)))
    ax2.set_yticklabels(game_labels, fontsize=9)
    ax2.set_xlabel('Arbitrage ROI (%)', fontsize=12)
    ax2.set_title('Top 10 Arbitrage Opportunities', fontsize=14, fontweight='bold')
    ax2.invert_yaxis()
    ax2.grid(axis='x', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_cumulative_profit(bets_df: pd.DataFrame, save_path: Optional[str] = None) -> Figure:
    """
    Plot cumulative profit over time.

    Args:
        bets_df: DataFrame with columns ['date', 'profit', 'model']
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    for model in bets_df['model'].unique():
        model_bets = bets_df[bets_df['model'] == model].sort_values('date')
        cumulative_profit = model_bets['profit'].cumsum()
        ax.plot(model_bets['date'], cumulative_profit, label=model, linewidth=2)

    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Cumulative Profit ($)', fontsize=12)
    ax.set_title('Cumulative Profit Over Time', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(alpha=0.3)

    plt.xticks(rotation=45)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_metrics_heatmap(results_df: pd.DataFrame, save_path: Optional[str] = None) -> Figure:
    """
    Plot heatmap of metrics across models and folds.

    Args:
        results_df: DataFrame with model evaluation results
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    # Pivot data
    metrics = ['accuracy', 'log_loss', 'brier_score', 'roc_auc']
    available_metrics = [m for m in metrics if m in results_df.columns]

    fig, axes = plt.subplots(1, len(available_metrics), figsize=(4 * len(available_metrics), 6))

    if len(available_metrics) == 1:
        axes = [axes]

    for idx, metric in enumerate(available_metrics):
        pivot_data = results_df.pivot_table(
            values=metric,
            index='model',
            columns='fold',
            aggfunc='mean'
        )

        sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='RdYlGn',
                   ax=axes[idx], cbar_kws={'label': metric.replace('_', ' ').title()})
        axes[idx].set_title(f'{metric.replace("_", " ").title()} by Model and Fold',
                           fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Fold', fontsize=11)
        axes[idx].set_ylabel('Model', fontsize=11)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig
