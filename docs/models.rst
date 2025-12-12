Prediction Models
=================

This package implements four distinct prediction models for NFL game outcomes.

ELO Rating System
-----------------

The ELO rating system is a well-established method for rating competitors based on game outcomes.

**How it works:**

1. Each team starts with an initial rating (default: 1500)
2. After each game, ratings are updated based on:
   - Expected win probability (calculated from rating difference)
   - Actual outcome (win/loss)
   - K-factor (determines rating volatility)
   - Home advantage bonus

**Mathematical formulation:**

.. math::

   E_A = \frac{1}{1 + 10^{(R_B - R_A)/400}}

   R'_A = R_A + K \cdot (S_A - E_A)

Where:
   - :math:`E_A` = Expected score for team A
   - :math:`R_A` = Current rating for team A
   - :math:`K` = K-factor (default: 32)
   - :math:`S_A` = Actual score (1 for win, 0 for loss)

**Advantages:**

- Simple and interpretable
- Proven track record in sports
- Automatically adjusts to team strength changes
- Low computational cost

**Disadvantages:**

- Sensitive to initialization and K-factor
- Doesn't incorporate game-specific features (injuries, weather, etc.)
- Slower to adapt to rapid team changes

**Parameters:**

- ``initial_rating`` (int): Starting rating for new teams (default: 1500)
- ``k_factor`` (int): Rating volatility (default: 32)
- ``home_advantage`` (int): Points added to home team rating (default: 100)

Rank Centrality (PageRank)
---------------------------

This model uses Google's PageRank algorithm on a directed graph of game outcomes.

**How it works:**

1. Build a directed graph where:
   - Each team is a node
   - Edge from loser → winner (winners receive "endorsements")
   - Edge weight = number of times loser lost to winner

2. Run PageRank algorithm to compute centrality scores
3. Convert centrality scores to win probabilities

**Mathematical formulation:**

.. math::

   PR(A) = \frac{1-d}{N} + d \sum_{B \in M(A)} \frac{PR(B)}{L(B)}

Where:
   - :math:`PR(A)` = PageRank score for team A
   - :math:`d` = Damping factor (default: 0.85)
   - :math:`N` = Total number of teams
   - :math:`M(A)` = Teams that lost to A
   - :math:`L(B)` = Total losses for team B

**Advantages:**

- Captures transitive relationships (if A beats B, B beats C, then A likely beats C)
- Robust to noise
- Works well with sparse data

**Disadvantages:**

- Doesn't account for margin of victory
- Requires sufficient game history
- Can struggle with new teams

**Parameters:**

- ``method`` (str): Centrality algorithm ('pagerank', 'eigenvector', 'hits')
- ``damping`` (float): PageRank damping factor (default: 0.85)

XGBoost Gradient Boosting
--------------------------

XGBoost is a powerful gradient boosting framework that builds an ensemble of decision trees.

**How it works:**

1. Extract features for each matchup:
   - Team win rates (overall, home, away)
   - Total games played
   - Temporal features (year, month, day of week)

2. Build sequential decision trees where each tree corrects errors of previous trees
3. Combine tree predictions to get final probability

**Feature Set:**

- ``home_win_rate``: Historical win rate for home team
- ``away_win_rate``: Historical win rate for away team
- ``home_home_win_rate``: Home team's win rate at home
- ``away_away_win_rate``: Away team's win rate on road
- ``home_total_games``: Experience (number of games played)
- ``away_total_games``: Experience (number of games played)
- ``year``: Season (captures roster/rule changes)
- ``month``: Early vs late season dynamics
- ``day_of_week``: Rest patterns (Monday vs Sunday games)

**Advantages:**

- Handles non-linear relationships
- Feature importance analysis
- Robust to outliers
- High accuracy on structured data

**Disadvantages:**

- Requires hyperparameter tuning
- Can overfit with too many trees
- Less interpretable than ELO

**Parameters:**

- ``n_estimators`` (int): Number of boosting rounds (default: 100)
- ``max_depth`` (int): Maximum tree depth (default: 6)
- ``learning_rate`` (float): Step size shrinkage (default: 0.1)
- ``random_state`` (int): Random seed for reproducibility

Random Forest
-------------

Random Forest builds an ensemble of decision trees trained on random subsets of data.

**How it works:**

1. Extract same features as XGBoost, plus:
   - Recent form (win rate over last 5 games)

2. Build multiple decision trees on:
   - Random subsets of training data (bootstrap sampling)
   - Random subsets of features at each split

3. Average predictions across all trees

**Additional Features:**

- ``home_recent_form``: Home team's wins in last 5 games / 5
- ``away_recent_form``: Away team's wins in last 5 games / 5

**Advantages:**

- Reduces overfitting through averaging
- Robust to noise
- Handles missing data well
- Feature importance ranking

**Disadvantages:**

- Can be slow with many trees
- Less accurate than gradient boosting on some tasks
- Biased toward dominant classes

**Parameters:**

- ``n_estimators`` (int): Number of trees (default: 100)
- ``max_depth`` (int): Maximum tree depth (default: 10)
- ``min_samples_split`` (int): Minimum samples to split node (default: 5)
- ``random_state`` (int): Random seed

Model Comparison
----------------

+------------------+------------+---------------+------------------+-------------------+
| Model            | Complexity | Interpretable | Feature Engineer | Cold Start Problem|
+==================+============+===============+==================+===================+
| ELO              | Low        | High          | None             | Medium            |
+------------------+------------+---------------+------------------+-------------------+
| Rank Centrality  | Low        | Medium        | None             | High              |
+------------------+------------+---------------+------------------+-------------------+
| XGBoost          | High       | Low           | Required         | Low               |
+------------------+------------+---------------+------------------+-------------------+
| Random Forest    | Medium     | Low           | Required         | Low               |
+------------------+------------+---------------+------------------+-------------------+

Performance Metrics
-------------------

Models are evaluated using:

- **Accuracy**: Percentage of correct predictions
- **ROC AUC**: Area under receiver operating characteristic curve
- **Log Loss**: Logarithmic loss (penalizes confident wrong predictions)
- **Brier Score**: Mean squared difference between predicted probabilities and outcomes
- **ROI**: Return on investment when betting based on model predictions

Typical Results (NFL 2020-2024):

+------------------+----------+---------+-----------+-------------+-------+
| Model            | Accuracy | ROC AUC | Log Loss  | Brier Score | ROI   |
+==================+==========+=========+===========+=============+=======+
| ELO              | 63.6%    | 0.688   | 0.636     | 0.223       | +4.8% |
+------------------+----------+---------+-----------+-------------+-------+
| Random Forest    | 61.5%    | 0.672   | 0.645     | 0.227       | +1.7% |
+------------------+----------+---------+-----------+-------------+-------+
| XGBoost          | 57.8%    | 0.683   | 0.697     | 0.244       | -3.9% |
+------------------+----------+---------+-----------+-------------+-------+
| Rank Centrality  | 53.5%    | 0.584   | 0.678     | 0.243       | -5.6% |
+------------------+----------+---------+-----------+-------------+-------+
