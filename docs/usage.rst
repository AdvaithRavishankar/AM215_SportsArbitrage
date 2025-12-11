Usage Guide
===========

This guide provides examples of how to use the Sports Betting Arbitrage package.

Basic Usage
-----------

Loading Data
~~~~~~~~~~~~

.. code-block:: python

   from sports_arbitrage.utils import load_odds_data, prepare_games_data

   # Load historical odds data
   odds_df = load_odds_data('../data/odds_2020_2024_combined.csv')

   # Prepare game-level data
   games_df = prepare_games_data(odds_df)

Training Models
~~~~~~~~~~~~~~~

ELO Model
^^^^^^^^^

.. code-block:: python

   from sports_arbitrage.models.elo import ELOModel

   # Initialize and train
   elo_model = ELOModel(
       initial_rating=1500,
       k_factor=32,
       home_advantage=100
   )
   elo_model.fit(games_df)

   # Make predictions
   predictions = elo_model.predict(test_games)
   home_prob, away_prob = elo_model.predict_game('Team_A', 'Team_B')

XGBoost Model
^^^^^^^^^^^^^

.. code-block:: python

   from sports_arbitrage.models.xgboost_model import XGBoostModel

   # Initialize with custom parameters
   xgb_model = XGBoostModel(
       n_estimators=100,
       max_depth=6,
       learning_rate=0.1
   )
   xgb_model.fit(games_df)

   # Get feature importance
   importance_df = xgb_model.get_feature_importance()
   print(importance_df)

Random Forest Model
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from sports_arbitrage.models.random_forest import RandomForestModel

   rf_model = RandomForestModel(
       n_estimators=100,
       max_depth=10,
       min_samples_split=5
   )
   rf_model.fit(games_df)

Rank Centrality Model
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from sports_arbitrage.models.rank_centrality import RankCentralityModel

   rc_model = RankCentralityModel(method='pagerank', damping=0.85)
   rc_model.fit(games_df)

   # Get team rankings
   rankings = rc_model.get_rankings()

Calculating ROI
~~~~~~~~~~~~~~~

Fixed Betting
^^^^^^^^^^^^^

.. code-block:: python

   from sports_arbitrage.utils import calculate_roi

   roi_result = calculate_roi(
       games_df=test_games,
       predictions=predictions,
       bet_amount=100
   )

   print(f"ROI: {roi_result['roi']:.2f}%")
   print(f"Profit: ${roi_result['profit']:.2f}")
   print(f"Number of bets: {roi_result['num_bets']}")

Kelly Criterion
^^^^^^^^^^^^^^^

.. code-block:: python

   from sports_arbitrage.utils import calculate_kelly_roi

   kelly_result = calculate_kelly_roi(
       games_df=test_games,
       predictions=predictions,
       initial_bankroll=10000,
       kelly_fraction=0.25  # Quarter Kelly for safety
   )

   print(f"Final bankroll: ${kelly_result['final_bankroll']:.2f}")

Markowitz Portfolio
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from sports_arbitrage.utils import calculate_markowitz_roi

   markowitz_result = calculate_markowitz_roi(
       games_df=test_games,
       predictions=predictions,
       initial_bankroll=10000,
       risk_aversion=2.0,
       max_position=0.3
   )

Finding Arbitrage
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from sports_arbitrage.utils import find_arbitrage_opportunities

   # Find arbitrage opportunities
   arb_opps = find_arbitrage_opportunities(
       odds_df=odds_df,
       min_roi=0.01  # Minimum 1% return
   )

   # Display top opportunities
   top_arbs = arb_opps.nlargest(5, 'arbitrage_roi')
   for _, arb in top_arbs.iterrows():
       print(f"{arb['away_team']} @ {arb['home_team']}")
       print(f"  ROI: {arb['arbitrage_roi']:.2f}%")
       print(f"  Home stake: {arb['home_stake_pct']:.1f}%")
       print(f"  Away stake: {arb['away_stake_pct']:.1f}%")

Visualization
~~~~~~~~~~~~~

.. code-block:: python

   from sports_arbitrage.plotting import (
       plot_roi_comparison,
       plot_calibration_curve,
       plot_feature_importance
   )

   # ROI comparison
   plot_roi_comparison(
       fixed_roi_results=roi_results,
       kelly_roi_results=kelly_results,
       markowitz_roi_results=markowitz_results,
       save_path='roi_comparison.png'
   )

   # Calibration curve
   plot_calibration_curve(
       y_true=actuals,
       y_pred=predictions,
       model_name='ELO',
       save_path='calibration.png'
   )

Cross-Validation
~~~~~~~~~~~~~~~~

.. code-block:: python

   from sports_arbitrage.utils import create_rolling_windows, calculate_metrics

   # Create time-series cross-validation folds
   folds = create_rolling_windows(games_df, n_folds=3)

   # Evaluate model on each fold
   for train_df, test_df in folds:
       model.fit(train_df)
       predictions = model.predict(test_df)
       metrics = calculate_metrics(test_df['home_won'], predictions)
       print(f"Accuracy: {metrics['accuracy']:.3f}")

Advanced Usage
--------------

Custom Model Ensemble
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from sports_arbitrage.utils import ensemble_predictions

   # Train multiple models
   models = {
       'ELO': ELOModel(),
       'XGBoost': XGBoostModel(),
       'RandomForest': RandomForestModel()
   }

   predictions = {}
   for name, model in models.items():
       model.fit(train_df)
       predictions[name] = model.predict(test_df)

   # Combine with weights
   ensemble_pred = ensemble_predictions(
       predictions,
       weights={'ELO': 0.4, 'XGBoost': 0.3, 'RandomForest': 0.3}
   )

Odds Conversion
~~~~~~~~~~~~~~~

.. code-block:: python

   from sports_arbitrage.utils import (
       american_to_probability,
       probability_to_american
   )

   # Convert American odds to probability
   prob = american_to_probability(-110)  # 0.524
   prob = american_to_probability(150)   # 0.400

   # Convert probability to American odds
   odds = probability_to_american(0.667)  # -200
   odds = probability_to_american(0.333)  # +200
