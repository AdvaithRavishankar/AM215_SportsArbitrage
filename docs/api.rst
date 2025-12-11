API Reference
=============

This page provides detailed API documentation for all modules, classes, and functions.

Models
------

ELO Model
~~~~~~~~~

.. automodule:: sports_arbitrage.models.elo
   :members:
   :undoc-members:
   :show-inheritance:

XGBoost Model
~~~~~~~~~~~~~

.. automodule:: sports_arbitrage.models.xgboost_model
   :members:
   :undoc-members:
   :show-inheritance:

Random Forest Model
~~~~~~~~~~~~~~~~~~~

.. automodule:: sports_arbitrage.models.random_forest
   :members:
   :undoc-members:
   :show-inheritance:

Rank Centrality Model
~~~~~~~~~~~~~~~~~~~~~

.. automodule:: sports_arbitrage.models.rank_centrality
   :members:
   :undoc-members:
   :show-inheritance:

Utilities
---------

Data Processing
~~~~~~~~~~~~~~~

.. autofunction:: sports_arbitrage.utils.load_odds_data
.. autofunction:: sports_arbitrage.utils.prepare_games_data
.. autofunction:: sports_arbitrage.utils.add_game_results
.. autofunction:: sports_arbitrage.utils.create_rolling_windows

Odds Conversion
~~~~~~~~~~~~~~~

.. autofunction:: sports_arbitrage.utils.american_to_probability
.. autofunction:: sports_arbitrage.utils.probability_to_american

Metrics
~~~~~~~

.. autofunction:: sports_arbitrage.utils.calculate_metrics
.. autofunction:: sports_arbitrage.utils.calculate_roi
.. autofunction:: sports_arbitrage.utils.calculate_kelly_roi
.. autofunction:: sports_arbitrage.utils.calculate_markowitz_roi

Arbitrage
~~~~~~~~~

.. autofunction:: sports_arbitrage.utils.find_arbitrage_opportunities

Optimization
~~~~~~~~~~~~

.. autofunction:: sports_arbitrage.utils.calculate_kelly_criterion
.. autofunction:: sports_arbitrage.utils.optimize_markowitz_portfolio

Ensemble
~~~~~~~~

.. autofunction:: sports_arbitrage.utils.ensemble_predictions

Plotting
--------

Model Comparison
~~~~~~~~~~~~~~~~

.. autofunction:: sports_arbitrage.plotting.plot_model_comparison
.. autofunction:: sports_arbitrage.plotting.plot_roi_comparison
.. autofunction:: sports_arbitrage.plotting.plot_metrics_heatmap

Model Evaluation
~~~~~~~~~~~~~~~~

.. autofunction:: sports_arbitrage.plotting.plot_calibration_curve
.. autofunction:: sports_arbitrage.plotting.plot_feature_importance
.. autofunction:: sports_arbitrage.plotting.plot_prediction_distribution

Strategy Analysis
~~~~~~~~~~~~~~~~~

.. autofunction:: sports_arbitrage.plotting.plot_strategy_roi_comparison
.. autofunction:: sports_arbitrage.plotting.plot_strategy_profit_comparison
.. autofunction:: sports_arbitrage.plotting.plot_strategy_bet_frequency
.. autofunction:: sports_arbitrage.plotting.plot_strategy_best_strategy_counts

Arbitrage
~~~~~~~~~

.. autofunction:: sports_arbitrage.plotting.plot_arbitrage_opportunities
