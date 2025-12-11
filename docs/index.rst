Sports Betting Arbitrage Analysis
===================================

A comprehensive sports betting analysis and arbitrage detection system using machine learning models and portfolio optimization techniques.

.. image:: https://github.com/AdvaithRavishankar/AM215_SportsArbitrage/actions/workflows/python-app.yml/badge.svg
   :target: https://github.com/AdvaithRavishankar/AM215_SportsArbitrage/actions
   :alt: CI Status

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License: MIT

Overview
--------

This project implements a complete pipeline for sports betting analysis, including:

- **Prediction Models**: ELO ratings, PageRank-based rankings, XGBoost, and Random Forest
- **Arbitrage Detection**: Find risk-free betting opportunities across sportsbooks
- **Portfolio Optimization**: Kelly Criterion and Markowitz mean-variance optimization
- **Performance Metrics**: Accuracy, ROC AUC, Brier score, log loss, and ROI analysis
- **Visualization**: Comprehensive plotting for model comparison and strategy analysis

Features
--------

✅ **Multiple Prediction Models**
   - ELO rating system with home advantage
   - Rank Centrality using PageRank algorithm
   - XGBoost gradient boosting
   - Random Forest ensemble learning

✅ **Betting Strategies**
   - Fixed bet sizing
   - Kelly Criterion with fractional Kelly
   - Markowitz portfolio optimization

✅ **Data Processing**
   - NFL game data from 2020-2024
   - Odds aggregation across multiple sportsbooks
   - Time-series cross-validation

✅ **Arbitrage Detection**
   - Automatic identification of arbitrage opportunities
   - Optimal stake calculation
   - Risk-free profit estimation

Quick Start
-----------

Installation::

   pip install -e .

Run analysis::

   cd src
   python run.py

Run tests::

   pytest tests/ -v

Documentation Contents
----------------------

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   usage
   models
   api

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
