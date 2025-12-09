# Flight Delay Analysis and Prediction Using 2024 US Flight Data

**Authors:** Pratik Narendra Gupta, Luke Blommesteyn, Augusts Zilakovs, Justin Zomer  
**Affiliation:** Western University

## Overview

This project presents a comprehensive analysis of the 2024 US domestic flight dataset (approx. 7 million records) to identify delay patterns and build predictive models. We employ a data science pipeline that includes rigorous data cleaning, Exploratory Data Analysis (EDA), Anomaly Detection using Isolation Forests, and Machine Learning modeling with XGBoost.

The goal is to characterize the factors contributing to delays and predict them at both the individual flight level and the aggregate daily level.

## Features

- **Comprehensive EDA**: Analysis of delays by hour, day, season, and airline.
- **Anomaly Detection**: Usage of Isolation Forest to detect and filter out non-representative flight data outliers.
- **Machine Learning Models**:
  - **Logistic Regression**: Baseline model for interpretability.
  - **XGBoost**: Advanced gradient boosting for high-performance classification.
  - **Daily Aggregation Model**: Time-series forecasting using rolling averages (`Rolling Stone`).

## Installation

Ensure you have Python 3.8+ installed. The project relies on the following major libraries:

```bash
pip install pandas matplotlib scikit-learn xgboost
```

## Dataset

The dataset used is the **2024 Domestic Flights Dataset** from Kaggle.
Ensure the dataset file `src/flight_data_2024.csv` is present in the `src/` directory before running the scripts.

## Project Structure

```
.
├── src/
│   ├── airlines.py                      # Airline reliability analysis
│   ├── anomalies.py                     # Isolation Forest anomaly detection
│   ├── data_prep.py                     # Data loading and preprocessing utilities
│   ├── delay_predictions_rolling_stone.py # Daily aggregate delay prediction model
│   ├── main.py                          # Main entry point for basic models and EDA
│   ├── no_leakage_xgboost.py            # Strict temporal split XGBoost model
│   ├── flight_data_2024.csv             # Dataset (download separately)
│   └── plots/                           # Generated visualization outputs
├── Final_Report.pdf                     # Comprehensive project report
├── Final_Report.md                      # Markdown version of the report
└── README.md                            # Project documentation
```

## Usage

### 1. Run Basic Analysis and Baseline Models

To generate initial EDA plots and run the baseline Logistic Regression and XGBoost models:

```bash
python src/main.py
```

### 2. Detect Anomalies

To run the Isolation Forest algorithm and identify outliers:

```bash
python src/anomalies.py
```

### 3. Run Rolling Window Prediction (Best Performance)

To execute the time-series forecasting model which achieved the best results ($R^2 \approx 0.78$):

```bash
python src/delay_predictions_rolling_stone.py
```

## Key Results

- **Systemic Trends**: Aggregate daily delays are highly predictable using previous days' statistics (Rolling Lag features).
- **Individual Prediction**: Predicting individual flight delays is challenging (ROC-AUC ~0.69) due to high variance and unobserved factors (weather, maintenance).
- **Anomaly Detection**: Identified ~100k data irregularities that were removed to improve model training stability.