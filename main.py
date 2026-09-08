"""
main.py
-------
Training entry-point for the LSTM Onion Price Forecasting System.

Run this script once to train the model and save artifacts:

    python main.py

The best model and scaler are saved to the models/ directory and then
loaded by app.py (the Streamlit dashboard) at runtime.
"""

from __future__ import annotations

import os
import pickle
import sys

import numpy as np

os.environ.setdefault('KERAS_BACKEND', 'torch')

from src.data_processor import analyze_frequency, combine_and_clean_sheets, inspect_excel_file
from src.dataset_prep import prepare_lookback_dataset
from src.lstm_model import evaluate_lookback_configurations, forecast_future_prices


def run_pipeline(
    excel_path: str = 'Onion Day Wise Chart Monthly 2023.xls',
    epochs: int = 80,
    batch_size: int = 32,
) -> tuple:
    """
    Full training pipeline:
        1. Load & clean Excel data (no CSV created)
        2. Prepare multi-lookback sliding-window datasets (trading days only)
        3. Train & compare LSTM models; pick the best lookback
        4. Save model + scaler artifacts
        5. Demo future forecasts for 7 / 14 / 30-day horizons
    """
    sep = '=' * 70
    print(sep)
    print('      LSTM ONION PRICE FORECASTING SYSTEM -- DIRECT EXCEL PIPELINE')
    print(sep)

    os.makedirs('models', exist_ok=True)

    # Step 1: Inspect & Load
    print('\n[STEP 1] Inspecting Excel workbook & loading sheets in memory...')
    info = inspect_excel_file(excel_path)
    n_valid = len(info['valid_sheets'])
    n_total = info['total_sheets']
    print(f'  -> Identified {n_valid} valid sheets out of {n_total} total.')

    df_clean = combine_and_clean_sheets(excel_path)
    print(f'  -> Total clean observations : {len(df_clean)} calendar days')

    # Trading-day stats
    if 'Is_Imputed' in df_clean.columns:
        n_trading = int((df_clean['Is_Imputed'] == False).sum())  # noqa: E712
        last_trade = df_clean[df_clean['Is_Imputed'] == False]['Date'].max().strftime('%Y-%m-%d')  # noqa: E712
        print(f'  -> Actual trading days     : {n_trading}')
        print(f'  -> Last trading date       : {last_trade}')

    freq = analyze_frequency(df_clean)
    print('  -> Data Frequency Analysis:')
    for key, val in freq.items():
        print(f'       {key}: {val}')

    # Step 2: Prepare Datasets for Model A (Price-Only) and Model B (Price + News)
    from src.dataset_prep import NEWS_ENHANCED_FEATURES, PRICE_FEATURES
    print('\n[STEP 2] Preparing chronological sequences & fitting MinMaxScaler...')
    print('         (Fitting scalers STRICTLY on training split only - No Data Leakage)')

    lookbacks = [7, 14]
    data_news: dict = {}

    for lb in lookbacks:
        data_news[lb] = prepare_lookback_dataset(df_clean, lookback_window=lb, feature_cols=NEWS_ENHANCED_FEATURES)
        print(
            f'  -> Lookback {lb:2d}d: '
            f'X_train Shape {data_news[lb]["X_train"].shape} | '
            f'y_train Shape {data_news[lb]["y_train"].shape}'
        )

    # Step 3: Train & Select Best Lookback
    print(f'\n[STEP 3] Training Corrected Stacked LSTM (24 inputs -> 3 price outputs, epochs={epochs}, batch={batch_size})...')
    res_news, lb_news = evaluate_lookback_configurations(data_news, epochs=epochs, batch_size=batch_size)

    print('\n' + '-' * 70)
    print(f"{'Model Architecture':<35} | {'Lookback':<10} | {'Test MAE':<10} | {'Test MAPE':<10}")
    print('-' * 70)
    m_n = res_news[lb_news]['test_metrics']
    print(f"{'Corrected LSTM (24 In -> 3 Out)':<35} | {lb_news:2d} days     | Rs.{m_n['MAE']:<7.2f} | {m_n['MAPE']:<7.2f}%")
    print('-' * 70)

    # Compute validation set residual standard error for confidence bounds
    from src.lstm_model import evaluate_model_performance
    val_m, val_true, val_pred = evaluate_model_performance(
        res_news[lb_news]['model'],
        data_news[lb_news]['X_val'],
        data_news[lb_news]['y_val'],
        res_news[lb_news]['scaler'],
    )
    val_errors = (val_true[:, 1] - val_pred[:, 1]) if val_true.ndim > 1 else (val_true - val_pred)
    std_error = float(np.std(val_errors)) if len(val_errors) > 0 else 120.0
    print(f'  -> Validation Residual Standard Error: Rs.{std_error:.2f} / Quintal')

    df_trading = df_clean[df_clean['Is_Imputed'] == False].reset_index(drop=True) \
        if 'Is_Imputed' in df_clean.columns else df_clean

    from src.lstm_model import compute_price_ratios
    ratios = compute_price_ratios(df_trading, window=14)

    # Save Corrected Artifacts (as requested: do not overwrite old production model yet)
    model_path_corrected = os.path.join('models', 'corrected_3out_model.keras')
    scaler_path_corrected = os.path.join('models', 'corrected_scaler.pkl')

    res_news[lb_news]['model'].save(model_path_corrected)
    with open(scaler_path_corrected, 'wb') as fh:
        pickle.dump({
            'X_scaler': res_news[lb_news]['X_scaler'],
            'y_scaler': res_news[lb_news]['y_scaler'],
            'scaler': res_news[lb_news]['y_scaler'],
            'best_lookback': lb_news,
            'min_ratio': ratios['min_ratio'],
            'max_ratio': ratios['max_ratio'],
            'std_error': std_error,
            'features': NEWS_ENHANCED_FEATURES,
            'targets': ['Min_Price', 'Modal_Price', 'Max_Price'],
        }, fh)

    print(f'  -> Corrected 3-Output Model saved -> {model_path_corrected}')
    print(f'  -> Corrected Dual Scalers saved -> {scaler_path_corrected}')

    # Step 4: Demo Forecasts
    print('\n[STEP 4] Generating multi-step price forecasts (7 / 14 / 30 days)...')
    last_date = df_trading['Date'].max()
    scalers_dict = {
        'X_scaler': res_news[lb_news]['X_scaler'],
        'y_scaler': res_news[lb_news]['y_scaler'],
    }
    last_seq = scalers_dict['X_scaler'].transform(df_trading[NEWS_ENHANCED_FEATURES].values[-lb_news:])

    for horizon in [7, 14, 30]:
        df_fc = forecast_future_prices(
            res_news[lb_news]['model'], last_seq, scalers_dict,
            horizon_days=horizon, last_date=last_date,
            std_error=std_error,
        )
        print(f'\n  --- {horizon}-Day Forecast (first 5 rows) ---')
        print(df_fc[['Date', 'Raw_Modal_Price', 'Predicted_Average_Price', 'Modal_Lower_Bound', 'Modal_Upper_Bound']].head().to_string(index=False))

    print('\n' + sep)
    print('      PIPELINE COMPLETED SUCCESSFULLY')
    print(sep)

    return df_clean, res_news, lb_news


if __name__ == '__main__':
    try:
        run_pipeline()
    except KeyboardInterrupt:
        print('\nTraining interrupted by user.')
        sys.exit(0)
