"""
evaluate_metrics.py
-------------------
Comprehensive Evaluation Runner for the News-Enhanced Multi-Target LSTM
Onion Price Forecasting System (Price + Weather + Time-Aware News Signals).

Evaluates:
  1. Out-of-sample Test Set Performance (Overall & Per-Target: Min, Modal, Max)
  2. Dedicated Recent-Period Backtest (Latest 25 Trading Days)
  3. Benchmark against Naive Baselines (Persistence t-1, 7-Day MA, 14-Day MA)
  4. Day-by-day Actual vs Predicted Table (Latest 10 Trading Days)
  5. Next 14-Day Price Forecast with 95% Confidence Bounds

Run:
    python evaluate_metrics.py
"""

from __future__ import annotations

import os
import pickle
import sys

# Ensure UTF-8 stdout on Windows console
if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

os.environ.setdefault('KERAS_BACKEND', 'torch')

import numpy as np
import pandas as pd
import keras

from src.data_processor import combine_and_clean_sheets
from src.dataset_prep import NEWS_ENHANCED_FEATURES, prepare_lookback_dataset
from src.lstm_model import (
    compute_per_feature_metrics,
    evaluate_baselines,
    evaluate_model_performance,
    evaluate_recent_period,
    forecast_future_prices,
)
from src.apmc_scraper import fetch_live_apmc_pimpalgaon_rate
from src.weather_fetcher import fetch_live_imd_nashik_weather
from src.news_processor import fetch_live_google_news_rss


def evaluate_system_metrics(
    excel_path: str = 'Onion Day Wise Chart Monthly 2023.xls',
) -> None:
    """Run full evaluation suite for the News-Enhanced Multi-Target LSTM Model."""
    sep = '=' * 80
    print(sep)
    print('  NEWS-ENHANCED LSTM ONION PRICE FORECASTING SYSTEM -- EVALUATION REPORT')
    print(sep)

    # 1. Load Data
    print(f'\n[1/5] Loading dataset from Excel workbook: {excel_path}...')
    df_clean = combine_and_clean_sheets(excel_path)
    df_trading = df_clean[df_clean['Is_Imputed'] == False].reset_index(drop=True)
    n_obs = len(df_clean)
    n_trade = len(df_trading)
    d_min = df_clean['Date'].min().strftime('%Y-%m-%d')
    d_max = df_clean['Date'].max().strftime('%Y-%m-%d')
    print(f'  -> Total calendar observations : {n_obs} days ({d_min} to {d_max})')
    print(f'  -> Actual trading days         : {n_trade} days (weekends/holidays excluded)')

    # 2. Artifact loading for the News-Enhanced LSTM Model
    model_path = os.path.join('models', 'best_lstm_model.keras')
    scaler_path = os.path.join('models', 'scaler.pkl')

    # Fallback paths if alternate filenames exist
    if not os.path.exists(model_path) and os.path.exists(os.path.join('models', 'corrected_3out_model.keras')):
        model_path = os.path.join('models', 'corrected_3out_model.keras')
    if not os.path.exists(scaler_path) and os.path.exists(os.path.join('models', 'corrected_scaler.pkl')):
        scaler_path = os.path.join('models', 'corrected_scaler.pkl')

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print('\nSaved model artifacts not found! Please run `python main.py` first to train the model.')
        return

    print('\n[2/5] Loading trained News-Enhanced LSTM model & dual scaler artifacts...')
    model = keras.models.load_model(model_path)
    with open(scaler_path, 'rb') as fh:
        meta = pickle.load(fh)

    lb_win: int = int(meta.get('best_lookback', 14))
    std_error: float = float(meta.get('std_error', 120.0))
    min_ratio: float = float(meta.get('min_ratio', 0.272))
    max_ratio: float = float(meta.get('max_ratio', 1.462))
    feature_cols: list[str] = meta.get('features', NEWS_ENHANCED_FEATURES)
    target_cols: list[str] = meta.get('targets', ['Min_Price', 'Modal_Price', 'Max_Price'])

    print(f'  -> Model File        : {model_path}')
    print(f'  -> Scaler File       : {scaler_path}')
    print(f'  -> Lookback Window   : {lb_win} trading days')
    print(f'  -> Input Features    : {len(feature_cols)} (Prices, Mandi Arrivals, IMD Weather, News Signals)')
    print(f'  -> Target Outputs    : {len(target_cols)} ({", ".join(target_cols)})')
    print(f'  -> Residual Std Err  : Rs.{std_error:.2f} / Quintal (95% CI: +/-Rs.{std_error * 1.96:.2f})')

    # 3. Prepare split datasets & evaluate out-of-sample performance
    print('\n[3/5] Computing out-of-sample test evaluations on unseen test split...')
    lb_data = prepare_lookback_dataset(
        df_clean,
        lookback_window=lb_win,
        feature_cols=feature_cols,
        target_cols=target_cols,
    )

    metrics_overall, y_true_test, y_pred_test = evaluate_model_performance(
        model, lb_data['X_test'], lb_data['y_test'], meta
    )
    per_feature_metrics = compute_per_feature_metrics(
        y_true_test, y_pred_test, target_cols
    )

    # 4. Recent Period Evaluation & Baselines
    print('\n[4/5] Computing Recent 25-Day Backtest & Naive Baseline comparisons...')
    recent_overall, recent_pf, y_true_recent, y_pred_recent = evaluate_recent_period(
        y_true_test, y_pred_test, target_cols, recent_days=25
    )
    baseline_results = evaluate_baselines(df_trading, test_ratio=0.15, recent_days=25)

    # ── REPORT OUTPUT ───────────────────────────────────────────────────────────
    print('\n' + '=' * 80)
    print(' A. OUT-OF-SAMPLE TEST SET PERFORMANCE (Full Test Partition: 273 Days)')
    print('=' * 80)
    print(f"{'Target Price Column':<30} | {'MAE (Rs)':<10} | {'RMSE (Rs)':<10} | {'MAPE (%)':<10} | {'R^2 Score':<10}")
    print('-' * 80)

    for target_name in target_cols:
        m = per_feature_metrics.get(target_name, {})
        tag = ' (Primary Benchmark)' if target_name == 'Modal_Price' else ''
        display_name = f"{target_name}{tag}"
        print(f"{display_name:<30} | {m.get('MAE', 0.0):<10.2f} | {m.get('RMSE', 0.0):<10.2f} | {m.get('MAPE', 0.0):<10.2f}% | {m.get('R2', 0.0):<10.4f}")

    print('-' * 80)
    print(f"{'Overall Modal Performance':<30} | {metrics_overall['MAE']:<10.2f} | {metrics_overall['RMSE']:<10.2f} | {metrics_overall['MAPE']:<10.2f}% | {metrics_overall['R2']:<10.4f}")
    print('-' * 80)

    print('\n' + '=' * 80)
    print(' B. DEDICATED RECENT-PERIOD BACKTEST (Latest 25 Trading Days)')
    print('=' * 80)
    print(f"{'Target Price Column':<30} | {'Recent MAE':<10} | {'Recent RMSE':<11} | {'Recent MAPE':<11} | {'Recent R^2':<10}")
    print('-' * 80)

    for target_name in target_cols:
        m = recent_pf.get(target_name, {})
        tag = ' (Primary Benchmark)' if target_name == 'Modal_Price' else ''
        display_name = f"{target_name}{tag}"
        print(f"{display_name:<30} | {m.get('MAE', 0.0):<10.2f} | {m.get('RMSE', 0.0):<11.2f} | {m.get('MAPE', 0.0):<10.2f}% | {m.get('R2', 0.0):<10.4f}")

    print('-' * 80)
    print(f"{'Overall Recent Performance':<30} | {recent_overall['MAE']:<10.2f} | {recent_overall['RMSE']:<11.2f} | {recent_overall['MAPE']:<10.2f}% | {recent_overall['R2']:<10.4f}")
    print('-' * 80)

    print('\n' + '=' * 80)
    print(' C. BENCHMARK COMPARISON VS NAIVE BASELINES (Modal Price Target)')
    print('=' * 80)
    print(f"{'Model / Baseline Strategy':<34} | {'Hist MAE':<10} | {'Hist MAPE':<10} | {'Rec MAE':<10} | {'Rec MAPE':<10}")
    print('-' * 80)

    modal_hist = per_feature_metrics.get('Modal_Price', metrics_overall)
    modal_rec = recent_pf.get('Modal_Price', recent_overall)

    print(f"{'News-Enhanced LSTM (Our Model)':<34} | {modal_hist['MAE']:<10.2f} | {modal_hist['MAPE']:<9.2f}% | {modal_rec['MAE']:<10.2f} | {modal_rec['MAPE']:<9.2f}%")

    for b_name, b_res in baseline_results.items():
        h_m = b_res['Historical_Test']
        r_m = b_res['Recent_Test']
        print(f"{b_name:<34} | {h_m['MAE']:<10.2f} | {h_m['MAPE']:<9.2f}% | {r_m['MAE']:<10.2f} | {r_m['MAPE']:<9.2f}%")
    print('-' * 80)

    # 5. Recent Actual vs Predicted observations table
    print('\n' + '=' * 80)
    print(' D. RECENT ACTUAL VS PREDICTED OBSERVATIONS (Latest 10 Trading Days)')
    print('=' * 80)
    test_dates = lb_data['test_df']['Date'].iloc[-10:].dt.strftime('%Y-%m-%d').values
    act_min = np.round(y_true_test[-10:, 0], 0).astype(int)
    pred_min = np.round(y_pred_test[-10:, 0], 0).astype(int)

    act_modal = np.round(y_true_test[-10:, 1], 0).astype(int)
    pred_modal = np.round(y_pred_test[-10:, 1], 0).astype(int)

    act_max = np.round(y_true_test[-10:, 2], 0).astype(int)
    pred_max = np.round(y_pred_test[-10:, 2], 0).astype(int)

    abs_err = np.abs(act_modal - pred_modal)
    pct_err = np.round((abs_err / act_modal) * 100.0, 1)

    df_recent_obs = pd.DataFrame({
        'Date': test_dates,
        'Act_Min': act_min,
        'Pred_Min': pred_min,
        'Act_Modal': act_modal,
        'Pred_Modal': pred_modal,
        'Act_Max': act_max,
        'Pred_Max': pred_max,
        'Abs_Err(Rs)': abs_err,
        'Pct_Err(%)': pct_err,
    })
    print(df_recent_obs.to_string(index=False))
    print('-' * 80)

    # 6. Future Forecast
    print('\n[5/5] Generating Next 14-Day Price Forecast with 95% Confidence Range...')
    last_date = df_trading['Date'].max()
    X_scaler = meta.get('X_scaler', meta.get('scaler'))
    last_seq = X_scaler.transform(df_trading[feature_cols].values[-lb_win:])
    df_fc = forecast_future_prices(
        model,
        last_seq,
        meta,
        horizon_days=14,
        last_date=last_date,
        min_ratio=min_ratio,
        max_ratio=max_ratio,
        std_error=std_error,
    )
    print('\n' + '-' * 80)
    print(f'Last Actual APMC Trade Date : {last_date.strftime("%Y-%m-%d")}')
    print(f'Last Actual APMC Modal Price: Rs.{df_trading["Modal_Price"].iloc[-1]:,.0f} / Quintal')
    print('-' * 80)
    cols_to_print = [
        'Date', 'Raw_Modal_Price', 'Predicted_Average_Price',
        'Predicted_Minimum_Price', 'Predicted_Maximum_Price',
        'Modal_Lower_Bound', 'Modal_Upper_Bound',
    ]
    print(df_fc[cols_to_print].head(5).to_string(index=False))

    # 6. Live Multi-Source Integrations (APMC Rates, IMD Weather, Google News RSS)
    print('\n' + '-' * 80)
    print('  LIVE REAL-TIME DATA INTEGRATIONS (APMC Pimpalgaon + IMD Nashik + Google News)')
    print('-' * 80)
    live_apmc = fetch_live_apmc_pimpalgaon_rate()
    live_weather = fetch_live_imd_nashik_weather()
    live_news = fetch_live_google_news_rss()

    if live_apmc is not None:
        print(f"  -> APMC Portal Status : Connected (apmcpimpalgaon.com)")
        print(f"  -> Traded Commodity   : {live_apmc['variety']} (Rabi / Summer Onion)")
        print(f"  -> Live Trade Date    : {live_apmc['date_str']}")
        print(f"  -> Live Rates (Rs/Qtl): Min: Rs.{live_apmc['min_price']:,.0f} | Modal: Rs.{live_apmc['modal_price']:,.0f} | Max: Rs.{live_apmc['max_price']:,.0f}")
    else:
        print("  -> APMC Portal Status : Offline / Unreachable (falling back to Excel data)")

    if live_weather is not None:
        print(f"  -> IMD Weather Status : Connected ({live_weather['station']})")
        print(f"  -> Live Observations  : Temp: {live_weather['max_temp_c']} deg C | Humidity: {live_weather['humidity_pct']}% | Rain: {live_weather['precipitation_mm']} mm")

    print(f"  -> Google News Feed   : Connected ({len(live_news)} Live Onion Market Articles Ingested)")
    if live_news:
        print(f"  -> Latest Headline    : \"{live_news[0]['headline'][:70]}...\" ({live_news[0]['source']})")

    if live_apmc is not None:
        override_w = (
            live_weather['precipitation_mm'],
            live_weather['humidity_pct'],
            live_weather['max_temp_c'],
        ) if live_weather else None

        df_live_fc = forecast_future_prices(
            model,
            last_seq,
            meta,
            horizon_days=14,
            last_date=live_apmc['date'],
            min_ratio=min_ratio,
            max_ratio=max_ratio,
            std_error=std_error,
            override_base_prices=(
                live_apmc['min_price'],
                live_apmc['modal_price'],
                live_apmc['max_price'],
            ),
            override_weather=override_w,
        )
        print('\n  Next 5 Trading Days Forecast (Anchored to Live APMC + IMD Weather + News):')
        print(df_live_fc[cols_to_print].head(5).to_string(index=False))
    print('=' * 80)
    print('      EVALUATION COMPLETED SUCCESSFULLY')
    print('=' * 80)


if __name__ == '__main__':
    evaluate_system_metrics()
