"""
dataset_prep.py
---------------
Prepares chronological train / validation / test splits and sliding-window
LSTM sequences for the multi-target onion price forecasting model.

Target features (3 outputs):
    Index 0 → Min_Price
    Index 1 → Modal_Price  (average / modal)
    Index 2 → Max_Price
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# ---------------------------------------------------------------------------
# Feature Schemas: Model A (Price-Only) vs Model B (Price + News)
# ---------------------------------------------------------------------------

PRICE_FEATURES: list[str] = [
    'Min_Price', 'Modal_Price', 'Max_Price',
    'Modal_Price_Lag1', 'Modal_Price_Pct_Change', 'Modal_3D_MA',
    'Total_Arrival', 'Price_Spread', 'Modal_Price_Change',
    'Precipitation_mm', 'Humidity_Pct', 'Max_Temp_C',
]

NEWS_ENHANCED_FEATURES: list[str] = PRICE_FEATURES + [
    'news_count', 'news_available', 'supply_pressure', 'demand_pressure',
    'arrival_pressure', 'government_intervention', 'weather_pressure',
    'export_pressure', 'overall_market_pressure', 'news_pressure_3d',
    'supply_pressure_7d', 'arrival_pressure_7d',
]

TARGET_COLS: list[str] = ['Min_Price', 'Modal_Price', 'Max_Price']
DEFAULT_FEATURES: list[str] = NEWS_ENHANCED_FEATURES


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------

def split_dataset_chronologically(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,  # noqa: ARG001 – kept for API symmetry
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split *df* into three non-overlapping chronological partitions.
    No random shuffling – temporal order is strictly preserved.
    """
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# Scale (Decoupled X and Y Scalers)
# ---------------------------------------------------------------------------

def fit_and_scale_data(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str] = DEFAULT_FEATURES,
    target_cols: list[str] = TARGET_COLS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, MinMaxScaler, MinMaxScaler]:
    """
    Fit separate MinMaxScalers **strictly on training data** (zero data leakage):
      - X_scaler: fits on feature_cols (24 input features)
      - y_scaler: fits on target_cols (3 target price outputs)
    Then transforms validation and test splits.
    """
    X_scaler = MinMaxScaler(feature_range=(0, 1))
    y_scaler = MinMaxScaler(feature_range=(0, 1))

    train_X_scaled = X_scaler.fit_transform(train_df[feature_cols].values)
    val_X_scaled = X_scaler.transform(val_df[feature_cols].values)
    test_X_scaled = X_scaler.transform(test_df[feature_cols].values)

    train_y_scaled = y_scaler.fit_transform(train_df[target_cols].values)
    val_y_scaled = y_scaler.transform(val_df[target_cols].values)
    test_y_scaled = y_scaler.transform(test_df[target_cols].values)

    return (
        train_X_scaled, val_X_scaled, test_X_scaled,
        train_y_scaled, val_y_scaled, test_y_scaled,
        X_scaler, y_scaler,
    )


# ---------------------------------------------------------------------------
# Sequence creation (Decoupled X and y)
# ---------------------------------------------------------------------------

def create_sequences(
    X_array: np.ndarray,
    y_array: np.ndarray,
    lookback_window: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build sliding-window (X, y) pairs for LSTM training.

    Parameters
    ----------
    X_array        : 2-D array of shape (timesteps, n_features)
    y_array        : 2-D array of shape (timesteps, n_targets)
    lookback_window: number of past steps used as input

    Returns
    -------
    X : shape (samples, lookback_window, n_features)
    y : shape (samples, n_targets)
    """
    if X_array.ndim == 1:
        X_array = X_array.reshape(-1, 1)
    if y_array.ndim == 1:
        y_array = y_array.reshape(-1, 1)

    n_features = X_array.shape[1]
    n_targets = y_array.shape[1]
    X_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []

    for i in range(lookback_window, len(X_array)):
        X_list.append(X_array[i - lookback_window: i, :])
        y_list.append(y_array[i, :])

    if not X_list:
        empty_X = np.empty((0, lookback_window, n_features))
        empty_y = np.empty((0, n_targets))
        return empty_X, empty_y

    return np.array(X_list), np.array(y_list)


# ---------------------------------------------------------------------------
# Main pipeline wrapper
# ---------------------------------------------------------------------------

def prepare_lookback_dataset(
    df: pd.DataFrame,
    lookback_window: int,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    feature_cols: list[str] = DEFAULT_FEATURES,
    target_cols: list[str] = TARGET_COLS,
) -> dict:
    """
    End-to-end helper: split -> dual scale -> create sequences.

    Only actual trading days (Is_Imputed=False) are used for training,
    validation and test sequences so the model learns real market dynamics
    and is not distorted by weekend/holiday flat-line repetition.

    Returns a dict with keys:
        X_train, y_train, X_val, y_val, X_test, y_test
        X_scaler, y_scaler, scaler (alias for y_scaler), train_df, val_df, test_df
    """
    # Filter to actual trading days for model training
    if 'Is_Imputed' in df.columns:
        df_trading = df[df['Is_Imputed'] == False].copy().reset_index(drop=True)  # noqa: E712
    else:
        df_trading = df.copy()

    train_df, val_df, test_df = split_dataset_chronologically(
        df_trading, train_ratio, val_ratio, test_ratio
    )
    (
        train_X_scaled, val_X_scaled, test_X_scaled,
        train_y_scaled, val_y_scaled, test_y_scaled,
        X_scaler, y_scaler,
    ) = fit_and_scale_data(
        train_df, val_df, test_df, feature_cols=feature_cols, target_cols=target_cols
    )

    # Build sequences using overlap-preserving slices so that val/test windows
    # can look back into the preceding partition.
    full_X_scaled = X_scaler.transform(df_trading[feature_cols].values)
    full_y_scaled = y_scaler.transform(df_trading[target_cols].values)
    n_train = len(train_X_scaled)
    n_val = len(val_X_scaled)

    X_train, y_train = create_sequences(train_X_scaled, train_y_scaled, lookback_window)

    val_slice_X = full_X_scaled[n_train - lookback_window: n_train + n_val]
    val_slice_y = full_y_scaled[n_train - lookback_window: n_train + n_val]
    X_val, y_val = create_sequences(val_slice_X, val_slice_y, lookback_window)

    test_slice_X = full_X_scaled[n_train + n_val - lookback_window:]
    test_slice_y = full_y_scaled[n_train + n_val - lookback_window:]
    X_test, y_test = create_sequences(test_slice_X, test_slice_y, lookback_window)

    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test,
        'X_scaler': X_scaler,
        'y_scaler': y_scaler,
        'scaler': y_scaler,  # alias for backward compatibility
        'train_df': train_df,
        'val_df': val_df,
        'test_df': test_df,
    }
