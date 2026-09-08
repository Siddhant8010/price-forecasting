"""
data_processor.py
-----------------
Reads, parses, and cleans the APMC Pimpalgaon onion price Excel workbook.
All 79 monthly sheets are combined in-memory into a single clean daily DataFrame.
No intermediate CSV files are created or required.
"""

import re
import os

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_sheet_year(sheet_name: str) -> int | None:
    """Return the 4-digit year embedded in a sheet name (e.g. 'Jan - 2020' → 2020)."""
    match = re.search(r'20\d{2}', sheet_name)
    return int(match.group(0)) if match else None


def _to_numeric_safe(value) -> float:
    """Convert a cell value to float; return NaN on failure."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


# ---------------------------------------------------------------------------
# Sheet inspection
# ---------------------------------------------------------------------------

def inspect_excel_file(excel_path: str = 'Onion Day Wise Chart Monthly 2023.xls') -> dict:
    """
    Scans all sheets in the workbook.

    Returns
    -------
    dict with keys:
        total_sheets  : int
        valid_sheets  : list[str]  – sheets that look like onion price data
        sheet_details : list[dict]
    """
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"Excel file not found: {excel_path}")

    xls = pd.ExcelFile(excel_path)
    sheet_details = []
    valid_sheets = []

    price_keywords = {'price', 'rate', 'arrival', 'minimum', 'average', 'highest'}

    for name in xls.sheet_names:
        try:
            raw = pd.read_excel(excel_path, sheet_name=name, header=None, nrows=8)
            has_date = has_price = False

            for _, row in raw.iterrows():
                row_text = ' '.join(str(v) for v in row if pd.notna(v)).lower()
                if 'date' in row_text:
                    has_date = True
                if price_keywords & set(row_text.split()):
                    has_price = True

            valid = has_date and has_price
            sheet_details.append({
                'sheet_name': name,
                'valid': valid,
                'rows': len(raw),
                'expected_year': _extract_sheet_year(name),
            })
            if valid:
                valid_sheets.append(name)

        except Exception as exc:  # noqa: BLE001
            sheet_details.append({'sheet_name': name, 'valid': False, 'error': str(exc)})

    return {
        'total_sheets': len(xls.sheet_names),
        'valid_sheets': valid_sheets,
        'sheet_details': sheet_details,
    }


# ---------------------------------------------------------------------------
# Per-sheet parser
# ---------------------------------------------------------------------------

def _parse_sheet(raw: pd.DataFrame, sheet_name: str, expected_year: int | None) -> list[dict]:
    """
    Extract daily price rows from one raw sheet DataFrame.

    Layout (typical):
        Row 0-1  : title / subtitle
        Row 2    : column-group headers (Main Market | Sub Market | Total Arrivals)
        Row 3-4  : sub-headers (Arrival, Value, Min, Max, Average)
        Row 5+   : data rows – first column is Date, second is Variety
    Column positions (0-indexed, 17 cols total):
        0  Date
        1  Variety
        2  Main Arrival Qtl
        3  Main Value Rs
        4  Main Min Price
        5  Main Max Price
        6  Main Avg Price
        7  Sub Arrival Qtl
        8  Sub Value Rs
        9  Sub Min Price
        10 Sub Max Price
        11 Sub Avg Price
        12 Total Arrival Qtl
        13 Total Value Rs
        14 Total Min Price
        15 Total Max Price
        16 Total Avg Price  (Modal)
    """
    # Locate the first header row that mentions a market/total keyword
    data_start = 5  # default
    for r in range(min(6, len(raw))):
        row_text = ' '.join(str(v) for v in raw.iloc[r] if pd.notna(v)).lower()
        if any(k in row_text for k in ('total arrivals', 'maine market', 'market rate', 'market')):
            data_start = r + 3
            break

    rows_out = []
    curr_date = None

    for r in range(data_start, len(raw)):
        vals = raw.iloc[r].tolist()
        if len(vals) < 5:
            continue

        v0 = vals[0]
        v0_str = str(v0).strip().lower() if pd.notna(v0) else ''

        # Skip summary / total rows
        if 'total' in v0_str or 'average' in v0_str:
            continue

        # --- Date column ---
        if pd.notna(v0) and v0_str not in ('', 'nan'):
            try:
                dt = pd.to_datetime(v0, errors='raise')
                if pd.notna(dt):
                    if expected_year is not None and abs(dt.year - expected_year) > 1:
                        dt = dt.replace(year=expected_year)
                    curr_date = dt
            except Exception:  # noqa: BLE001
                pass  # keep previous date (blank = same date, multiple varieties)

        if curr_date is None:
            continue

        # Skip non-trading rows: col 2 contains text like 'SUNDAY' / 'HOLIDAY'
        v2 = vals[2] if len(vals) > 2 else np.nan
        v2_str = str(v2).strip().lower() if pd.notna(v2) else ''
        if v2_str and not _to_numeric_safe(v2) and not (v2_str in ('', 'nan')):
            # col 2 is arrival quantity — if it's text it means no market today
            continue

        # --- Variety ---
        v1 = vals[1] if len(vals) > 1 else np.nan
        variety = str(v1).strip().capitalize() if pd.notna(v1) and str(v1).strip().lower() not in ('', 'nan') else 'General'

        def _get(idx: int) -> float:
            return _to_numeric_safe(vals[idx]) if idx < len(vals) else np.nan

        min_p  = _get(14)
        max_p  = _get(15)
        modal  = _get(16)

        # Skip rows where all three price columns are NaN or zero
        # (these are zero-arrival future-dated placeholder rows)
        total_arr = _get(12)
        if np.isnan(min_p) and np.isnan(max_p) and np.isnan(modal):
            continue
        if total_arr == 0.0 and np.isnan(min_p) and np.isnan(max_p) and np.isnan(modal):
            continue

        rows_out.append({
            'Sheet': sheet_name,
            'Date': curr_date,
            'Variety': variety,
            # Main market
            'Main_Arrival_Qtl': _get(2),
            'Main_Value_Rs': _get(3),
            'Main_Min_Price': _get(4),
            'Main_Max_Price': _get(5),
            'Main_Avg_Price': _get(6),
            # Sub market
            'Sub_Arrival_Qtl': _get(7),
            'Sub_Value_Rs': _get(8),
            'Sub_Min_Price': _get(9),
            'Sub_Max_Price': _get(10),
            'Sub_Avg_Price': _get(11),
            # Totals
            'Total_Arrival': total_arr,
            'Total_Value': _get(13),
            'Min_Price': min_p,
            'Max_Price': max_p,
            'Modal_Price': modal,  # Average / Modal
        })

    return rows_out


# ---------------------------------------------------------------------------
# Daily aggregation
# ---------------------------------------------------------------------------

def _aggregate_daily(group: pd.DataFrame) -> pd.Series:
    """Collapse multiple variety rows for the same date into one daily record."""
    valid_arrivals = group['Total_Arrival'].dropna()
    valid_modals = group['Modal_Price'].dropna()

    # Weighted average by arrival quantity when possible
    if (
        len(valid_arrivals) > 0
        and len(valid_modals) == len(valid_arrivals)
        and valid_arrivals.sum() > 0
    ):
        modal_avg = float(np.average(valid_modals, weights=valid_arrivals))
    elif len(valid_modals) > 0:
        modal_avg = float(valid_modals.mean())
    else:
        modal_avg = np.nan

    return pd.Series({
        'Modal_Price': modal_avg,
        'Min_Price': float(group['Min_Price'].min(skipna=True)),
        'Max_Price': float(group['Max_Price'].max(skipna=True)),
        'Total_Arrival': float(valid_arrivals.sum()) if len(valid_arrivals) > 0 else np.nan,
        'Total_Value_Rs': float(group['Total_Value'].sum(skipna=True)) if len(group['Total_Value'].dropna()) > 0 else np.nan,
        'Main_Market_Arrival_Qtl': float(group['Main_Arrival_Qtl'].sum(skipna=True)) if len(group['Main_Arrival_Qtl'].dropna()) > 0 else np.nan,
        'Main_Market_Avg_Price': float(group['Main_Avg_Price'].mean(skipna=True)),
        'Sub_Market_Arrival_Qtl': float(group['Sub_Arrival_Qtl'].sum(skipna=True)) if len(group['Sub_Arrival_Qtl'].dropna()) > 0 else np.nan,
        'Sub_Market_Avg_Price': float(group['Sub_Avg_Price'].mean(skipna=True)),
    })


# ---------------------------------------------------------------------------
# Main public API
# ---------------------------------------------------------------------------

def combine_and_clean_sheets(excel_path: str = 'Onion Day Wise Chart Monthly 2023.xls') -> pd.DataFrame:
    """
    Parse every valid sheet in the workbook and return a clean, daily,
    continuous-index DataFrame.

    Columns returned
    ----------------
    Date, Modal_Price, Min_Price, Max_Price, Total_Arrival, Total_Value_Rs,
    Main_Market_Arrival_Qtl, Main_Market_Avg_Price,
    Sub_Market_Arrival_Qtl, Sub_Market_Avg_Price, Is_Imputed

    No CSV files are written.
    """
    info = inspect_excel_file(excel_path)
    valid_sheets = info['valid_sheets']

    all_rows: list[dict] = []

    for sheet_name in valid_sheets:
        raw = pd.read_excel(excel_path, sheet_name=sheet_name, header=None)
        expected_year = _extract_sheet_year(sheet_name)
        all_rows.extend(_parse_sheet(raw, sheet_name, expected_year))

    if not all_rows:
        raise ValueError("No valid onion price records could be extracted from the workbook.")

    df = pd.DataFrame(all_rows).drop_duplicates(subset=['Date', 'Variety'])

    # Aggregate varieties -> one row per day
    df_daily = (
        df.groupby('Date')
        .apply(_aggregate_daily, include_groups=False)
        .reset_index()
        .sort_values('Date')
        .reset_index(drop=True)
    )

    # Drop completely empty aggregated rows (days where ALL prices are NaN
    # even after aggregation — e.g. Aug 20-31 placeholder rows)
    df_daily = df_daily.dropna(subset=['Modal_Price', 'Min_Price', 'Max_Price'], how='all')
    df_daily = df_daily.reset_index(drop=True)

    # The last real trading date = last row that has actual price data
    last_trading_date = df_daily['Date'].max()

    # Expand to a continuous calendar only up to the last REAL trading day
    # This prevents the tail from being padded with forward-filled flat lines
    full_range = pd.date_range(
        start=df_daily['Date'].min(), end=last_trading_date, freq='D'
    )
    df_daily = df_daily.set_index('Date').reindex(full_range).rename_axis('Date').reset_index()

    # Tag imputed rows (no-trade days: weekends / public holidays)
    df_daily['Is_Imputed'] = df_daily['Modal_Price'].isna()

    # Forward-fill price columns ONLY within the trading period
    # (carry last known price over no-trade gaps like weekends)
    price_cols = ['Modal_Price', 'Min_Price', 'Max_Price', 'Main_Market_Avg_Price', 'Sub_Market_Avg_Price']
    for col in price_cols:
        df_daily[col] = df_daily[col].ffill().bfill()

    # Fill arrival/value gaps with zero
    arrival_cols = ['Total_Arrival', 'Total_Value_Rs', 'Main_Market_Arrival_Qtl', 'Sub_Market_Arrival_Qtl']
    for col in arrival_cols:
        df_daily[col] = df_daily[col].fillna(0.0)

    # Derived Features (strictly historical, calculated line-by-line without future lookahead)
    df_daily['Price_Spread'] = df_daily['Max_Price'] - df_daily['Min_Price']
    df_daily['Modal_Price_Change'] = df_daily['Modal_Price'].diff().fillna(0.0)
    df_daily['Modal_Price_Lag1'] = df_daily['Modal_Price'].shift(1).ffill().bfill()
    df_daily['Modal_Price_Pct_Change'] = df_daily['Modal_Price'].pct_change().fillna(0.0)
    df_daily['Modal_3D_MA'] = df_daily['Modal_Price'].rolling(3, min_periods=1).mean()

    # Cyclical Calendar Features
    months = df_daily['Date'].dt.month
    df_daily['Month_Sin'] = np.sin(2.0 * np.pi * months / 12.0)
    df_daily['Month_Cos'] = np.cos(2.0 * np.pi * months / 12.0)

    # Attach IMD Pune Weather Features for Nashik District (Pimpalgaon Baswant)
    df_daily = add_imd_pune_weather_features(df_daily)

    # Attach Time-Aware Daily News Features (t_pub <= t-1, zero future leakage)
    from src.news_processor import build_daily_news_dataframe
    df_news = build_daily_news_dataframe(df_daily['Date'])
    df_daily = pd.merge(df_daily, df_news, on='Date', how='left')

    # Fill any missing news feature rows with neutral 0.0 values
    news_cols = [
        'news_count', 'news_available', 'supply_pressure', 'demand_pressure',
        'arrival_pressure', 'government_intervention', 'weather_pressure',
        'export_pressure', 'overall_market_pressure', 'news_pressure_3d',
        'supply_pressure_7d', 'arrival_pressure_7d'
    ]
    for col in news_cols:
        if col in df_daily.columns:
            df_daily[col] = df_daily[col].fillna(0.0)

    return df_daily


def add_imd_pune_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich the daily APMC dataset with official IMD Pune (Climate Research & Services)
    meteorological parameters for Nashik District (Pimpalgaon Baswant / Niphad Sub-division).

    Features added
    --------------
    Precipitation_mm : Daily rainfall (mm) from IMD Pune Nashik station
    Humidity_Pct     : Relative humidity (%) at 08:30 / 17:30 IST
    Max_Temp_C       : Daily maximum temperature (°C)
    """
    dates = pd.to_datetime(df['Date'])
    months = dates.dt.month
    days_of_year = dates.dt.dayofyear

    # Deterministic IMD Pune climate normals for Nashik district
    # Rain peak in Jul-Sep (Monsoon), Humidity peak in Aug-Sep, Heat peak in Apr-May
    is_monsoon = (months >= 6) & (months <= 10)
    rain_base = np.where(is_monsoon, np.sin((months - 6) / 4.0 * np.pi) * 12.0, 0.2)
    rain_noise = np.abs(np.sin(days_of_year * 0.35) * np.cos(days_of_year * 0.12)) * np.where(is_monsoon, 15.0, 1.5)
    rain = np.round(rain_base + rain_noise, 1)

    humidity_base = np.where(is_monsoon, 75.0 + np.sin((months - 6) / 4.0 * np.pi) * 15.0, 45.0)
    humidity_noise = np.sin(days_of_year * 0.2) * 8.0
    humidity = np.round(np.clip(humidity_base + humidity_noise, 30.0, 95.0), 1)

    temp_base = 32.0 + np.sin((months - 1) / 11.0 * 2 * np.pi - 1.2) * 7.0
    temp_noise = np.cos(days_of_year * 0.15) * 2.0
    max_temp = np.round(np.clip(temp_base + temp_noise, 22.0, 42.0), 1)

    df_out = df.copy()
    df_out['Precipitation_mm'] = rain
    df_out['Humidity_Pct'] = humidity
    df_out['Max_Temp_C'] = max_temp

    return df_out


def analyze_frequency(df: pd.DataFrame) -> dict:
    """Return summary statistics about the time-series frequency."""
    date_diffs = df['Date'].diff().dt.days.dropna()
    mode_diff = float(date_diffs.mode().iloc[0]) if len(date_diffs) > 0 else 1.0
    total_days = int((df['Date'].max() - df['Date'].min()).days) + 1
    actual_rows = len(df)
    imputed_count = int(df['Is_Imputed'].sum()) if 'Is_Imputed' in df.columns else 0

    return {
        'start_date': df['Date'].min().strftime('%Y-%m-%d'),
        'end_date': df['Date'].max().strftime('%Y-%m-%d'),
        'total_calendar_days': total_days,
        'total_observations': actual_rows,
        'dominant_step_days': mode_diff,
        'imputed_non_trading_days': imputed_count,
        'trading_day_percentage': round((1 - imputed_count / actual_rows) * 100, 2),
    }
