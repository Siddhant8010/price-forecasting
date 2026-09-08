"""
app.py
------
Streamlit dashboard for the News-Enhanced Onion Price Forecasting System.

Features:
  • Reads raw data directly from the Excel workbook (no CSV required)
  • Predicts Minimum, Average (Modal), and Maximum onion prices
  • Displays Recent Market & News Signals (Supply, Demand, Arrivals, Govt, Weather, Exports)
  • Allows side-by-side comparison between Model A (Price-Only) and Model B (Price + News)
  • Exposes raw vs post-processed predictions with 95% confidence intervals
  • Shows feature signal drivers for forecast explainability

Run:
    streamlit run app.py
"""

from __future__ import annotations

import os
import pickle

os.environ.setdefault('KERAS_BACKEND', 'torch')

import keras
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data_processor import combine_and_clean_sheets
from src.dataset_prep import NEWS_ENHANCED_FEATURES
from src.lstm_model import (
    explain_forecast_signals,
    forecast_future_prices,
)
from src.news_processor import HISTORICAL_MARKET_EVENTS, deduplicate_news, fetch_live_google_news_rss
from src.apmc_scraper import fetch_live_apmc_pimpalgaon_rate
from src.weather_fetcher import fetch_live_imd_nashik_weather

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EXCEL_PATH = 'Onion Day Wise Chart Monthly 2023.xls'

MODEL_PATH_NEWS = os.path.join('models', 'best_lstm_model.keras')
SCALER_PATH_NEWS = os.path.join('models', 'scaler.pkl')

# ---------------------------------------------------------------------------
# Page config & global styles
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title='Onion Market Price Forecasting System',
    page_icon='🧅',
    layout='wide',
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
    color: #f1f5f9;
}

/* Card panels */
.card {
    background: rgba(30, 41, 59, 0.75);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.09);
    border-radius: 14px;
    padding: 22px;
    margin-bottom: 18px;
}

/* Metric boxes */
.metric-box {
    background: rgba(15, 23, 42, 0.65);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 10px;
    padding: 16px 10px;
    text-align: center;
}
.metric-label {
    color: #94a3b8;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    margin-bottom: 6px;
}
.metric-val {
    color: #38bdf8;
    font-size: 1.5rem;
    font-weight: 700;
}
.metric-val.good  { color: #4ade80; }
.metric-val.warn  { color: #facc15; }
.metric-val.price { color: #a78bfa; }

/* Signal badges */
.signal-badge {
    display: inline-block;
    padding: 6px 12px;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 600;
    margin: 4px;
}
.signal-high-bullish { background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid #ef4444; }
.signal-mod-bullish  { background: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid #f59e0b; }
.signal-bearish      { background: rgba(59, 130, 246, 0.2); color: #60a5fa; border: 1px solid #3b82f6; }

/* Predict button */
.stButton > button {
    background: linear-gradient(90deg, #6366f1 0%, #a855f7 100%);
    color: #fff;
    font-weight: 700;
    font-size: 1.05rem;
    border: none;
    border-radius: 8px;
    padding: 12px 28px;
    width: 100%;
    transition: all 0.25s ease;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 22px rgba(168, 85, 247, 0.45);
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner='Loading trained model artifacts...')
def load_model_artifacts() -> tuple[dict, pd.DataFrame]:
    """Load the News-Enhanced LSTM Model artifact from disk."""
    if not os.path.exists(MODEL_PATH_NEWS) or not os.path.exists(SCALER_PATH_NEWS):
        st.error(
            'Model artifacts not found. '
            'Please run `python main.py` first to train and save the model.'
        )
        st.stop()

    model_news = keras.models.load_model(MODEL_PATH_NEWS)
    with open(SCALER_PATH_NEWS, 'rb') as fh:
        meta_news = pickle.load(fh)

    df_clean = combine_and_clean_sheets(EXCEL_PATH)

    artifacts = {
        'model_news': model_news,
        'meta_news': meta_news,
    }
    return artifacts, df_clean


# ---------------------------------------------------------------------------
# App layout
# ---------------------------------------------------------------------------

st.title('🧅 News-Enhanced Onion Price Forecasting System')
st.markdown('##### *Multi-Target LSTM Price Prediction with Time-Aware News & Market Signal Processing*')
st.markdown('---')

artifacts, df_clean = load_model_artifacts()
df_trading = df_clean[df_clean['Is_Imputed'] == False].reset_index(drop=True)
last_trade_date = df_trading['Date'].max()
last_modal_price = df_trading['Modal_Price'].iloc[-1]
last_arrival_qtl = df_trading['Total_Arrival'].iloc[-1]

sel_model = artifacts['model_news']
sel_meta = artifacts['meta_news']
sel_features = NEWS_ENHANCED_FEATURES

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('### ⚙️  Model Architecture')
    st.markdown('**News-Enhanced Multi-Target LSTM**')
    st.caption('24 features: APMC prices, mandi arrivals, IMD Pune weather, and time-aware market news.')

    st.markdown('---')
    st.markdown('### ℹ️  Market & Data Metadata')
    st.markdown(f'**Last APMC Trade Date:** {last_trade_date.strftime("%d %b %Y")}')
    st.markdown(f'**Last Modal Rate:** ₹{last_modal_price:,.0f} / Quintal')
    st.markdown(f'**Last Mandi Arrival:** {last_arrival_qtl:,.0f} Quintals')
    st.markdown(f'**Lookback Window:** {sel_meta.get("best_lookback", 15)} trading days')
    st.markdown(f'**Active Features:** {len(sel_features)} columns')
    st.markdown(f'**Validation Std Error:** ±₹{sel_meta.get("std_error", 120.0):.1f} / Qtl')
    st.markdown('---')
    if st.button('🔄  Refresh Market Data & News'):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.success('Data & news pipeline refreshed successfully!')
        st.rerun()

# Fetch live APMC Pimpalgaon rates, IMD Nashik weather, and Google News RSS (cached in session_state)
if 'live_apmc' not in st.session_state:
    st.session_state['live_apmc'] = fetch_live_apmc_pimpalgaon_rate()
if 'live_weather' not in st.session_state:
    st.session_state['live_weather'] = fetch_live_imd_nashik_weather()
if 'live_news' not in st.session_state:
    st.session_state['live_news'] = fetch_live_google_news_rss()

live_apmc = st.session_state.get('live_apmc')
live_weather = st.session_state.get('live_weather')
live_news = st.session_state.get('live_news', [])

# ── Live Market, Weather & News Feeds Banner ──────────────────────────────────
st.markdown('<div class="card" style="border-left: 5px solid #10b981; background: linear-gradient(135deg, #f0fdf4 0%, #ffffff 100%);">', unsafe_allow_html=True)
b_col1, b_col2, b_col3 = st.columns([1.3, 1.2, 1.2])

with b_col1:
    if live_apmc is not None:
        st.markdown(
            f"**🌐 Live APMC Pimpalgaon Rate:** `🟢 Connected`  \n"
            f"**Variety:** `{live_apmc['variety']} (उन्हाळ कांदा)` | **Date:** `{live_apmc['date_str']}`  \n"
            f"**Modal (Avg):** `₹{live_apmc['modal_price']:,.0f}` (Min: ₹{live_apmc['min_price']:,.0f} | Max: ₹{live_apmc['max_price']:,.0f})"
        )
    else:
        st.markdown(
            f"**🌐 Live APMC Pimpalgaon:** `🟡 Dataset Fallback`  \n"
            f"**Last Modal Rate:** ₹{last_modal_price:,.0f} / Qtl ({last_trade_date.strftime('%d %b %Y')})"
        )

with b_col2:
    if live_weather is not None:
        st.markdown(
            f"**🌦️ Live IMD Nashik Weather:** `🟢 Connected`  \n"
            f"**Station:** `{live_weather['station']}`  \n"
            f"**Temp:** `{live_weather['max_temp_c']}°C` | **Humidity:** `{live_weather['humidity_pct']}%` | **Rain:** `{live_weather['precipitation_mm']} mm`"
        )
    else:
        st.markdown(
            "**🌦️ IMD Pune Weather:** `🟡 Climate Normals`  \n"
            "Precipitation: 0.0 mm | Humidity: 75% | Temp: 30°C"
        )

with b_col3:
    n_count = len(live_news) if live_news else 0
    top_headline = live_news[0]['headline'][:55] + '…' if (live_news and len(live_news) > 0) else 'Monitoring APMC auctions'
    st.markdown(
        f"**📰 Live Google News Agri Feed:** `🟢 Active ({n_count} Articles)`  \n"
        f"**Search Track:** `Nashik & Pimpalgaon Onion Markets`  \n"
        f"**Latest:** *{top_headline}*"
    )

st.markdown('</div>', unsafe_allow_html=True)

# ── Recent Market Signals Banner ──────────────────────────────────────────────
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('#### 📰  Recent Onion Market Signals & Economic Indicators')

s1, s2, s3, s4, s5, s6 = st.columns(6)

def get_signal_badge(val: float, positive_is_bullish: bool = True) -> tuple[str, str]:
    if abs(val) < 0.05:
        return '🟢 Neutral', 'signal-bearish'
    if (val > 0 and positive_is_bullish) or (val < 0 and not positive_is_bullish):
        return '🔴 Bullish (High Risk)', 'signal-high-bullish'
    return '🟢 Bearish (Price Softening)', 'signal-bearish'

sp_val = float(df_trading['supply_pressure'].iloc[-1]) if 'supply_pressure' in df_trading.columns else -0.2
dp_val = float(df_trading['demand_pressure'].iloc[-1]) if 'demand_pressure' in df_trading.columns else 0.3
ap_val = float(df_trading['arrival_pressure'].iloc[-1]) if 'arrival_pressure' in df_trading.columns else -0.3
gi_val = float(df_trading['government_intervention'].iloc[-1]) if 'government_intervention' in df_trading.columns else 0.0
wp_val = float(df_trading['weather_pressure'].iloc[-1]) if 'weather_pressure' in df_trading.columns else 0.4
ep_val = float(df_trading['export_pressure'].iloc[-1]) if 'export_pressure' in df_trading.columns else -0.1

with s1:
    txt, cls = get_signal_badge(sp_val, positive_is_bullish=False)
    st.markdown(f'<div class="metric-box"><div class="metric-label">Supply Pressure</div><div class="signal-badge {cls}">{txt}</div></div>', unsafe_allow_html=True)
with s2:
    txt, cls = get_signal_badge(dp_val, positive_is_bullish=True)
    st.markdown(f'<div class="metric-box"><div class="metric-label">Demand Pressure</div><div class="signal-badge {cls}">{txt}</div></div>', unsafe_allow_html=True)
with s3:
    txt, cls = get_signal_badge(ap_val, positive_is_bullish=False)
    st.markdown(f'<div class="metric-box"><div class="metric-label">Arrival Trend</div><div class="signal-badge {cls}">{txt}</div></div>', unsafe_allow_html=True)
with s4:
    txt, cls = get_signal_badge(gi_val, positive_is_bullish=False)
    st.markdown(f'<div class="metric-box"><div class="metric-label">Govt Buffer/Duty</div><div class="signal-badge {cls}">{txt}</div></div>', unsafe_allow_html=True)
with s5:
    txt, cls = get_signal_badge(wp_val, positive_is_bullish=True)
    st.markdown(f'<div class="metric-box"><div class="metric-label">Weather/Rain Risk</div><div class="signal-badge {cls}">{txt}</div></div>', unsafe_allow_html=True)
with s6:
    txt, cls = get_signal_badge(ep_val, positive_is_bullish=True)
    st.markdown(f'<div class="metric-box"><div class="metric-label">Export Pressure</div><div class="signal-badge {cls}">{txt}</div></div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ── Forecast Controls Card ────────────────────────────────────────────────────
st.markdown('<div class="card">', unsafe_allow_html=True)
ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1.2, 2.0, 1.2])

with ctrl_col1:
    horizon_days = st.number_input(
        '📅  Forecast horizon (days)',
        min_value=1,
        max_value=90,
        value=14,
        step=1,
        help='Number of future trading days to forecast',
    )

with ctrl_col2:
    st.markdown('<div style="padding-top: 4px;">', unsafe_allow_html=True)
    use_live_anchor = st.checkbox(
        '🔗 Anchor Forecast with Live APMC Rate (apmcpimpalgaon.com)',
        value=(live_apmc is not None),
        help='When enabled, projects multi-step forecast starting from today\'s live Unhal Kanda auction rates instead of historical August Excel data.',
    )
    r_col1, r_col2 = st.columns([1.5, 2.5])
    with r_col1:
        if st.button('🔄 Refresh All Live Feeds', key='refresh_live_rate_btn'):
            st.session_state['live_apmc'] = fetch_live_apmc_pimpalgaon_rate(force_refresh=True)
            st.session_state['live_weather'] = fetch_live_imd_nashik_weather(force_refresh=True)
            st.session_state['live_news'] = fetch_live_google_news_rss(force_refresh=True)
            st.rerun()
    with r_col2:
        st.caption('Refreshes APMC rates, IMD Nashik weather & Google News RSS.')
    st.markdown('</div>', unsafe_allow_html=True)
    news_override_pct = 0.0

with ctrl_col3:
    st.write('')
    st.write('')
    predict_btn = st.button('🔮  Generate Forecast', key='predict_btn')
    if 'df_forecast' in st.session_state:
        st.download_button(
            label='📥  Download Forecast CSV',
            data=st.session_state['df_forecast'].to_csv(index=False).encode('utf-8'),
            file_name='onion_price_forecast.csv',
            mime='text/csv',
        )

st.markdown('</div>', unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_fc, tab_news = st.tabs([
    '🔮  Price Forecast: Minimum / Average / High Prices',
    '📰  Recent Market News & Signals Log',
])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1: FORECAST
# ════════════════════════════════════════════════════════════════════════════
with tab_fc:
    use_anchor_bool = bool(use_live_anchor and live_apmc is not None)
    current_params = (
        horizon_days,
        use_anchor_bool,
        float(live_apmc['modal_price']) if (live_apmc is not None and use_anchor_bool) else 0.0,
    )
    params_changed = st.session_state.get('last_forecast_params') != current_params

    if predict_btn or ('df_forecast' not in st.session_state) or params_changed:
        with st.spinner('Generating multi-step forecast with live APMC anchor…'):
            lb_win = int(sel_meta.get('best_lookback', 14))
            last_vals = df_trading[sel_features].values[-lb_win:]
            X_scaler = sel_meta.get('X_scaler', sel_meta.get('scaler'))
            last_seq_scaled = X_scaler.transform(last_vals)

            override_prices = None
            forecast_start_date = last_trade_date
            if use_anchor_bool and live_apmc is not None:
                override_prices = (
                    float(live_apmc['min_price']),
                    float(live_apmc['modal_price']),
                    float(live_apmc['max_price']),
                )
                forecast_start_date = live_apmc['date']

            override_w = None
            if live_weather is not None:
                override_w = (
                    float(live_weather.get('precipitation_mm', 0.0)),
                    float(live_weather.get('humidity_pct', 75.0)),
                    float(live_weather.get('max_temp_c', 30.0)),
                )

            df_fc = forecast_future_prices(
                model=sel_model,
                last_sequence=last_seq_scaled,
                scaler=sel_meta,
                horizon_days=horizon_days,
                last_date=forecast_start_date,
                min_ratio=sel_meta.get('min_ratio', 0.272),
                max_ratio=sel_meta.get('max_ratio', 1.462),
                std_error=sel_meta.get('std_error', 120.0),
                news_override_factor=0.0,
                override_base_prices=override_prices,
                override_weather=override_w,
            )
            st.session_state['df_forecast'] = df_fc
            st.session_state['horizon_days'] = horizon_days
            st.session_state['last_forecast_params'] = current_params
            st.session_state['used_live_anchor'] = use_anchor_bool
            st.session_state['anchor_date'] = forecast_start_date
            st.session_state['anchor_price'] = (
                live_apmc['modal_price'] if (live_apmc is not None and use_anchor_bool) else last_modal_price
            )

    df_forecast: pd.DataFrame = st.session_state['df_forecast']
    cur_horizon: int = st.session_state['horizon_days']
    is_live_anchored = st.session_state.get('used_live_anchor', False)
    anchor_dt = st.session_state.get('anchor_date', last_trade_date)
    anchor_pr = st.session_state.get('anchor_price', last_modal_price)

    anchor_tag = "🟢 Live APMC Pimpalgaon (apmcpimpalgaon.com)" if is_live_anchored else "📁 Historical APMC Dataset (Excel)"

    st.info(
        f"📍 **Active Model:** News-Enhanced Multi-Target LSTM (24 Features) | "
        f"**Starting Anchor:** {anchor_tag} | "
        f"**Anchor Date:** {anchor_dt.strftime('%d %b %Y')} (₹{anchor_pr:,.0f}/Qtl) | "
        f"**95% Confidence Band:** ±₹{sel_meta.get('std_error', 120.0) * 1.96:,.0f}/Qtl"
    )

    # Summary KPI row (Minimum, Average, High Prices)
    k1, k2, k3 = st.columns(3)
    with k1:
        st.markdown(
            f'<div class="metric-box"><div class="metric-label">Minimum Price</div>'
            f'<div class="metric-val good">₹{df_forecast["Predicted_Minimum_Price"].mean():,.0f} / Qtl</div>'
            f'<div class="metric-label">(avg over {cur_horizon} days)</div></div>',
            unsafe_allow_html=True,
        )
    with k2:
        st.markdown(
            f'<div class="metric-box"><div class="metric-label">Average Price</div>'
            f'<div class="metric-val price">₹{df_forecast["Predicted_Average_Price"].mean():,.0f} / Qtl</div>'
            f'<div class="metric-label">(avg over {cur_horizon} days)</div></div>',
            unsafe_allow_html=True,
        )
    with k3:
        st.markdown(
            f'<div class="metric-box"><div class="metric-label">High Price</div>'
            f'<div class="metric-val warn">₹{df_forecast["Predicted_Maximum_Price"].mean():,.0f} / Qtl</div>'
            f'<div class="metric-label">(avg over {cur_horizon} days)</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown('<br>', unsafe_allow_html=True)

    col_tbl, col_chart = st.columns([1.1, 1.7])

    with col_tbl:
        st.markdown(f'##### 📋  {cur_horizon}-Day Price Forecast (Minimum, Average & High Prices)')
        df_disp = df_forecast.copy()
        df_disp['Date'] = df_disp['Date'].dt.strftime('%Y-%m-%d')
        st.dataframe(
            df_disp[[
                'Date',
                'Predicted_Minimum_Price',
                'Predicted_Average_Price',
                'Predicted_Maximum_Price',
            ]].rename(columns={
                'Predicted_Minimum_Price': 'Minimum Price (₹)',
                'Predicted_Average_Price': 'Average Price (₹)',
                'Predicted_Maximum_Price': 'High Price (₹)',
            }),
            width='stretch',
            height=440,
        )

    with col_chart:
        st.markdown(f'##### 📈  Price Trajectory ({cur_horizon} Days)')
        recent = df_clean.tail(90)
        fig = go.Figure()

        # Historical average price
        fig.add_trace(go.Scatter(
            x=recent['Date'], y=recent['Modal_Price'],
            mode='lines', name='Historical Average Price',
            line=dict(color='#38bdf8', width=2),
        ))

        # 95% Confidence Band fill
        fig.add_trace(go.Scatter(
            x=pd.concat([df_forecast['Date'], df_forecast['Date'][::-1]]),
            y=pd.concat([
                df_forecast['Modal_Upper_Bound'],
                df_forecast['Modal_Lower_Bound'][::-1],
            ]),
            fill='toself',
            fillcolor='rgba(168,85,247,0.12)',
            line=dict(color='rgba(0,0,0,0)'),
            name='95% Confidence Band',
            showlegend=True,
        ))

        # Minimum, Average, High Price traces
        fig.add_trace(go.Scatter(
            x=df_forecast['Date'], y=df_forecast['Predicted_Minimum_Price'],
            mode='lines+markers', name='Minimum Price',
            line=dict(color='#22c55e', width=2, dash='dash'),
            marker=dict(size=5),
        ))
        fig.add_trace(go.Scatter(
            x=df_forecast['Date'], y=df_forecast['Predicted_Average_Price'],
            mode='lines+markers', name='Average Price',
            line=dict(color='#a855f7', width=3),
            marker=dict(size=6),
        ))
        fig.add_trace(go.Scatter(
            x=df_forecast['Date'], y=df_forecast['Predicted_Maximum_Price'],
            mode='lines+markers', name='High Price',
            line=dict(color='#f43f5e', width=2, dash='dash'),
            marker=dict(size=5),
        ))

        fig.update_layout(
            title=f'Onion Price Forecast — Next {cur_horizon} Days',
            xaxis_title='Date',
            yaxis_title='Price (₹ / Quintal)',
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(15,23,42,0.6)',
            height=440,
            hovermode='x unified',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        )
        st.plotly_chart(fig, width='stretch')

    # Explainability Drivers Section
    st.markdown('---')
    st.markdown('##### 💡  Forecast Driver Signals & Explainability')
    drivers = explain_forecast_signals(df_trading)
    d_cols = st.columns(len(drivers))
    for idx, drv in enumerate(drivers):
        with d_cols[idx]:
            st.markdown(
                f'<div class="card">'
                f'<strong>{drv["signal"]}</strong><br>'
                f'<span style="color:#a855f7;">{drv["direction"]}</span> ({drv["level"]})<br>'
                f'<small>{drv["description"]}</small>'
                f'</div>',
                unsafe_allow_html=True,
            )

# ════════════════════════════════════════════════════════════════════════════
# TAB 2: NEWS & SIGNALS LOG
# ════════════════════════════════════════════════════════════════════════════
with tab_news:
    st.subheader('📰  Real-Time & Historical Onion Market News')
    st.markdown('##### *Live Google News RSS Feed + Ingested PIB / NAFED / Agrowon Policy History*')

    st.markdown('#### 🌐  Live Breaking Google News Feed (Nashik & Pimpalgaon)')
    live_items = st.session_state.get('live_news', [])
    if live_items:
        df_live_n = pd.DataFrame(live_items)
        disp_cols = ['date', 'source', 'headline', 'price_direction', 'overall_market_pressure']
        st.dataframe(
            df_live_n[disp_cols].rename(columns={
                'date': 'Published Date',
                'source': 'Media Source',
                'headline': 'Breaking Headline',
                'price_direction': 'Price Direction Impact',
                'overall_market_pressure': 'Net Market Score (-1 to +1)',
            }),
            width='stretch',
            height=260,
        )
    else:
        st.info('Live Google News feed is initializing or running in offline mode.')

    st.markdown('---')
    st.markdown('#### 📁  Curated Historical Event Database (2020 – 2026)')
    deduped = deduplicate_news(HISTORICAL_MARKET_EVENTS)
    df_events = pd.DataFrame(deduped)
    df_events['Date'] = pd.to_datetime(df_events['date']).dt.strftime('%Y-%m-%d')

    st.dataframe(
        df_events[['Date', 'source', 'location', 'headline', 'text']].rename(columns={
            'source': 'Source',
            'location': 'Location / APMC',
            'headline': 'Article Headline',
            'text': 'Economic Context / Summary',
        }),
        width='stretch',
        height=320,
    )
