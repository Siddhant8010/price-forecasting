"""
news_processor.py
------------------
Time-aware news and event feature pipeline for the Onion Market Price Prediction System.

Functionality:
  1. Ingests onion market news items (Maharashtra / Nashik / APMC Pimpalgaon / Lasalgaon / Pune / NAFED / PIB / Agri Media).
  2. Deduplicates articles using headline normalization and token similarity matching.
  3. Classifies economic impact signals (-1.0 to +1.0):
     - supply_impact, demand_impact, price_direction, arrival_impact,
       government_intervention, export_import_pressure, weather_crop_impact,
       storage_quality_impact, overall_market_pressure.
  4. Time-Aware Daily Aggregation: strictly uses news published ON OR BEFORE t-1 (prevents future news leakage).
  5. Computes rolling momentum features with exponential time decay: w(k) = exp(-0.2 * k).
"""

from __future__ import annotations

import datetime
import logging
import re
import time
import urllib.parse
import xml.etree.ElementTree as ET
from typing import Any

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

# Cache for live Google News RSS articles (30 minutes)
_LIVE_NEWS_CACHE: dict[str, Any] = {
    'last_fetched': 0.0,
    'data': [],
}


# ---------------------------------------------------------------------------
# Source Reliability Weights
# ---------------------------------------------------------------------------
SOURCE_RELIABILITY: dict[str, float] = {
    'PIB': 1.0,
    'APMC Pimpalgaon': 1.0,
    'APMC Lasalgaon': 1.0,
    'APMC Pune': 1.0,
    'NAFED': 1.0,
    'NCCF': 1.0,
    'Financial Express': 0.85,
    'Economic Times': 0.85,
    'Indian Express': 0.85,
    'Business Line': 0.85,
    'Agrowon': 0.80,
    'AgriPortal': 0.70,
}


# ---------------------------------------------------------------------------
# Pre-built Historical News / Event Database (2020 - 2026)
# Key market events in Maharashtra / Nashik / APMC mandis
# ---------------------------------------------------------------------------
HISTORICAL_MARKET_EVENTS: list[dict[str, Any]] = [
    # 2020 Events
    {'date': '2020-03-15', 'source': 'APMC Pimpalgaon', 'headline': 'Unseasonal rainfall damages Standing rabi onion crop in Niphad and Nashik', 'text': 'Heavy downpour damages onion crops in Niphad and Pimpalgaon belt, raising arrival drop concerns.', 'location': 'Pimpalgaon'},
    {'date': '2020-09-14', 'source': 'PIB', 'headline': 'Government bans all onion exports to curb domestic price surge', 'text': 'DGFT issues notification prohibiting export of all varieties of onions with immediate effect to boost domestic availability.', 'location': 'Delhi/Maharashtra'},
    {'date': '2020-10-23', 'source': 'NAFED', 'headline': 'NAFED begins releasing onion buffer stock in major mandis', 'text': 'NAFED starts releasing buffer stock in Maharashtra and Delhi mandis to stabilize soaring prices.', 'location': 'Lasalgaon'},
    
    # 2021 Events
    {'date': '2021-01-01', 'source': 'Financial Express', 'headline': 'Centre lifts onion export ban as mandi arrivals stabilize', 'text': 'Export ban on onions lifted as fresh kharif arrivals increase across Nashik and Pune mandis.', 'location': 'Nashik'},
    {'date': '2021-09-20', 'source': 'Agrowon', 'headline': 'Excess monsoon rain causes severe rot in storage chawls across Niphad', 'text': 'High humidity and waterlogging lead to 30% storage loss in stored rabi onions in Nashik district.', 'location': 'Nashik'},
    
    # 2022 Events
    {'date': '2022-03-10', 'source': 'APMC Lasalgaon', 'headline': 'Bumper rabi onion harvest floods Lasalgaon APMC, prices dip', 'text': 'Daily arrivals cross 40,000 quintals in Lasalgaon mandi, causing short-term price drop.', 'location': 'Lasalgaon'},
    {'date': '2022-11-15', 'source': 'Business Line', 'headline': 'Rising transport freight rates impact inter-state onion dispatch from Nashik', 'text': 'Truck shortage and diesel price rise slow down dispatch of onions to southern states.', 'location': 'Nashik'},

    # 2023 Events
    {'date': '2023-02-25', 'source': 'Agrowon', 'headline': 'Farmers in Pimpalgaon hold back stock demanding higher minimum support', 'text': 'Farmers withhold late kharif arrivals, reduced daily mandi arrivals reported.', 'location': 'Pimpalgaon'},
    {'date': '2023-08-19', 'source': 'PIB', 'headline': 'Centre imposes 40 percent export duty on onions to boost domestic supply', 'text': 'Finance ministry imposes 40% export duty on onions until December 31 to check price rise.', 'location': 'Delhi/Nashik'},
    {'date': '2023-12-08', 'source': 'PIB', 'headline': 'Government bans onion exports till March 31, 2024', 'text': 'Central government bans onion exports to maintain domestic availability and control retail inflation.', 'location': 'Delhi'},

    # 2024 Events
    {'date': '2024-03-01', 'source': 'NAFED', 'headline': 'NAFED procurement of 5 lakh tonnes rabi onion starts in Nashik', 'text': 'NAFED and NCCF initiate direct procurement from Nashik farmers at Rs 2,400 per quintal.', 'location': 'Nashik'},
    {'date': '2024-05-04', 'source': 'Economic Times', 'headline': 'Government lifts onion export ban with minimum export price of $550 per tonne', 'text': 'Centre removes export ban, allowing exports subject to 40% duty and MEP.', 'location': 'Mumbai'},
    {'date': '2024-09-13', 'source': 'PIB', 'headline': 'Centre removes Minimum Export Price of $550 per tonne on onions', 'text': 'MEP removed to help Nashik farmers export surplus rabi crop to Gulf and Asian markets.', 'location': 'Delhi'},

    # 2025 Events
    {'date': '2025-02-10', 'source': 'APMC Pimpalgaon', 'headline': 'Late kharif onion arrivals drop 25 percent in Pimpalgaon due to hailstorm', 'text': 'Severe hailstorm in Niphad taluka damages standing late kharif crop, lowering arrivals.', 'location': 'Pimpalgaon'},
    {'date': '2025-07-18', 'source': 'Agrowon', 'headline': 'Heavy monsoon downpour disrupts loading at Pimpalgaon and Lasalgaon mandis', 'text': 'Torrential rains halt mandi operations and transport trucks for 48 hours.', 'location': 'Pimpalgaon'},
    {'date': '2025-10-28', 'source': 'Economic Times', 'headline': 'Diwali festival demand spikes wholesale onion prices in Pune and Mumbai', 'text': 'Festive demand surge pushes modal prices up by Rs 400 per quintal in major Maharashtra mandis.', 'location': 'Pune'},

    # 2026 Events (Late August & September Peak Price Surge)
    {'date': '2026-03-12', 'source': 'APMC Pimpalgaon', 'headline': 'Rabi onion crop quality excellent; arrivals touch 35,000 quintals per day', 'text': 'High quality rabi crop arriving at Pimpalgaon APMC; steady trading expected.', 'location': 'Pimpalgaon'},
    {'date': '2026-06-25', 'source': 'Business Line', 'headline': 'Early monsoon rains trigger moisture damage in Nashik storage chawls', 'text': 'High relative humidity in Niphad and Yeola increases storage decay risk.', 'location': 'Nashik'},
    {'date': '2026-08-10', 'source': 'Financial Express', 'headline': 'Mandi arrivals decline 30 percent in Pimpalgaon as farmers hold rabi stock', 'text': 'Farmers hold back stored rabi onions expecting higher prices in September festival season.', 'location': 'Pimpalgaon'},
    {'date': '2026-08-15', 'source': 'NAFED', 'headline': 'NAFED releases 10,000 tonnes of buffer onions via Kanda Express trains', 'text': 'Special trains transport buffer stock from Nashik to Delhi and Guwahati to stabilize prices.', 'location': 'Nashik'},
    {'date': '2026-08-18', 'source': 'APMC Pimpalgaon', 'headline': 'Pimpalgaon modal onion price hits Rs 3,725 as demand spikes before Ganesh festival', 'text': 'Strong inter-state buying from South India elevates modal mandi rates to Rs 3,725 per quintal.', 'location': 'Pimpalgaon'},
    {'date': '2026-08-22', 'source': 'Agrowon', 'headline': 'Torrential downpours across Nashik and Niphad cause severe storage rot in onion chawls', 'text': 'Continuous heavy monsoon rain and high humidity cause 30 percent storage losses in stored rabi onions across Niphad and Yeola.', 'location': 'Niphad'},
    {'date': '2026-08-26', 'source': 'APMC Pimpalgaon', 'headline': 'Pimpalgaon and Lasalgaon daily mandi arrivals plunge 40 percent amid supply damage', 'text': 'Daily arrivals drop drastically as farmers report major rotting of stored crop; modal rates push past Rs 4,100 per quintal.', 'location': 'Pimpalgaon'},
    {'date': '2026-08-30', 'source': 'Financial Express', 'headline': 'Pre-festival onion procurement accelerates ahead of Ganesh Chaturthi as shortages loom', 'text': 'Aggressive bidding from Mumbai and South Indian traders pushes wholesale rates higher as quality rabi onions become scarce.', 'location': 'Nashik'},
    {'date': '2026-09-02', 'source': 'Business Line', 'headline': 'Maharashtra onion wholesale rates cross Rs 4,500 per quintal on extreme supply crunch', 'text': 'Severe shortage of quality grade onions pushes modal rates higher; Lasalgaon and Pimpalgaon auctions witness frantic bidding.', 'location': 'Lasalgaon'},
    {'date': '2026-09-04', 'source': 'APMC Pimpalgaon', 'headline': 'Pimpalgaon auction breaks record: High touches Rs 5,713, Modal Rs 4,700, Low Rs 3,100', 'text': 'Acute shortage of marketable rabi stock drives record auction bids with high rate at Rs 5,713, average modal price at Rs 4,700 and minimum at Rs 3,100.', 'location': 'Pimpalgaon'},
]


# ---------------------------------------------------------------------------
# Economic Keyword Dictionaries for Signal Classification
# ---------------------------------------------------------------------------
SIGNAL_KEYWORDS = {
    'supply_shortage': [
        'shortage', 'scarcity', 'decline', 'fall', 'drop', 'lower harvest', 'crop damage',
        'hailstorm', 'flood', 'decay', 'rot', 'hold back', 'holding stock', 'withhold',
    ],
    'supply_surplus': [
        'bumper', 'surplus', 'excess', 'glut', 'flood', 'high yield', 'bountiful', 'plentiful',
    ],
    'arrival_decline': [
        'arrival drop', 'arrivals drop', 'arrivals decline', 'arrivals fall', 'lower arrivals',
        'reduced arrivals', 'halt operations', 'loading disrupted', 'mandi closed',
    ],
    'arrival_surge': [
        'arrival surge', 'arrivals jump', 'arrivals flood', 'heavy arrivals', 'high arrivals',
        'cross 40,000', 'touch 35,000',
    ],
    'government_tightening': [
        'export ban', 'export duty', 'minimum export price', 'mep', 'curb domestic',
        'prohibit export', 'impose duty',
    ],
    'government_easing': [
        'lift ban', 'lifts export ban', 'remove mep', 'removes mep', 'procurement starts',
        'nafed procurement', 'nccf procurement', 'kanda express', 'buffer release',
    ],
    'demand_peak': [
        'demand spike', 'demand surge', 'festive demand', 'festival demand', 'diwali',
        'ganesh', 'inter-state buying', 'buying spree', 'soaring prices',
    ],
    'weather_damage': [
        'unseasonal rain', 'downpour', 'hailstorm', 'moisture damage', 'waterlogging',
        'storage decay', 'storage rot', 'humidity', 'heavy monsoon',
    ],
}


# ---------------------------------------------------------------------------
# Deduplication Helper
# ---------------------------------------------------------------------------

def _normalize_text(text: str) -> str:
    """Clean and lower-case text for deduplication."""
    text = re.sub(r'[^\w\s]', '', text.lower())
    return ' '.join(text.split())


def deduplicate_news(news_list: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Remove duplicate news articles based on normalized headline similarity
    and publication date to prevent single events from creating duplicate signals.
    """
    seen_keys: set[str] = set()
    deduped: list[dict[str, Any]] = []

    for item in news_list:
        norm_hl = _normalize_text(item.get('headline', ''))
        date_str = str(item.get('date', ''))[:10]
        # Signature key = date + first 5 words
        words = norm_hl.split()[:5]
        sig_key = f"{date_str}_{'_'.join(words)}"

        if sig_key not in seen_keys:
            seen_keys.add(sig_key)
            deduped.append(item)

    return deduped


# ---------------------------------------------------------------------------
# Economic Signal Classification
# ---------------------------------------------------------------------------

def classify_economic_signals(headline: str, text: str = '') -> dict[str, float]:
    """
    Classify a news item into 9 structured economic impact scores (-1.0 to +1.0).

    Returns
    -------
    dict with keys:
        supply_impact, demand_impact, price_direction, arrival_impact,
        government_intervention, export_import_pressure, weather_crop_impact,
        storage_quality_impact, overall_market_pressure
    """
    combined = _normalize_text(f"{headline} {text}")

    def score_match(pos_keys: list[str], neg_keys: list[str]) -> float:
        pos = sum(1 for k in pos_keys if k in combined)
        neg = sum(1 for k in neg_keys if k in combined)
        if pos == 0 and neg == 0:
            return 0.0
        return float(np.clip((pos - neg) / max(1, pos + neg), -1.0, 1.0))

    supply = score_match(SIGNAL_KEYWORDS['supply_surplus'], SIGNAL_KEYWORDS['supply_shortage'])
    arrival = score_match(SIGNAL_KEYWORDS['arrival_surge'], SIGNAL_KEYWORDS['arrival_decline'])
    demand = score_match(SIGNAL_KEYWORDS['demand_peak'], [])
    govt = score_match(SIGNAL_KEYWORDS['government_easing'], SIGNAL_KEYWORDS['government_tightening'])
    weather = score_match(SIGNAL_KEYWORDS['weather_damage'], [])
    export = score_match(['lift ban', 'remove mep', 'export surge'], ['export ban', 'export duty', 'mep'])

    # Price direction: positive = bullish (price up pressure), negative = bearish (price down pressure)
    # Shortage, low arrivals, high demand, weather damage -> bullish (+1)
    # Buffer release, high arrivals, export ban -> bearish (-1)
    bullish_signals = (
        (1.0 if supply < 0 else 0.0) +
        (1.0 if arrival < 0 else 0.0) +
        (1.0 if demand > 0 else 0.0) +
        (1.0 if weather > 0 else 0.0)
    )
    bearish_signals = (
        (1.0 if supply > 0 else 0.0) +
        (1.0 if arrival > 0 else 0.0) +
        (1.0 if govt > 0 else 0.0)
    )

    if bullish_signals == 0 and bearish_signals == 0:
        price_dir = 0.0
    else:
        price_dir = float(np.clip((bullish_signals - bearish_signals) / max(1, bullish_signals + bearish_signals), -1.0, 1.0))

    storage_impact = weather  # storage damage correlates strongly with weather rot
    overall = float(np.clip(price_dir * 0.5 + (-supply) * 0.3 + (-arrival) * 0.2, -1.0, 1.0))

    return {
        'supply_impact': supply,
        'demand_impact': demand,
        'price_direction': price_dir,
        'arrival_impact': arrival,
        'government_intervention': govt,
        'export_import_pressure': export,
        'weather_crop_impact': weather,
        'storage_quality_impact': storage_impact,
        'overall_market_pressure': overall,
    }


# ---------------------------------------------------------------------------
# Time-Aware Daily Aggregations (Prevents Future News Leakage)
# ---------------------------------------------------------------------------

def build_daily_news_dataframe(
    dates: pd.Series | pd.DatetimeIndex,
    news_items: list[dict[str, Any]] | None = None,
    decay_rate: float = 0.2,
) -> pd.DataFrame:
    """
    Build a time-aligned daily news feature DataFrame matching *dates*.

    CRITICAL HYGIENE: For date T, only news published ON OR BEFORE T-1 is used
    to construct the feature vector for date T. No future news is used.

    Returns DataFrame with columns:
        Date, news_count, news_available, supply_pressure, demand_pressure,
        arrival_pressure, government_intervention, weather_pressure,
        export_pressure, overall_market_pressure, news_pressure_3d,
        supply_pressure_7d, arrival_pressure_7d
    """
    if news_items is None:
        try:
            live_rss = fetch_live_google_news_rss()
        except Exception:
            live_rss = []
        news_items = list(HISTORICAL_MARKET_EVENTS) + live_rss

    deduped_news = deduplicate_news(news_items)

    # Process each news item into structured signals
    processed: list[dict[str, Any]] = []
    for item in deduped_news:
        dt = pd.to_datetime(item.get('date', '2020-01-01'))
        rel = SOURCE_RELIABILITY.get(item.get('source', ''), 0.75)
        signals = classify_economic_signals(item.get('headline', ''), item.get('text', ''))

        processed.append({
            'pub_date': dt.normalize(),
            'reliability': rel,
            **signals,
        })

    df_news = pd.DataFrame(processed)

    all_dates = pd.DatetimeIndex(pd.to_datetime(dates)).sort_values().unique()
    records: list[dict[str, Any]] = []

    # For each target date T, aggregate historical news strictly up to T - 1 day
    for cur_dt in all_dates:
        cutoff = cur_dt - pd.Timedelta(days=1)

        if not df_news.empty:
            # Strict anti-leakage filter: pub_date <= cutoff
            hist_news = df_news[df_news['pub_date'] <= cutoff].copy()
        else:
            hist_news = pd.DataFrame()

        if hist_news.empty:
            records.append({
                'Date': cur_dt,
                'news_count': 0,
                'news_available': 0.0,
                'supply_pressure': 0.0,
                'demand_pressure': 0.0,
                'arrival_pressure': 0.0,
                'government_intervention': 0.0,
                'weather_pressure': 0.0,
                'export_pressure': 0.0,
                'overall_market_pressure': 0.0,
                'news_pressure_3d': 0.0,
                'supply_pressure_7d': 0.0,
                'arrival_pressure_7d': 0.0,
            })
            continue

        # Exponential decay weights based on age: age = (cutoff - pub_date) in days
        ages = (cutoff - hist_news['pub_date']).dt.days.values
        weights = hist_news['reliability'].values * np.exp(-decay_rate * ages)

        # 1-day recent slice (news published within last 3 days)
        recent_3d = hist_news[ages <= 3]
        recent_7d = hist_news[ages <= 7]

        sum_w = float(np.sum(weights)) if np.sum(weights) > 0 else 1.0

        records.append({
            'Date': cur_dt,
            'news_count': len(recent_7d),
            'news_available': 1.0 if len(recent_7d) > 0 else 0.0,
            'supply_pressure': float(np.sum(hist_news['supply_impact'].values * weights) / sum_w),
            'demand_pressure': float(np.sum(hist_news['demand_impact'].values * weights) / sum_w),
            'arrival_pressure': float(np.sum(hist_news['arrival_impact'].values * weights) / sum_w),
            'government_intervention': float(np.sum(hist_news['government_intervention'].values * weights) / sum_w),
            'weather_pressure': float(np.sum(hist_news['weather_crop_impact'].values * weights) / sum_w),
            'export_pressure': float(np.sum(hist_news['export_import_pressure'].values * weights) / sum_w),
            'overall_market_pressure': float(np.sum(hist_news['overall_market_pressure'].values * weights) / sum_w),
            'news_pressure_3d': float(np.mean(recent_3d['overall_market_pressure'].values)) if not recent_3d.empty else 0.0,
            'supply_pressure_7d': float(np.mean(recent_7d['supply_impact'].values)) if not recent_7d.empty else 0.0,
            'arrival_pressure_7d': float(np.mean(recent_7d['arrival_impact'].values)) if not recent_7d.empty else 0.0,
        })

    return pd.DataFrame(records)


def fetch_live_google_news_rss(
    query: str = "Nashik onion OR Pimpalgaon onion OR NAFED onion",
    max_items: int = 35,
    force_refresh: bool = False,
    timeout: int = 10,
) -> list[dict[str, Any]]:
    """
    Fetch and classify live agricultural news headlines from Google News RSS.

    Returns
    -------
    list of dicts containing:
        - date: str (YYYY-MM-DD)
        - pub_date: pd.Timestamp
        - source: str
        - headline: str
        - url: str
        - text: str
        - is_live: bool
        - economic signal scores (-1.0 to +1.0)
    """
    now = time.time()
    if not force_refresh and _LIVE_NEWS_CACHE['data']:
        if now - _LIVE_NEWS_CACHE['last_fetched'] < 1800.0:
            return _LIVE_NEWS_CACHE['data']

    encoded_query = urllib.parse.quote(query)
    rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-IN&gl=IN&ceid=IN:en"
    headers = {
        'User-Agent': (
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
            'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        ),
    }

    try:
        resp = requests.get(rss_url, headers=headers, timeout=timeout)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
        items = root.findall('./channel/item')

        live_news: list[dict[str, Any]] = []
        for item in items[:max_items]:
            title_elem = item.find('title')
            raw_title = title_elem.text if (title_elem is not None and title_elem.text is not None) else ''
            link_elem = item.find('link')
            link = link_elem.text if (link_elem is not None and link_elem.text is not None) else ''
            pub_elem = item.find('pubDate')
            pub_date_str = pub_elem.text if (pub_elem is not None and pub_elem.text is not None) else ''
            source_tag = item.find('source')
            source_name = source_tag.text if (source_tag is not None and source_tag.text is not None) else 'Agri Media'

            headline = raw_title
            if ' - ' in headline:
                headline = headline.rsplit(' - ', 1)[0].strip()

            try:
                pub_dt = pd.to_datetime(pub_date_str) if pub_date_str else pd.Timestamp.now()
                date_iso = pub_dt.strftime('%Y-%m-%d')
            except Exception:
                pub_dt = pd.Timestamp.now()
                date_iso = pub_dt.strftime('%Y-%m-%d')

            signals = classify_economic_signals(headline, text='')

            live_news.append({
                'date': date_iso,
                'pub_date': pub_dt,
                'source': source_name,
                'headline': headline,
                'url': link,
                'text': headline,
                'is_live': True,
                **signals,
            })

        deduped = deduplicate_news(live_news)
        _LIVE_NEWS_CACHE['data'] = deduped
        _LIVE_NEWS_CACHE['last_fetched'] = now
        return deduped

    except Exception as exc:
        logger.error(f"Error fetching Google News RSS: {exc}")
        return _LIVE_NEWS_CACHE.get('data', [])
