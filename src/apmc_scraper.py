"""
apmc_scraper.py
---------------
Real-time web scraper for official daily onion auction rates from
Agricultural Produce Market Committee (APMC), Pimpalgaon Baswant.

Source:
    https://apmcpimpalgaon.com/
Commodity tracked:
    Unhal Kanda (उन्हाळ कांदा / Summer Rabi Onion)
"""

from __future__ import annotations

import datetime
import logging
import re
import time
from typing import Any

import pandas as pd
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# In-memory cache to avoid excessive requests during interactive dashboard use
_CACHE: dict[str, Any] = {
    'last_fetched': 0.0,
    'data': None,
}
_CACHE_EXPIRY_SECONDS = 1800.0  # 30 minutes


def _parse_date_string(date_str: str) -> pd.Timestamp | None:
    """Parse typical Indian APMC date format (DD-MM-YYYY or DD/MM/YYYY)."""
    clean_str = date_str.strip()
    for fmt in ('%d-%m-%Y', '%d/%m/%Y', '%Y-%m-%d'):
        try:
            return pd.to_datetime(clean_str, format=fmt)
        except (ValueError, TypeError):
            continue
    try:
        return pd.to_datetime(clean_str)
    except Exception:
        return None


def fetch_live_apmc_pimpalgaon_rate(
    force_refresh: bool = False,
    timeout: int = 10,
) -> dict[str, Any] | None:
    """
    Scrape the latest Unhal Kanda daily auction rates from apmcpimpalgaon.com.

    Parameters
    ----------
    force_refresh : bool
        If True, ignores cache and makes a fresh HTTP request.
    timeout : int
        Request timeout in seconds.

    Returns
    -------
    dict with keys:
        - variety: str
        - date_str: str (DD-MM-YYYY)
        - date: pd.Timestamp
        - min_price: float (Rs/Quintal)
        - max_price: float (Rs/Quintal)
        - modal_price: float (Rs/Quintal)
        - spread: float (Rs/Quintal)
        - source_url: str
        - fetched_at: datetime
    or None if the site is unreachable or parsing fails.
    """
    now = time.time()
    if not force_refresh and _CACHE['data'] is not None:
        if now - _CACHE['last_fetched'] < _CACHE_EXPIRY_SECONDS:
            return _CACHE['data']

    url = 'https://apmcpimpalgaon.com/'
    headers = {
        'User-Agent': (
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
            'AppleWebKit/537.36 (KHTML, like Gecko) '
            'Chrome/120.0.0.0 Safari/537.36'
        ),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    }

    try:
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # Find the Daily Rate table
        # Structure: <th>ID</th>, <th>Vegetable Name</th>, <th>Date</th>, <th>Minimum Rate</th>, <th>Maximum Rate</th>, <th>Average Rate</th>
        matched_row = None
        for tr in soup.find_all(['tr', 'tbody']):
            tds = [td.get_text(strip=True) for td in tr.find_all('td')]
            if len(tds) >= 6:
                veg_name = tds[1].lower()
                # Target 'Unhal Kanda' specifically (avoid Goltti variety as main benchmark)
                if 'unhal kanda' in veg_name and 'goltti' not in veg_name:
                    matched_row = tds
                    break

        if not matched_row:
            # Fallback: check any row mentioning Unhal Kanda
            for tr in soup.find_all(['tr', 'tbody']):
                tds = [td.get_text(strip=True) for td in tr.find_all('td')]
                if len(tds) >= 6 and 'unhal' in tds[1].lower():
                    matched_row = tds
                    break

        if not matched_row:
            logger.warning("Unhal Kanda rate row not found on apmcpimpalgaon.com homepage.")
            return _CACHE.get('data')

        raw_id = matched_row[0]
        variety_name = matched_row[1]
        raw_date = matched_row[2]
        raw_min = float(re.sub(r'[^\d.]', '', matched_row[3]))
        raw_max = float(re.sub(r'[^\d.]', '', matched_row[4]))
        raw_avg = float(re.sub(r'[^\d.]', '', matched_row[5]))

        # Sanity check rates
        if raw_avg <= 0 and raw_max > 0:
            raw_avg = (raw_min + raw_max) / 2.0

        if raw_min <= 0 or raw_avg <= 0 or raw_max <= 0:
            logger.warning(f"Invalid non-positive rates extracted: {matched_row}")
            return _CACHE.get('data')

        date_dt = _parse_date_string(raw_date)
        if date_dt is None:
            date_dt = pd.Timestamp.now().normalize()

        result = {
            'variety': variety_name,
            'date_str': raw_date,
            'date': date_dt,
            'min_price': raw_min,
            'modal_price': raw_avg,
            'max_price': raw_max,
            'spread': raw_max - raw_min,
            'source_url': url,
            'source_name': 'APMC Pimpalgaon Baswant (Official Web Portal)',
            'fetched_at': datetime.datetime.now(),
        }

        _CACHE['data'] = result
        _CACHE['last_fetched'] = now
        return result

    except Exception as exc:
        logger.error(f"Error scraping apmcpimpalgaon.com: {exc}")
        # Return previously cached data if available
        return _CACHE.get('data')
