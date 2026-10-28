import os
import logging
from typing import Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Massive REST client (optional – falls back to yfinance if key is invalid)
# ---------------------------------------------------------------------------
try:
    from massive import RESTClient as _RESTClient
    _MASSIVE_AVAILABLE = True
except ImportError:
    _MASSIVE_AVAILABLE = False

MEMBERS_FOR_BACKTEST = [
    'AAPL','MSFT','JPM','JNJ','XOM','GE','BAC','WFC','PG','CVX',
    'IBM','T','VZ','MRK','PFE','INTC','KO','PEP','ABT','MCD',
    'MMM','HON','UPS','BA','CAT','GS','MS','C','USB','AXP',
    'WMT','TGT','COST','HD','LOW','F','GM','FCX','AA','X',
    'NEM','APA','DVN','HAL','SLB','BK','STT','MTB','CMA','KEY',
    'DD','DOW','EMN','LYB','PPG','GD','LMT','NOC','RTX','L',
    'SO','D','DUK','AEP','EXC','AMT','PLD','SPG','AVB','EQR',
    'MDT','BSX','SYK','BDX','ZBH','DHR','TMO','BAX','BK','HIG',
    'CB','TRV','AON','MMC','AFL','MET','PRU','LNC','UNH','CI'
]


def _massive_key_works() -> bool:
    """Quick probe to check if the Massive API key is valid."""
    if not _MASSIVE_AVAILABLE:
        return False
    try:
        key = os.getenv('MASSIVE_TOKEN') or os.getenv('MASSIVE_API_KEY', '')
        client = _RESTClient(api_key=key)
        # Attempt one tiny fetch as a probe
        list(client.list_aggs('AAPL', multiplier=1, timespan='day',
                               from_='2013-01-02', to='2013-01-03',
                               adjusted=True, limit=5))
        return True
    except Exception:
        return False


_USE_MASSIVE: Optional[bool] = None   # lazy-cached


def _should_use_massive() -> bool:
    global _USE_MASSIVE
    if _USE_MASSIVE is None:
        _USE_MASSIVE = _massive_key_works()
        if _USE_MASSIVE:
            logger.info("Massive API key is valid – using Massive for data.")
        else:
            logger.info("Massive API key invalid or unavailable – falling back to yfinance.")
    return _USE_MASSIVE


# ---------------------------------------------------------------------------
# Massive fetch path
# ---------------------------------------------------------------------------

def _fetch_massive_bars(tickers: list[str], from_date: str, to_date: str,
                        adjusted: bool) -> pd.DataFrame:
    key = os.getenv('MASSIVE_TOKEN') or os.getenv('MASSIVE_API_KEY', '')
    client = _RESTClient(api_key=key)
    all_data: dict = {}
    for ticker in tickers:
        try:
            aggs = []
            for a in client.list_aggs(
                ticker=ticker, multiplier=1, timespan='day',
                from_=from_date, to=to_date, adjusted=adjusted, limit=50000,
            ):
                aggs.append({
                    'date': pd.Timestamp(a.timestamp, unit='ms', tz='UTC').date(),
                    'open': a.open, 'high': a.high,
                    'low': a.low, 'close': a.close, 'volume': a.volume,
                })
            if aggs:
                df = pd.DataFrame(aggs)
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date').sort_index()
                df = df[~df.index.duplicated(keep='last')]
                all_data[ticker] = df
                logger.info(f'Massive: fetched {len(df)} rows for {ticker}')
        except Exception as e:
            logger.warning(f'Massive fetch failed for {ticker}: {e}')
    if not all_data:
        return pd.DataFrame()
    dfs = []
    for ticker, df in all_data.items():
        df.columns = pd.MultiIndex.from_tuples([(c, ticker) for c in df.columns])
        dfs.append(df)
    return pd.concat(dfs, axis=1).sort_index()


# ---------------------------------------------------------------------------
# yfinance fetch path
# ---------------------------------------------------------------------------

def _fetch_yfinance_bars(tickers: list[str], from_date: str, to_date: str,
                         adjusted: bool) -> pd.DataFrame:
    """Fetch OHLCV bars via yfinance.

    yfinance returns adjusted close natively; we approximate unadjusted OHLCV
    by fetching with auto_adjust=False when adjusted=False, and auto_adjust=True
    when adjusted=True. For open prices we use the same adjustment ratio
    applied to close, consistent with the paper's practice.
    """
    import yfinance as yf
    # Batch download
    logger.info(f'yfinance: downloading {len(tickers)} tickers '
                f'[{from_date} → {to_date}] adjusted={adjusted}')

    # yf.download returns adjusted (auto_adjust=True) or raw (auto_adjust=False)
    raw = yf.download(
        tickers=tickers,
        start=from_date,
        end=pd.Timestamp(to_date) + pd.Timedelta(days=1),  # yf end is exclusive
        auto_adjust=adjusted,
        progress=False,
        group_by='ticker',
        threads=True,
    )

    if raw.empty:
        logger.error("yfinance returned empty DataFrame")
        return pd.DataFrame()

    all_data: dict = {}
    valid_tickers = tickers if len(tickers) > 1 else tickers

    for ticker in valid_tickers:
        try:
            if len(tickers) == 1:
                df_t = raw.copy()
            else:
                if ticker not in raw.columns.get_level_values(0):
                    continue
                df_t = raw[ticker].copy()

            df_t.columns = [c.lower() for c in df_t.columns]
            df_t = df_t[['open', 'high', 'low', 'close', 'volume']].dropna(how='all')
            df_t.index = pd.to_datetime(df_t.index)
            df_t = df_t[~df_t.index.duplicated(keep='last')]
            if not df_t.empty:
                all_data[ticker] = df_t
        except Exception as e:
            logger.warning(f'yfinance parse failed for {ticker}: {e}')

    if not all_data:
        return pd.DataFrame()

    dfs = []
    for ticker, df in all_data.items():
        df.columns = pd.MultiIndex.from_tuples([(c, ticker) for c in df.columns])
        dfs.append(df)
    return pd.concat(dfs, axis=1).sort_index()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_daily_bars(
    tickers: list[str],
    from_date: str,
    to_date: str,
    adjusted: bool = True,
) -> pd.DataFrame:
    """Fetch daily OHLCV bars, using Massive if the key works, else yfinance."""
    if _should_use_massive():
        return _fetch_massive_bars(tickers, from_date, to_date, adjusted)
    return _fetch_yfinance_bars(tickers, from_date, to_date, adjusted)


def fetch_both_adjusted_and_unadjusted(
    tickers: list[str],
    from_date: str,
    to_date: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch both adjusted and unadjusted daily bars."""
    logger.info(f'Fetching adjusted bars for {len(tickers)} tickers {from_date}→{to_date}')
    adj = fetch_daily_bars(tickers, from_date, to_date, adjusted=True)
    logger.info(f'Fetching unadjusted bars for {len(tickers)} tickers {from_date}→{to_date}')
    unadj = fetch_daily_bars(tickers, from_date, to_date, adjusted=False)
    return adj, unadj


def build_price_matrices(
    tickers: list[str],
    from_date: str = '2008-08-01',
    to_date: str = '2014-09-05',
) -> dict[str, pd.DataFrame]:
    """Fetch all required price data and return a dict of price matrices.

    Keys: 'adj_close', 'adj_open', 'unadj_close', 'unadj_open', 'unadj_volume'
    Each is a DataFrame with dates as index, tickers as columns.
    """
    adj, unadj = fetch_both_adjusted_and_unadjusted(tickers, from_date, to_date)

    result = {}

    def extract(df: pd.DataFrame, field: str) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame()
        cols = [(field, t) for t in tickers if (field, t) in df.columns]
        sub = df[cols]
        sub.columns = [t for _, t in sub.columns]
        return sub

    result['adj_close'] = extract(adj, 'close')
    result['adj_open'] = extract(adj, 'open')
    result['unadj_close'] = extract(unadj, 'close')
    result['unadj_open'] = extract(unadj, 'open')
    result['unadj_volume'] = extract(unadj, 'volume')

    return result


def load_or_fetch_price_data(
    tickers: list[str],
    from_date: str = '2007-01-01',
    to_date: str = '2014-09-05',
    cache_path: Optional[str] = None,
) -> dict[str, pd.DataFrame]:
    """Load price data from pickle cache if available, otherwise fetch.

    Cache format: pickle (.pkl). If cache_path ends with '.h5' it is treated
    as a legacy HDF5 path and re-routed to a sibling '.pkl' file.
    """
    import pickle

    # Normalise cache path to pickle
    if cache_path and cache_path.endswith('.h5'):
        cache_path = cache_path.replace('.h5', '.pkl')

    if cache_path and os.path.exists(cache_path):
        logger.info(f'Loading price data from cache: {cache_path}')
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    data = build_price_matrices(tickers, from_date, to_date)

    if cache_path:
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f, protocol=4)
        logger.info(f'Price data cached to: {cache_path}')

    return data
