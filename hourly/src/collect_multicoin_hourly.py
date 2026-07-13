"""
멀티코인 스테이블코인 시간단위(1h) OHLCV 수집
- Binance klines (무료·무인증, 상장폐지 심볼도 과거 데이터 제공):
    USTUSDT(~2022-05 붕괴·폐지), BUSDUSDT(~2023 폐지), TUSDUSDT, USDPUSDT,
    FDUSDUSDT(2023-07~), USDEUSDT(2025-09~)
- Coinbase Exchange candles (무료·무인증, 300봉/요청):
    DAI-USD (2020년 말~, SVB 2023-03 커버)
출력: data/collect/raw/binance_hourly_{SYM}.csv, coinbase_hourly_DAI-USD.csv
"""

import requests
import pandas as pd
import time
import os
from datetime import datetime, timezone, timedelta

SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw")
os.makedirs(SAVE_DIR, exist_ok=True)

START = datetime(2020, 1, 1, tzinfo=timezone.utc)

BINANCE_SYMBOLS = ["USTUSDT", "BUSDUSDT", "TUSDUSDT", "USDPUSDT", "FDUSDUSDT", "USDEUSDT"]
BINANCE_COLS = ["open_time", "open", "high", "low", "close", "volume",
                "close_time", "quote_volume", "n_trades",
                "taker_buy_base", "taker_buy_quote", "ignore"]


def fetch_binance(symbol):
    rows = []
    start = int(START.timestamp() * 1000)
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    while start < now_ms:
        r = requests.get("https://api.binance.com/api/v3/klines",
                         params={"symbol": symbol, "interval": "1h",
                                 "startTime": start, "limit": 1000}, timeout=30)
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        rows.extend(batch)
        start = batch[-1][0] + 3600_000
        if len(batch) < 1000:
            break
        time.sleep(0.3)
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=BINANCE_COLS)
    df["datetime"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    num = ["open", "high", "low", "close", "volume", "quote_volume", "n_trades"]
    df[num] = df[num].astype(float)
    return df[["datetime"] + num].drop_duplicates("datetime").sort_values("datetime").reset_index(drop=True)


def fetch_coinbase(product):
    """Coinbase Exchange candles: 최대 300봉/요청, [time, low, high, open, close, volume]"""
    rows = []
    cur = START
    now = datetime.now(timezone.utc)
    step = timedelta(hours=300)
    while cur < now:
        end = min(cur + step, now)
        r = requests.get(f"https://api.exchange.coinbase.com/products/{product}/candles",
                         params={"granularity": 3600,
                                 "start": cur.isoformat(), "end": end.isoformat()},
                         timeout=30, headers={"User-Agent": "research"})
        if r.status_code == 200:
            batch = r.json()
            if isinstance(batch, list):
                rows.extend(batch)
        cur = end
        time.sleep(0.15)
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=["time", "low", "high", "open", "close", "volume"])
    df["datetime"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df["quote_volume"] = df["volume"] * df["close"]   # 근사 (Coinbase는 base volume)
    df["n_trades"] = float("nan")                      # 미제공
    cols = ["datetime", "open", "high", "low", "close", "volume", "quote_volume", "n_trades"]
    return df[cols].drop_duplicates("datetime").sort_values("datetime").reset_index(drop=True)


def main():
    for sym in BINANCE_SYMBOLS:
        print(f"수집 중: Binance {sym} ...")
        df = fetch_binance(sym)
        if df is None:
            print("  데이터 없음\n"); continue
        path = os.path.join(SAVE_DIR, f"binance_hourly_{sym}.csv")
        df.to_csv(path, index=False)
        print(f"  {len(df):,}행 | {df['datetime'].min()} ~ {df['datetime'].max()} | 최저종가={df['close'].min():.4f}\n")

    print("수집 중: Coinbase DAI-USD ...")
    df = fetch_coinbase("DAI-USD")
    if df is not None:
        path = os.path.join(SAVE_DIR, "coinbase_hourly_DAI-USD.csv")
        df.to_csv(path, index=False)
        print(f"  {len(df):,}행 | {df['datetime'].min()} ~ {df['datetime'].max()} | 최저종가={df['close'].min():.4f}")


if __name__ == "__main__":
    main()
