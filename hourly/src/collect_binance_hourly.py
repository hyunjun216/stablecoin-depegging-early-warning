"""
Binance klines 시간단위(1h) OHLCV 수집
- 대상: USDCUSDT(디페깅 분석 주코인), BTCUSDT, ETHUSDT(암호시장 피처)
- 기간: 2020-01-01 ~ 현재, 1시간봉
- 무료·무인증 (API 키 불필요)
출력: data/raw/binance_hourly_{SYMBOL}.csv
"""

import requests
import pandas as pd
import time
import os
from datetime import datetime, timezone

SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw")
os.makedirs(SAVE_DIR, exist_ok=True)

BASE = "https://api.binance.com/api/v3/klines"
SYMBOLS = ["USDCUSDT", "BTCUSDT", "ETHUSDT"]
INTERVAL = "1h"
START = int(datetime(2020, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
LIMIT = 1000  # Binance 최대

COLS = ["open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "n_trades",
        "taker_buy_base", "taker_buy_quote", "ignore"]


def fetch_symbol(symbol):
    """페이징으로 START~현재 전체 1h 봉 수집"""
    rows = []
    start = START
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    while start < now_ms:
        params = {"symbol": symbol, "interval": INTERVAL,
                  "startTime": start, "limit": LIMIT}
        r = requests.get(BASE, params=params, timeout=30)
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        rows.extend(batch)
        last_open = batch[-1][0]
        # 다음 배치는 마지막 봉 다음 시각부터
        start = last_open + 3600_000
        if len(batch) < LIMIT:
            break
        time.sleep(0.3)  # rate-limit 여유

    df = pd.DataFrame(rows, columns=COLS)
    # 타입 변환
    df["datetime"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    num = ["open", "high", "low", "close", "volume", "quote_volume", "n_trades"]
    df[num] = df[num].astype(float)
    df = df[["datetime"] + num].drop_duplicates(subset="datetime").sort_values("datetime").reset_index(drop=True)
    return df


def main():
    for sym in SYMBOLS:
        print(f"수집 중: {sym} ({INTERVAL}) ...")
        df = fetch_symbol(sym)
        path = os.path.join(SAVE_DIR, f"binance_hourly_{sym}.csv")
        df.to_csv(path, index=False)
        print(f"  완료: {len(df):,}행 | {df['datetime'].min()} ~ {df['datetime'].max()}")
        print(f"  저가 최저={df['low'].min():.4f}, 종가범위={df['close'].min():.4f}~{df['close'].max():.4f}")
        print(f"  저장: {path}\n")


if __name__ == "__main__":
    main()
