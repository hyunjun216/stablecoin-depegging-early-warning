"""
09_build_multicoin_dataset.py — 멀티코인 pooled hourly 데이터셋 빌드

코인: USDC, UST, BUSD, TUSD, USDP, FDUSD, USDE (Binance) + DAI (Coinbase)
각 코인에 h1과 동일 파이프라인(wick 정제 → 세그먼트 → 종가기준 타겟 → 코어 피처) 적용 후
long format으로 pooling.

h1 대비 변경 (pooled 대응):
  - scale-free 피처만: 절대수준 변수(log_qvol, log_ntrades, btc_vol 레벨) 제외
    → 코인·거래소 간 스케일 차이가 모델을 오염시키지 않게
  - UST 좀비구간 절단: 종가<0.90이 72h 지속되는 시점 이후 제거
    (영구 붕괴된 죽은 코인은 조기경보 대상이 아님 — 붕괴 진입까지만 유지)
출력: data/processed/df_multicoin_hourly.csv
"""

import pandas as pd
import numpy as np
import os
from pandas.api.indexers import FixedForwardWindowIndexer

PROJECT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
RAW = os.path.join(PROJECT_DIR, "data", "collect", "raw")
OUT_DIR = os.path.join(PROJECT_DIR, "data", "processed")

COINS = {
    "USDC":  ("binance_hourly_USDCUSDT.csv", "fiat"),
    "UST":   ("binance_hourly_USTUSDT.csv", "algo"),
    "BUSD":  ("binance_hourly_BUSDUSDT.csv", "fiat"),
    "TUSD":  ("binance_hourly_TUSDUSDT.csv", "fiat"),
    "USDP":  ("binance_hourly_USDPUSDT.csv", "fiat"),
    "FDUSD": ("binance_hourly_FDUSDUSDT.csv", "fiat"),
    "USDE":  ("binance_hourly_USDEUSDT.csv", "crypto"),
    "DAI":   ("coinbase_hourly_DAI-USD.csv", "crypto"),
}

HORIZONS = [1, 6, 24]
TAU = 0.005
PERSIST_N, PERSIST_K = 3, 2
GAP = pd.Timedelta("1h")
WARMUP = 6
ZOMBIE_HOURS = 72          # 종가<0.9가 이 시간 지속되면 이후 절단


def load_coin(fname):
    df = pd.read_csv(os.path.join(RAW, fname), parse_dates=["datetime"])
    df["datetime"] = df["datetime"].dt.tz_localize(None)
    return df.sort_values("datetime").reset_index(drop=True)


def clean_wicks(df):
    bad_low = (df["low"] < 0.90) & (df["close"] > 0.97)
    bad_high = (df["high"] > 1.10) & (df["close"] < 1.03)
    df.loc[bad_low, "low"] = df.loc[bad_low, ["open", "close"]].min(axis=1)
    df.loc[bad_high, "high"] = df.loc[bad_high, ["open", "close"]].max(axis=1)
    return df, int(bad_low.sum() + bad_high.sum())


def truncate_zombie(df):
    """영구 붕괴 이후 절단: 종가<0.9의 ZOMBIE_HOURS 연속 rolling min 확인"""
    dead = (df["close"] < 0.90).rolling(ZOMBIE_HOURS).min()
    idx = np.where(dead == 1)[0]
    if len(idx) == 0:
        return df, None
    cut = idx[0]  # 이 시점에서 이미 72h 연속 죽어있음 → 여기까지만 유지
    cut_time = df.loc[cut, "datetime"]
    return df.iloc[:cut + 1].copy(), cut_time


def build_one(coin, fname):
    df = load_coin(fname)
    df, n_wick = clean_wicks(df)
    df, cut_time = truncate_zombie(df)

    # 세그먼트 (결측 블록 경계)
    df["seg"] = (df["datetime"].diff() > GAP).cumsum()

    def g(col, fn):
        return df.groupby("seg")[col].transform(fn)

    # ── 타겟 (h1과 동일: 종가, tau05, 2-of-3, 전방라벨) ──
    df["dev"] = (df["close"] - 1.0).abs()
    raw = (df["dev"] > TAU).astype(int)
    rs = df.assign(_r=raw).groupby("seg")["_r"].transform(
        lambda s: s.rolling(PERSIST_N, min_periods=1).sum())
    df["depeg"] = (rs >= PERSIST_K).astype(int)
    for H in HORIZONS:
        fwd = FixedForwardWindowIndexer(window_size=H)
        df[f"y_h{H}"] = (df.groupby("seg")["depeg"]
                           .transform(lambda s: s.shift(-1).rolling(fwd, min_periods=1).max()))
        df[f"ymax_dev_h{H}"] = (df.groupby("seg")["dev"]
                                  .transform(lambda s: s.shift(-1).rolling(fwd, min_periods=1).max()))

    # ── scale-free 코어 피처 ──
    for n in [1, 6, 24]:
        df[f"ret_{n}h"] = g("close", lambda s: s.pct_change(n))
    df["ret1"] = df["ret_1h"]
    for w in [6, 24, 72]:
        df[f"vol_{w}h"] = g("ret1", lambda s: s.rolling(w).std())
    for w in [24, 168]:
        ma = g("close", lambda s: s.rolling(w).mean())
        df[f"ma{w}_dev"] = df["close"] - ma
    delta = g("close", lambda s: s.diff())
    df["_gain"] = delta.clip(lower=0); df["_loss"] = (-delta.clip(upper=0))
    gain = g("_gain", lambda s: s.rolling(24).mean())
    loss = g("_loss", lambda s: s.rolling(24).mean())
    df["rsi_24h"] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))
    df["hl_spread"] = (df["high"] - df["low"]) / df["close"]
    df["upper_shadow"] = (df["high"] - df["close"]) / df["close"]
    df["lower_shadow"] = (df["close"] - df["low"]) / df["close"]
    volma = g("volume", lambda s: s.rolling(24).mean())
    df["volume_ratio"] = df["volume"] / volma.replace(0, np.nan)
    df["volume_surge"] = (df["volume_ratio"] > 3).astype(int)
    df.drop(columns=["ret1", "_gain", "_loss"], inplace=True)

    # warmup 드롭
    df["_rank"] = df.groupby("seg").cumcount()
    df = df[df["_rank"] >= WARMUP].drop(columns="_rank")
    df = df.dropna(subset=["ret_6h", "vol_6h"]).reset_index(drop=True)
    df["coin"] = coin
    return df, n_wick, cut_time


def add_market_features(pooled):
    """BTC/ETH 공통 피처 (scale-free만: 수익률·변동성)"""
    def mk(sym, pre):
        d = pd.read_csv(os.path.join(RAW, f"binance_hourly_{sym}.csv"), parse_dates=["datetime"])
        d["datetime"] = d["datetime"].dt.tz_localize(None)
        d = d.sort_values("datetime").reset_index(drop=True)
        out = pd.DataFrame({"datetime": d["datetime"]})
        r1 = d["close"].pct_change(1)
        out[f"{pre}_ret_1h"] = r1
        out[f"{pre}_ret_24h"] = d["close"].pct_change(24)
        out[f"{pre}_vol_24h"] = r1.rolling(24).std()
        return out
    btc = mk("BTCUSDT", "btc"); eth = mk("ETHUSDT", "eth")
    pooled = pooled.merge(btc, on="datetime", how="left").merge(eth, on="datetime", how="left")
    return pooled


def main():
    parts = []
    print("=" * 64)
    for coin, (fname, mech) in COINS.items():
        p = os.path.join(RAW, fname)
        if not os.path.exists(p):
            print(f"{coin}: 파일 없음 → 스킵"); continue
        df, n_wick, cut = build_one(coin, fname)
        df["mech"] = mech
        parts.append(df)
        cutmsg = f", 좀비절단@{cut}" if cut is not None else ""
        print(f"{coin:6s}: {len(df):6,}행 | {df['datetime'].min().date()}~{df['datetime'].max().date()} "
              f"| depeg={int(df['depeg'].sum()):4d} | wick보정 {n_wick}{cutmsg}")

    pooled = pd.concat(parts, ignore_index=True)
    pooled = add_market_features(pooled)
    out = os.path.join(OUT_DIR, "df_multicoin_hourly.csv")
    pooled.to_csv(out, index=False)
    print(f"\npooled: {pooled.shape} → {out}")

    # 게이트: 코인×연도 양성 분포 (y_h6)
    print("\n[y_h6 양성 분포: 코인 × 연도]")
    d = pooled.dropna(subset=["y_h6"])
    tab = d.assign(year=d["datetime"].dt.year).pivot_table(
        index="coin", columns="year", values="y_h6", aggfunc="sum").fillna(0).astype(int)
    print(tab.to_string())


if __name__ == "__main__":
    main()
