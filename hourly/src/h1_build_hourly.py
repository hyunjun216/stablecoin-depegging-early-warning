"""
h1_build_hourly.py — 시간단위(1h) USDC 데이터셋 빌드

입력: data/collect/raw/binance_hourly_{USDCUSDT,BTCUSDT,ETHUSDT}.csv
출력: data/processed/df_usdc_hourly.csv

핵심 처리:
  1) 나쁜 wick 정제 — 종가는 정상인데 저가/고가만 극단(나쁜 프린트) 제거
  2) 타겟은 종가 기준 (dev = |close-1|) — wick에 강건. v2의 TP 대신 hourly 노이즈 대응
  3) 세그먼트 분할 — 2022-09~2023-03 BUSD 자동전환 166일 결측 블록을 rolling이 넘나들지 않게
  4) 다중 시계(1h/6h/24h) 전방 라벨: 이진 y^H + 연속 max(dev)
  5) A 코어 피처(hourly 네이티브) + B 일별 ffill 피처(애블레이션용, b_ 접두사)
"""

import pandas as pd
import numpy as np
import os
from pandas.api.indexers import FixedForwardWindowIndexer

PROJECT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
RAW_HOURLY = os.path.join(PROJECT_DIR, "data", "collect", "raw")
RAW_DAILY = os.path.join(PROJECT_DIR, "data", "raw")
OUT_DIR = os.path.join(PROJECT_DIR, "data", "processed")
os.makedirs(OUT_DIR, exist_ok=True)

HORIZONS = [1, 6, 24]
TAUS = {"tau05": 0.005, "tau10": 0.01}
PERSIST_N, PERSIST_K = 3, 2       # 2-of-3 연속시간 persistence
GAP = pd.Timedelta("1h")
WARMUP = 6                         # 세그먼트 시작 후 단기피처 미성숙 구간만 드롭(긴 창은 h2 imputer가 처리)


# ── 1. 로딩 & 병합 ──────────────────────────────────────────────────────────
def load_binance():
    def rd(sym):
        d = pd.read_csv(os.path.join(RAW_HOURLY, f"binance_hourly_{sym}.csv"),
                        parse_dates=["datetime"])
        d["datetime"] = d["datetime"].dt.tz_localize(None)  # UTC naive
        return d

    usdc = rd("USDCUSDT").rename(columns={
        "open": "open", "high": "high", "low": "low", "close": "close",
        "volume": "volume", "quote_volume": "quote_vol", "n_trades": "n_trades"})
    btc = rd("BTCUSDT")[["datetime", "close", "volume"]].rename(
        columns={"close": "btc_close", "volume": "btc_vol"})
    eth = rd("ETHUSDT")[["datetime", "close", "volume"]].rename(
        columns={"close": "eth_close", "volume": "eth_vol"})

    df = usdc.merge(btc, on="datetime", how="left").merge(eth, on="datetime", how="left")
    return df.sort_values("datetime").reset_index(drop=True)


# ── 2. 나쁜 wick 정제 ────────────────────────────────────────────────────────
def clean_wicks(df):
    """종가는 페그 근처인데 저가/고가만 극단 → 나쁜 프린트로 판정해 OHLC 정합값으로 대체.
       진짜 디페깅(종가도 이탈, 예: SVB close=0.9129)은 보존."""
    bad_low = (df["low"] < 0.90) & (df["close"] > 0.97)
    bad_high = (df["high"] > 1.10) & (df["close"] < 1.03)
    df.loc[bad_low, "low"] = df.loc[bad_low, ["open", "close"]].min(axis=1)
    df.loc[bad_high, "high"] = df.loc[bad_high, ["open", "close"]].max(axis=1)
    print(f"  나쁜 wick 정제: 저가 {int(bad_low.sum())}건, 고가 {int(bad_high.sum())}건")
    return df


# ── 3. 세그먼트 분할 (결측 블록 경계) ────────────────────────────────────────
def add_segments(df):
    gap = df["datetime"].diff() > GAP
    df["seg"] = gap.cumsum()
    sizes = df.groupby("seg").size()
    print(f"  세그먼트 {df['seg'].nunique()}개 (최대 {sizes.max()}행, 최소 {sizes.min()}행)")
    return df


def g_transform(df, col, fn):
    """세그먼트 내부에서만 시계열 변환 (결측 블록 넘나들기 방지)"""
    return df.groupby("seg")[col].transform(fn)


# ── 4. 타겟 ─────────────────────────────────────────────────────────────────
def make_targets(df):
    df["dev"] = (df["close"] - 1.0).abs()

    for name, tau in TAUS.items():
        raw = (df["dev"] > tau).astype(int)
        # 2-of-3 연속시간 persistence (과거방향, 세그먼트 내)
        rs = df.assign(_r=raw).groupby("seg")["_r"].transform(
            lambda s: s.rolling(PERSIST_N, min_periods=1).sum())
        df[f"depeg_{name}"] = (rs >= PERSIST_K).astype(int)

    # 전방 라벨 (세그먼트 내, (t, t+H] 구간)
    fwd = {H: FixedForwardWindowIndexer(window_size=H) for H in HORIZONS}
    for name in TAUS:
        d = df[f"depeg_{name}"]
        for H in HORIZONS:
            # (t, t+H] = shift(-1) 후 향후 H개 max
            df[f"y_{name}_h{H}"] = (
                df.assign(_d=d).groupby("seg")["_d"]
                  .transform(lambda s: s.shift(-1).rolling(fwd[H], min_periods=1).max())
            )
    # 연속 보조 타겟: (t, t+H] 최대 이탈
    for H in HORIZONS:
        df[f"ymax_dev_h{H}"] = (
            df.groupby("seg")["dev"]
              .transform(lambda s: s.shift(-1).rolling(fwd[H], min_periods=1).max())
        )
    return df


# ── 5. A 코어 피처 (hourly 네이티브) ─────────────────────────────────────────
def make_core_features(df):
    # 수익률
    for n in [1, 6, 24]:
        df[f"ret_{n}h"] = g_transform(df, "close", lambda s: s.pct_change(n))
    # 변동성 (1h 수익률의 롤링 std)
    df["ret1"] = df["ret_1h"]
    for w in [6, 24, 72]:
        df[f"vol_{w}h"] = g_transform(df, "ret1", lambda s: s.rolling(w).std())
    # 이동평균 이탈
    for w in [24, 168]:
        ma = g_transform(df, "close", lambda s: s.rolling(w).mean())
        df[f"ma{w}_dev"] = df["close"] - ma
    # RSI(24h)
    delta = g_transform(df, "close", lambda s: s.diff())
    df["_gain"] = delta.clip(lower=0)
    df["_loss"] = (-delta.clip(upper=0))
    gain = g_transform(df, "_gain", lambda s: s.rolling(24).mean())
    loss = g_transform(df, "_loss", lambda s: s.rolling(24).mean())
    rs = gain / loss.replace(0, np.nan)
    df["rsi_24h"] = 100 - (100 / (1 + rs))
    # 캔들 구조 (정제된 OHLC)
    df["hl_spread"] = (df["high"] - df["low"]) / df["close"]
    df["upper_shadow"] = (df["high"] - df["close"]) / df["close"]
    df["lower_shadow"] = (df["close"] - df["low"]) / df["close"]
    # 거래량
    volma = g_transform(df, "volume", lambda s: s.rolling(24).mean())
    df["volume_ratio"] = df["volume"] / volma.replace(0, np.nan)
    df["volume_surge"] = (df["volume_ratio"] > 3).astype(int)
    df["log_qvol"] = np.log1p(df["quote_vol"])
    df["log_ntrades"] = np.log1p(df["n_trades"])
    # BTC/ETH 암호시장
    for a in ["btc", "eth"]:
        df[f"{a}_ret_1h"] = g_transform(df, f"{a}_close", lambda s: s.pct_change(1))
        df[f"{a}_ret_24h"] = g_transform(df, f"{a}_close", lambda s: s.pct_change(24))
        df[f"{a}_vol_24h"] = g_transform(df, f"{a}_ret_1h", lambda s: s.rolling(24).std())
    df.drop(columns=["ret1", "_gain", "_loss"], inplace=True)
    return df


# ── 6. B 일별 ffill 피처 (애블레이션용) ──────────────────────────────────────
def make_daily_features(df):
    """기존 일별 raw를 hourly 격자에 merge_asof(backward=ffill). b_ 접두사."""
    df = df.sort_values("datetime").reset_index(drop=True)
    specs = [
        ("macro_data.csv", ["vix", "dxy", "federal_funds_rate"]),
        ("macro_additional.csv", ["us_10y_yield", "yield_spread_2s10s", "credit_spread", "m2_supply"]),
        ("fear_and_greed_index.csv", None),   # value
        ("onchain_supply.csv", ["supply_change_USDC", "circ_USDC"]),
        ("gas_price.csv", ["gas_price_gwei"]),
        ("defi_protocols_tvl.csv", ["lending_tvl_total"]),
        ("google_trends.csv", ["gt_usdc_depeg", "gt_stablecoin_crash"]),
    ]
    for fname, cols in specs:
        p = os.path.join(RAW_DAILY, fname)
        if not os.path.exists(p):
            continue
        d = pd.read_csv(p)
        dcol = "timestamp" if "timestamp" in d.columns else "Date"
        d["datetime"] = pd.to_datetime(d[dcol]).dt.tz_localize(None)
        if fname.startswith("fear"):
            d = d[["datetime", "value"]].rename(columns={"value": "b_fgi"})
        else:
            use = [c for c in (cols or []) if c in d.columns]
            if not use:
                continue
            d = d[["datetime"] + use].rename(columns={c: f"b_{c}" for c in use})
        d = d.dropna(subset=["datetime"]).sort_values("datetime")
        df = pd.merge_asof(df, d, on="datetime", direction="backward")
    bcols = [c for c in df.columns if c.startswith("b_")]
    print(f"  B 일별 피처 {len(bcols)}개 병합: {bcols}")
    return df


# ── 7. warmup 드롭 & 저장 ────────────────────────────────────────────────────
def finalize(df):
    # 세그먼트별 앞 WARMUP행(단기피처 미성숙) 제거
    df["_rank"] = df.groupby("seg").cumcount()
    df = df[df["_rank"] >= WARMUP].drop(columns="_rank")
    # 단기 코어 피처만 결측 제거 (긴 창 피처의 NaN은 h2 imputer가 채움)
    df = df.dropna(subset=["ret_6h", "vol_6h"]).reset_index(drop=True)
    return df


def main():
    print("=" * 60)
    print("h1: 시간단위 USDC 데이터셋 빌드")
    print("=" * 60)

    print("[1] Binance hourly 로딩·병합")
    df = load_binance()
    print(f"    {df.shape}, {df['datetime'].min()} ~ {df['datetime'].max()}")

    print("[2] 나쁜 wick 정제")
    df = clean_wicks(df)

    print("[3] 세그먼트 분할")
    df = add_segments(df)

    print("[4] 타겟 생성 (종가 기준, 다중 시계)")
    df = make_targets(df)

    print("[5] A 코어 피처")
    df = make_core_features(df)

    print("[6] B 일별 ffill 피처")
    df = make_daily_features(df)

    print("[7] warmup 드롭 & 정리")
    df = finalize(df)

    out = os.path.join(OUT_DIR, "df_usdc_hourly.csv")
    df.to_csv(out, index=False)
    print(f"\n저장: {out}  shape={df.shape}")

    # ── 게이트: 타겟 분포 확인 ──
    print("\n" + "=" * 60)
    print("타겟 분포 (walk-forward 가능성 게이트)")
    print("=" * 60)
    for name in TAUS:
        base = df[f"depeg_{name}"]
        print(f"\n[{name} (±{TAUS[name]*100:.1f}%)] 시점 디페깅={int(base.sum())} ({base.mean()*100:.2f}%)")
        yr = df.assign(year=df["datetime"].dt.year)
        for H in HORIZONS:
            col = f"y_{name}_h{H}"
            byyr = yr.groupby("year")[col].sum().astype(int)
            print(f"  y_h{H}: 양성 {int(df[col].sum())} | 연도별 {byyr.to_dict()}")


if __name__ == "__main__":
    main()
