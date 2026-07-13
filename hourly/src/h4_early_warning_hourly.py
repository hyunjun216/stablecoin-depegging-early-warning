"""
h4_early_warning_hourly.py — 최적 구성(A/h6)의 SVB out-of-sample 탐지 시각화

- 2023(SVB) 제외 학습 → SVB 구간 예측 → 가격·P(depeg)·경보(1% 예산) 타임라인
- 2020 COVID, 2022 UST 구간도 참고 플롯
출력: outputs/ml/hourly_early_warning_*.png
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier

PROJECT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
PROC = os.path.join(PROJECT_DIR, "data", "processed")
OUT = os.path.join(PROJECT_DIR, "outputs", "ml")
os.makedirs(OUT, exist_ok=True)

TARGET = "y_tau05_h6"             # 최적: A / 6시간 시계
ALERT_BUDGET = 0.01
DROP_PREFIX = ("depeg_", "y_", "ymax_")
DROP_EXACT = {"datetime", "seg", "open", "high", "low", "close", "volume",
              "quote_vol", "n_trades", "btc_close", "eth_close", "dev", "b_"}
EMBARGO = pd.Timedelta("24h")


def feats_A(df):
    return [c for c in df.columns
            if c not in DROP_EXACT and not c.startswith(DROP_PREFIX)
            and not c.startswith("b_")]


def train_predict_window(df, feats, s, e):
    s, e = pd.Timestamp(s), pd.Timestamp(e)
    te = df[(df["datetime"] >= s) & (df["datetime"] <= e)].dropna(subset=[TARGET]).reset_index(drop=True)
    tr = df[(df["datetime"] < s - EMBARGO) | (df["datetime"] > e + EMBARGO)].dropna(subset=[TARGET])
    # 임계값: train 검증꼬리 1% 예산
    cut = int(len(tr) * 0.8)
    imp = SimpleImputer(strategy="median")
    Xf = imp.fit_transform(tr.iloc[:cut][feats])
    yf = tr.iloc[:cut][TARGET].astype(int)
    spw = (yf == 0).sum() / max(int(yf.sum()), 1)
    mv = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8,
                       colsample_bytree=0.8, scale_pos_weight=spw, eval_metric="logloss",
                       random_state=42, verbosity=0, n_jobs=-1).fit(Xf, yf)
    pv = mv.predict_proba(imp.transform(tr.iloc[cut:][feats]))[:, 1]
    # baseline(~0.001)과 위기(0.005~) 구분하는 운영점. 절대확률이 작아 하한 0.005 적용
    thr = max(float(np.quantile(pv, 1 - ALERT_BUDGET)), 0.005)
    # 전체 train 재학습 → test 예측
    imp2 = SimpleImputer(strategy="median")
    Xtr = imp2.fit_transform(tr[feats]); ytr = tr[TARGET].astype(int)
    spw2 = (ytr == 0).sum() / max(int(ytr.sum()), 1)
    m = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8,
                      colsample_bytree=0.8, scale_pos_weight=spw2, eval_metric="logloss",
                      random_state=42, verbosity=0, n_jobs=-1).fit(Xtr, ytr)
    te["proba"] = m.predict_proba(imp2.transform(te[feats]))[:, 1]
    return te, thr


def plot_event(te, thr, name):
    t = te["datetime"]
    alert = te["proba"] >= thr
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1.3]})

    # 상단: 가격 + 경보시간 음영
    ax1.plot(t, te["close"], color="#2C3E50", lw=1.3, label="USDC/USDT close")
    ax1.axhline(1.0, color="gray", ls=":", lw=0.8)
    ax1.axhline(0.995, color="#E67E22", ls="--", lw=0.8, label="peg ±0.5%")
    ax1.axhline(1.005, color="#E67E22", ls="--", lw=0.8)
    for i in np.where(alert.values)[0]:
        ax1.axvspan(t.iloc[i], t.iloc[min(i + 1, len(t) - 1)], color="#E74C3C", alpha=0.12)
    ax1.set_ylabel("USDC price")
    ax1.set_title(f"{name} — out-of-sample (trained excluding this event) | A/h6, 1% alert budget",
                  fontsize=12, fontweight="bold")
    ax1.legend(loc="lower right", fontsize=8)

    # 하단: P(depeg) 자동스케일 + 임계값
    ax2.fill_between(t, te["proba"], alpha=0.3, color="#3498DB")
    ax2.plot(t, te["proba"], color="#2980B9", lw=1.0, label="P(depeg within 6h)")
    ax2.axhline(thr, color="#E74C3C", ls="--", lw=1.2, label=f"alert thr={thr:.4f}")
    ax2.scatter(t[alert], te["proba"][alert], color="#E74C3C", s=8, zorder=5)
    ax2.set_ylabel("P(depeg≤6h)")
    ymax = max(float(te["proba"].max()) * 1.15, thr * 2)
    ax2.set_ylim(0, ymax)
    ax2.legend(loc="upper right", fontsize=8)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %Hh"))
    plt.xticks(rotation=45)
    plt.tight_layout()
    p = os.path.join(OUT, f"hourly_early_warning_{name}.png")
    plt.savefig(p, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  {name}: {p}  (경보 {int(alert.sum())}/{len(te)}시간, thr={thr:.4f})")


def main():
    df = pd.read_csv(os.path.join(PROC, "df_usdc_hourly.csv"), parse_dates=["datetime"])
    df = df.replace([np.inf, -np.inf], np.nan).sort_values("datetime").reset_index(drop=True)
    feats = feats_A(df)
    print(f"A 피처 {len(feats)}개, target={TARGET}")

    for name, s, e in [("SVB_2023", "2023-03-09", "2023-03-20"),
                       ("COVID_2020", "2020-03-08", "2020-03-25"),
                       ("UST_2022", "2022-05-08", "2022-05-16")]:
        te, thr = train_predict_window(df, feats, s, e)
        plot_event(te, thr, name)


if __name__ == "__main__":
    main()
