"""
h10_timeline_viz.py — USDC 전체 기간 연도별 경보 타임라인

각 연도 패널: 가격(선) + 실제 디페깅(진한 빨강 밴드) + 경보(연빨강 배경) + 주의(주황 배경)
모델: USDC 단독 onset(depeg_t=0에서 6h 내 진입), 임계값 = 경보예산(주의 5%/경보 1%)
주의: 시각화용 전기간 학습(in-sample) — OOS 성능은 h2/h9에서 별도 검증됨(v2 s10과 동일 관행)
출력: outputs/ml/hourly_warning_timeline_by_year.png
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
from matplotlib.patches import Patch

from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier

PROJECT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
PROC = os.path.join(PROJECT_DIR, "data", "processed")
OUT = os.path.join(PROJECT_DIR, "outputs", "ml")

FEATS = ["ret_1h", "ret_6h", "ret_24h", "vol_6h", "vol_24h", "vol_72h",
         "ma24_dev", "ma168_dev", "rsi_24h", "hl_spread", "upper_shadow", "lower_shadow",
         "volume_ratio", "volume_surge",
         "btc_ret_1h", "btc_ret_24h", "btc_vol_24h", "eth_ret_1h", "eth_ret_24h", "eth_vol_24h"]


def main():
    df = pd.read_csv(os.path.join(PROC, "df_multicoin_hourly.csv"),
                     parse_dates=["datetime"], low_memory=False)
    u = df[df["coin"] == "USDC"].replace([np.inf, -np.inf], np.nan).sort_values("datetime").reset_index(drop=True)

    # onset 모델 학습 (depeg=0 행) → 전체 정상 시간에 확률
    d0 = u[(u["depeg"] == 0) & u["y_h6"].notna()]
    imp = SimpleImputer(strategy="median")
    X = imp.fit_transform(d0[FEATS]); y = d0["y_h6"].astype(int).values
    spw = (y == 0).sum() / max(y.sum(), 1)
    m = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8,
                      colsample_bytree=0.8, scale_pos_weight=spw, eval_metric="logloss",
                      random_state=42, verbosity=0, n_jobs=-1).fit(X, y)

    u["proba"] = np.nan
    mask0 = u["depeg"] == 0
    u.loc[mask0, "proba"] = m.predict_proba(imp.transform(u.loc[mask0, FEATS]))[:, 1]

    p0 = u.loc[mask0, "proba"].dropna()
    t_caution = float(np.quantile(p0, 0.95))
    t_alert = float(np.quantile(p0, 0.99))
    print(f"주의(상위5%)={t_caution:.5f}, 경보(상위1%)={t_alert:.5f}")

    years = sorted(u["datetime"].dt.year.unique())
    fig, axes = plt.subplots(len(years), 1, figsize=(16, 2.2 * len(years)))

    for ax, yr in zip(axes, years):
        g = u[u["datetime"].dt.year == yr]
        t = g["datetime"].values
        # 배경: 주의/경보/실제 디페깅
        caution = (g["proba"] >= t_caution) & (g["proba"] < t_alert)
        alert = g["proba"] >= t_alert
        depeg = g["depeg"] == 1
        for mask, color, alpha in [(caution, "#F39C12", 0.35), (alert, "#E74C3C", 0.45),
                                   (depeg, "#8E0000", 0.65)]:
            idx = np.where(mask.values)[0]
            for i in idx:
                if i + 1 < len(t):
                    ax.axvspan(t[i], t[i + 1], color=color, alpha=alpha, lw=0)
        ax.plot(g["datetime"], g["close"], color="#1B2631", lw=0.7)
        ax.axhline(1.0, color="gray", ls=":", lw=0.6)
        ax.axhline(0.995, color="#B9770E", ls="--", lw=0.5)
        ax.axhline(1.005, color="#B9770E", ls="--", lw=0.5)
        lo = min(0.993, float(g["close"].min()) - 0.002)
        hi = max(1.007, float(g["close"].max()) + 0.002)
        ax.set_ylim(lo, hi)
        ax.set_ylabel(str(yr), fontsize=11, fontweight="bold", rotation=0, labelpad=25, va="center")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.tick_params(labelsize=8)

    legend = [plt.Line2D([0], [0], color="#1B2631", lw=1, label="USDC close"),
              Patch(facecolor="#8E0000", alpha=0.65, label="Actual depeg"),
              Patch(facecolor="#E74C3C", alpha=0.45, label="Alert (top 1%)"),
              Patch(facecolor="#F39C12", alpha=0.35, label="Caution (top 5%)")]
    axes[0].legend(handles=legend, loc="lower right", fontsize=8, ncol=4)
    axes[0].set_title("USDC hourly — price, actual depegs, and model warnings by year "
                      "(onset model, in-sample visualization; OOS validated separately)",
                      fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(OUT, "hourly_warning_timeline_by_year.png")
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"저장: {path}")

    # 연도별 요약
    for yr in years:
        g = u[u["datetime"].dt.year == yr]
        print(f"  {yr}: depeg {int((g['depeg']==1).sum())}h, "
              f"경보 {int((g['proba']>=t_alert).sum())}h, "
              f"주의 {int(((g['proba']>=t_caution)&(g['proba']<t_alert)).sum())}h / {len(g)}h")


if __name__ == "__main__":
    main()
