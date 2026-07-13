"""
h11_event_zoom.py — 디페깅 에피소드 전후(±7일) 확대 비교
각 패널: 가격 + 실제 디페깅(진빨강) + 경보/주의 배경 + P(onset) 라인 + 진입 시점 수직선
모델: h10과 동일 (USDC 단독 onset, in-sample 시각화)
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

EVENTS = [("COVID #1", "2020-03-13 11:00"), ("COVID #2", "2020-03-17 22:00"),
          ("UST contagion", "2022-05-12 03:00"), ("SVB", "2023-03-12 14:00")]
WIN = pd.Timedelta("7D")


def main():
    df = pd.read_csv(os.path.join(PROC, "df_multicoin_hourly.csv"),
                     parse_dates=["datetime"], low_memory=False)
    u = df[df["coin"] == "USDC"].replace([np.inf, -np.inf], np.nan).sort_values("datetime").reset_index(drop=True)

    d0 = u[(u["depeg"] == 0) & u["y_h6"].notna()]
    imp = SimpleImputer(strategy="median")
    X = imp.fit_transform(d0[FEATS]); y = d0["y_h6"].astype(int).values
    m = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8,
                      colsample_bytree=0.8, scale_pos_weight=(y == 0).sum() / max(y.sum(), 1),
                      eval_metric="logloss", random_state=42, verbosity=0, n_jobs=-1).fit(X, y)
    u["proba"] = np.nan
    m0 = u["depeg"] == 0
    u.loc[m0, "proba"] = m.predict_proba(imp.transform(u.loc[m0, FEATS]))[:, 1]
    p0 = u.loc[m0, "proba"].dropna()
    t_c, t_a = float(np.quantile(p0, 0.95)), float(np.quantile(p0, 0.99))

    fig, axes = plt.subplots(len(EVENTS), 1, figsize=(15, 3.2 * len(EVENTS)))
    for ax, (name, onset) in zip(axes, EVENTS):
        o = pd.Timestamp(onset)
        g = u[(u["datetime"] >= o - WIN) & (u["datetime"] <= o + WIN)].reset_index(drop=True)
        t = g["datetime"].values
        caution = (g["proba"] >= t_c) & (g["proba"] < t_a)
        alert = g["proba"] >= t_a
        depeg = g["depeg"] == 1
        for mask, color, alpha in [(caution, "#F39C12", 0.30), (alert, "#E74C3C", 0.40),
                                   (depeg, "#8E0000", 0.55)]:
            for i in np.where(mask.values)[0]:
                if i + 1 < len(t):
                    ax.axvspan(t[i], t[i + 1], color=color, alpha=alpha, lw=0)
        ax.plot(g["datetime"], g["close"], color="#1B2631", lw=1.0)
        ax.axvline(o, color="#2C3E50", ls="-", lw=1.4, alpha=0.8)
        ax.axhline(0.995, color="#B9770E", ls="--", lw=0.6)
        ax.axhline(1.005, color="#B9770E", ls="--", lw=0.6)
        # 첫 경보/주의 → 진입까지 시간
        pre = g[g["datetime"] < o]
        lead_c = pre[pre["proba"] >= t_c]
        lead_a = pre[pre["proba"] >= t_a]
        lc = (o - lead_c["datetime"].iloc[0]).total_seconds() / 3600 if len(lead_c) else None
        la = (o - lead_a["datetime"].iloc[0]).total_seconds() / 3600 if len(lead_a) else None
        info = f"onset: {onset}  |  첫 주의 {lc:+.0f}h 전" if lc else f"onset: {onset}  |  사전 주의 없음"
        if la: info += f", 첫 경보 {la:+.0f}h 전"
        ax.set_title(f"{name} — {info}", fontsize=11, fontweight="bold")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        ax.tick_params(labelsize=8)
        print(f"{name}: 주의 lead={lc}, 경보 lead={la}")

    legend = [plt.Line2D([0], [0], color="#1B2631", lw=1, label="USDC close"),
              plt.Line2D([0], [0], color="#2C3E50", lw=1.4, label="depeg onset"),
              Patch(facecolor="#8E0000", alpha=0.55, label="Actual depeg"),
              Patch(facecolor="#E74C3C", alpha=0.40, label="Alert"),
              Patch(facecolor="#F39C12", alpha=0.30, label="Caution")]
    axes[0].legend(handles=legend, loc="lower left", fontsize=8, ncol=5)
    plt.tight_layout()
    path = os.path.join(OUT, "hourly_event_zoom_pre_post.png")
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"저장: {path}")


if __name__ == "__main__":
    main()
