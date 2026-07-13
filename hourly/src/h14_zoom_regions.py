"""
h14_zoom_regions.py — OOS 타임라인에서 사용자가 표시한 구간 확대
h12와 동일: 각 연도 = 이전 연도만으로 학습한 모델의 OOS 예측 (2020은 모델 없음)
출력: outputs/ml/hourly_zoom_regions.png
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
EMB = pd.Timedelta("24h")

WINDOWS = [
    ("2020-03 COVID (no model)", "2020-03-05", "2020-04-05"),
    ("2021-01 alert cluster", "2021-01-01", "2021-01-25"),
    ("2021-12", "2021-11-25", "2021-12-20"),
    ("2022-05 UST contagion", "2022-05-05", "2022-05-20"),
    ("2023-03 SVB", "2023-03-08", "2023-03-28"),
    ("2023-10~11", "2023-10-15", "2023-11-10"),
    ("2024-11~12", "2024-11-15", "2024-12-15"),
    ("2025-02~03", "2025-02-18", "2025-03-15"),
    ("2025-04", "2025-03-25", "2025-04-25"),
    ("2025-11~12", "2025-11-15", "2025-12-15"),
    ("2026-03", "2026-02-20", "2026-03-20"),
]


def main():
    df = pd.read_csv(os.path.join(PROC, "df_multicoin_hourly.csv"),
                     parse_dates=["datetime"], low_memory=False)
    u = df[df["coin"] == "USDC"].replace([np.inf, -np.inf], np.nan).sort_values("datetime").reset_index(drop=True)
    u["proba"] = np.nan; u["t_c"] = np.nan; u["t_a"] = np.nan

    years = sorted({pd.Timestamp(s).year for _, s, _ in WINDOWS} |
                   {pd.Timestamp(e).year for _, _, e in WINDOWS})
    for yr in years:
        if yr == 2020:
            continue
        t0 = pd.Timestamp(f"{yr}-01-01")
        tr = u[(u["datetime"] < t0 - EMB) & (u["depeg"] == 0)].dropna(subset=["y_h6"])
        if tr["y_h6"].sum() < 5:
            continue
        imp = SimpleImputer(strategy="median")
        Xtr = imp.fit_transform(tr[FEATS]); ytr = tr["y_h6"].astype(int).values
        m = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8,
                          colsample_bytree=0.8, scale_pos_weight=(ytr == 0).sum() / max(ytr.sum(), 1),
                          eval_metric="logloss", random_state=42, verbosity=0, n_jobs=-1).fit(Xtr, ytr)
        p_tr = m.predict_proba(Xtr)[:, 1]
        mask = (u["datetime"].dt.year == yr) & (u["depeg"] == 0)
        u.loc[mask, "proba"] = m.predict_proba(imp.transform(u.loc[mask, FEATS]))[:, 1]
        u.loc[u["datetime"].dt.year == yr, "t_c"] = float(np.quantile(p_tr, 0.95))
        u.loc[u["datetime"].dt.year == yr, "t_a"] = float(np.quantile(p_tr, 0.99))

    ncol = 2
    nrow = int(np.ceil(len(WINDOWS) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(16, 2.6 * nrow))
    axes = axes.flatten()

    for ax, (name, s, e) in zip(axes, WINDOWS):
        s, e = pd.Timestamp(s), pd.Timestamp(e)
        g = u[(u["datetime"] >= s) & (u["datetime"] <= e)].reset_index(drop=True)
        if len(g) == 0:
            ax.set_visible(False); continue
        t = g["datetime"].values
        caution = (g["proba"] >= g["t_c"]) & (g["proba"] < g["t_a"])
        alert = g["proba"] >= g["t_a"]
        depeg = g["depeg"] == 1
        for mask, color, alpha in [(caution, "#F39C12", 0.35), (alert, "#E74C3C", 0.5),
                                   (depeg, "#8E0000", 0.6)]:
            for i in np.where(mask.values)[0]:
                if i + 1 < len(t):
                    ax.axvspan(t[i], t[i + 1], color=color, alpha=alpha, lw=0)
        ax.plot(g["datetime"], g["close"], color="#1B2631", lw=0.9)
        ax.axhline(0.995, color="#B9770E", ls="--", lw=0.6)
        ax.axhline(1.005, color="#B9770E", ls="--", lw=0.6)
        ax.axhline(1.0, color="gray", ls=":", lw=0.5)
        n_a, n_c, n_d = int(alert.sum()), int(caution.sum()), int(depeg.sum())
        ax.set_title(f"{name}  |  depeg {n_d}h · Alert {n_a}h · Caution {n_c}h",
                     fontsize=10, fontweight="bold")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        ax.tick_params(labelsize=7)
        print(f"  {name}: depeg {n_d}h, Alert {n_a}h, Caution {n_c}h, 종가범위 "
              f"{g['close'].min():.4f}~{g['close'].max():.4f}")
    for ax in axes[len(WINDOWS):]:
        ax.set_visible(False)

    legend = [plt.Line2D([0], [0], color="#1B2631", lw=1, label="USDC close"),
              Patch(facecolor="#8E0000", alpha=0.6, label="Actual depeg"),
              Patch(facecolor="#E74C3C", alpha=0.5, label="Alert (OOS)"),
              Patch(facecolor="#F39C12", alpha=0.35, label="Caution (OOS)")]
    axes[0].legend(handles=legend, loc="lower left", fontsize=7, ncol=2)
    plt.tight_layout()
    path = os.path.join(OUT, "hourly_zoom_regions.png")
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"저장: {path}")


if __name__ == "__main__":
    main()
