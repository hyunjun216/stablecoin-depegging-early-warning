"""
h12_timeline_oos.py — walk-forward(OOS) 연도별 경보 타임라인
각 연도 = 그 이전 데이터로만 학습한 모델로 예측 (임계값도 train 분위수 → 전부 OOS).
2020은 이전 데이터 없음 → 경보 미표시(가격+실제 디페깅만).
출력: outputs/ml/hourly_warning_timeline_oos.png
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


def main():
    df = pd.read_csv(os.path.join(PROC, "df_multicoin_hourly.csv"),
                     parse_dates=["datetime"], low_memory=False)
    u = df[df["coin"] == "USDC"].replace([np.inf, -np.inf], np.nan).sort_values("datetime").reset_index(drop=True)
    u["proba"] = np.nan
    u["t_c"] = np.nan
    u["t_a"] = np.nan

    years = sorted(u["datetime"].dt.year.unique())
    for yr in years:
        if yr == years[0]:
            continue  # 2020: 이전 데이터 없음
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
        t_c, t_a = float(np.quantile(p_tr, 0.95)), float(np.quantile(p_tr, 0.99))
        mask = (u["datetime"].dt.year == yr) & (u["depeg"] == 0)
        u.loc[mask, "proba"] = m.predict_proba(imp.transform(u.loc[mask, FEATS]))[:, 1]
        u.loc[u["datetime"].dt.year == yr, "t_c"] = t_c
        u.loc[u["datetime"].dt.year == yr, "t_a"] = t_a

    fig, axes = plt.subplots(len(years), 1, figsize=(16, 2.2 * len(years)))
    for ax, yr in zip(axes, years):
        g = u[u["datetime"].dt.year == yr]
        t = g["datetime"].values
        caution = (g["proba"] >= g["t_c"]) & (g["proba"] < g["t_a"])
        alert = g["proba"] >= g["t_a"]
        depeg = g["depeg"] == 1
        for mask, color, alpha in [(caution, "#F39C12", 0.35), (alert, "#E74C3C", 0.5),
                                   (depeg, "#8E0000", 0.65)]:
            for i in np.where(mask.values)[0]:
                if i + 1 < len(t):
                    ax.axvspan(t[i], t[i + 1], color=color, alpha=alpha, lw=0)
        ax.plot(g["datetime"], g["close"], color="#1B2631", lw=0.7)
        ax.axhline(0.995, color="#B9770E", ls="--", lw=0.5)
        ax.axhline(1.005, color="#B9770E", ls="--", lw=0.5)
        lo = min(0.993, float(g["close"].min()) - 0.002)
        hi = max(1.007, float(g["close"].max()) + 0.002)
        ax.set_ylim(lo, hi)
        label = f"{yr}" + ("\n(no model)" if yr == years[0] else "")
        ax.set_ylabel(label, fontsize=10, fontweight="bold", rotation=0, labelpad=30, va="center")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.tick_params(labelsize=8)
        n_a = int((g["proba"] >= g["t_a"]).sum())
        n_c = int(((g["proba"] >= g["t_c"]) & (g["proba"] < g["t_a"])).sum())
        print(f"  {yr}: depeg {int(depeg.sum())}h | 경보 {n_a}h, 주의 {n_c}h / {len(g)}h")

    legend = [plt.Line2D([0], [0], color="#1B2631", lw=1, label="USDC close"),
              Patch(facecolor="#8E0000", alpha=0.65, label="Actual depeg"),
              Patch(facecolor="#E74C3C", alpha=0.5, label="Alert (OOS)"),
              Patch(facecolor="#F39C12", alpha=0.35, label="Caution (OOS)")]
    axes[0].legend(handles=legend, loc="lower right", fontsize=8, ncol=4)
    axes[0].set_title("USDC hourly — walk-forward OUT-OF-SAMPLE warnings by year "
                      "(each year predicted by a model trained only on prior years)",
                      fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(OUT, "hourly_warning_timeline_oos.png")
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"저장: {path}")


if __name__ == "__main__":
    main()
