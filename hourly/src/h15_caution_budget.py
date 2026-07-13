"""
h15_caution_budget.py — 주의(Caution) 예산 축소의 트레이드오프 실측
walk-forward OOS(h12와 동일)에서 예산 5%/3%/2%별:
  - 위기 연도(2022, 2023) onset 양성 recall (미탐 증가 여부)
  - 평온 연도 주의 점등 시간 (오탐 감소 폭)
  - UST 2022 사전(pre-onset) 주의 유지 여부
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier

PROJECT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
PROC = os.path.join(PROJECT_DIR, "data", "processed")

FEATS = ["ret_1h", "ret_6h", "ret_24h", "vol_6h", "vol_24h", "vol_72h",
         "ma24_dev", "ma168_dev", "rsi_24h", "hl_spread", "upper_shadow", "lower_shadow",
         "volume_ratio", "volume_surge",
         "btc_ret_1h", "btc_ret_24h", "btc_vol_24h", "eth_ret_1h", "eth_ret_24h", "eth_vol_24h"]
EMB = pd.Timedelta("24h")
BUDGETS = [0.05, 0.03, 0.02]
UST_ONSET = pd.Timestamp("2022-05-12 03:00")


def main():
    df = pd.read_csv(os.path.join(PROC, "df_multicoin_hourly.csv"),
                     parse_dates=["datetime"], low_memory=False)
    u = df[df["coin"] == "USDC"].replace([np.inf, -np.inf], np.nan).sort_values("datetime").reset_index(drop=True)
    u["proba"] = np.nan
    thr = {}  # (year, budget) -> threshold

    for yr in range(2021, 2027):
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
        for b in BUDGETS:
            thr[(yr, b)] = float(np.quantile(p_tr, 1 - b))

    print(f"{'예산':>5} | {'2022 recall':>11} | {'2023 recall':>11} | "
          f"{'UST 사전주의':>10} | {'평온기 점등(24~26 합)':>18} | {'2021 점등':>9}")
    for b in BUDGETS:
        recalls = {}
        for yr in [2022, 2023]:
            g = u[(u["datetime"].dt.year == yr) & (u["depeg"] == 0)].dropna(subset=["y_h6"])
            pos = g[g["y_h6"] == 1]
            recalls[yr] = (pos["proba"] >= thr[(yr, b)]).mean() if len(pos) else np.nan
        # UST 사전 주의 (onset 이전 7일 내 첫 점등)
        pre = u[(u["datetime"] >= UST_ONSET - pd.Timedelta("7D")) & (u["datetime"] < UST_ONSET)]
        pre_c = pre[pre["proba"] >= thr[(2022, b)]]
        lead = (UST_ONSET - pre_c["datetime"].iloc[0]).total_seconds() / 3600 if len(pre_c) else None
        calm = sum(int((u[(u["datetime"].dt.year == yr)]["proba"] >= thr.get((yr, b), np.inf)).sum())
                   for yr in [2024, 2025, 2026])
        y21 = int((u[u["datetime"].dt.year == 2021]["proba"] >= thr[(2021, b)]).sum())
        lead_s = f"{lead:.0f}h 전" if lead else "없음"
        print(f"{b:>4.0%} | {recalls[2022]:>11.3f} | {recalls[2023]:>11.3f} | "
              f"{lead_s:>10} | {calm:>18,}h | {y21:>8,}h")


if __name__ == "__main__":
    main()
