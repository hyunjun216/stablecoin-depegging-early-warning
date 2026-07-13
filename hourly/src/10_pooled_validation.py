"""
10_pooled_validation.py — 멀티코인 pooled 검증

과제 정의 (onset 조건화):
  현재 페그 정상(depeg_t=0)인 시점에서 "6시간 내 디페깅 진입" 예측.
  → 이미 디페깅 중인 시점의 자명한 지속성 예측 제거 (만성 프리미엄 코인의 양성 범람 차단)

검증 2축:
  1) LOCO (leave-one-crisis-out): 위기 하나를 전 코인에서 시간窓째로 제외하고 학습
     → 그 위기에서 평가. 위기 6개: COVID(USDC), UST붕괴, SVB(USDC/DAI),
       TUSD_2024, FDUSD_2025, USDe_2025
  2) Cross-coin: 코인 자체를 학습에서 통째로 제외 → 그 코인의 위기에서 평가
     ("한 번도 본 적 없는 코인의 첫 위기를 잡는가" — 가장 강한 일반화 주장)

피처: scale-free 코어 + BTC/ETH (coin/mech 식별자는 피처 제외)
지표: AUC-PRC + base rate(리프트 해석), recall/FAR @ 경보예산 1%
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, recall_score
from xgboost import XGBClassifier

PROJECT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
PROC = os.path.join(PROJECT_DIR, "data", "processed")
ML_DIR = os.path.join(PROJECT_DIR, "data", "ml")

TARGET = "y_h6"
EMBARGO = pd.Timedelta("24h")
BUDGET = 0.01

FEATS = ["ret_1h", "ret_6h", "ret_24h", "vol_6h", "vol_24h", "vol_72h",
         "ma24_dev", "ma168_dev", "rsi_24h", "hl_spread", "upper_shadow", "lower_shadow",
         "volume_ratio", "volume_surge",
         "btc_ret_1h", "btc_ret_24h", "btc_vol_24h", "eth_ret_1h", "eth_ret_24h", "eth_vol_24h"]

CRISES = [
    # (이름, 시작, 끝, 피해 코인들)
    ("COVID_2020", "2020-03-05", "2020-03-25", ["USDC"]),
    ("UST_2022",   "2022-05-05", "2022-05-20", ["UST"]),
    ("SVB_2023",   "2023-02-15", "2023-04-15", ["USDC", "DAI"]),
    ("TUSD_2024",  "2024-01-10", "2024-02-10", ["TUSD"]),
    ("FDUSD_2025", "2025-03-28", "2025-04-12", ["FDUSD"]),
    ("USDe_2025",  "2025-10-08", "2025-10-18", ["USDE"]),
]


def make_xgb(spw):
    return XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8,
                         colsample_bytree=0.8, scale_pos_weight=spw, eval_metric="logloss",
                         random_state=42, verbosity=0, n_jobs=-1)


def far(y, pred):
    tn = int(((pred == 0) & (y == 0)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    return fp / (fp + tn) if (fp + tn) > 0 else 0.0


def fit_eval(tr, te, label):
    imp = SimpleImputer(strategy="median")
    Xtr = imp.fit_transform(tr[FEATS]); ytr = tr[TARGET].astype(int).values
    Xte = imp.transform(te[FEATS]); yte = te[TARGET].astype(int).values
    if yte.sum() == 0 or ytr.sum() < 5:
        return None
    m = make_xgb((ytr == 0).sum() / max(ytr.sum(), 1)).fit(Xtr, ytr)
    p_tr = m.predict_proba(Xtr)[:, 1]
    p_te = m.predict_proba(Xte)[:, 1]
    thr = float(np.quantile(p_tr, 1 - BUDGET))
    pred = (p_te >= thr).astype(int)
    base = yte.mean()
    ap = average_precision_score(yte, p_te)
    return {"test": label, "n_test": len(yte), "n_pos": int(yte.sum()),
            "base_rate": round(base, 4), "AUC_PRC": round(ap, 3),
            "lift": round(ap / base, 1) if base > 0 else None,
            "recall@1%": round(recall_score(yte, pred, zero_division=0), 3),
            "FAR@1%": round(far(yte, pred), 4)}


def main():
    df = pd.read_csv(os.path.join(PROC, "df_multicoin_hourly.csv"),
                     parse_dates=["datetime"], low_memory=False)
    df = df.replace([np.inf, -np.inf], np.nan)

    # onset 조건화: 현재 정상(depeg=0)인 행만 + 라벨 유효
    d = df[(df["depeg"] == 0)].dropna(subset=[TARGET]).reset_index(drop=True)
    print(f"pooled(onset 조건화): {len(d):,}행, 양성 {int(d[TARGET].sum())} "
          f"({d[TARGET].mean()*100:.2f}%)")
    tab = d.assign(year=d["datetime"].dt.year).pivot_table(
        index="coin", columns="year", values=TARGET, aggfunc="sum").fillna(0).astype(int)
    print("[onset 양성: 코인×연도]"); print(tab.to_string()); print()

    results = []

    # ── 1) LOCO ──
    print("=" * 70); print("  1) Leave-One-Crisis-Out (전 코인 pooled 학습)"); print("=" * 70)
    for name, s, e, coins in CRISES:
        s, e = pd.Timestamp(s), pd.Timestamp(e)
        tr = d[(d["datetime"] < s - EMBARGO) | (d["datetime"] > e + EMBARGO)]
        te_win = d[(d["datetime"] >= s) & (d["datetime"] <= e)]
        for coin in coins:
            te = te_win[te_win["coin"] == coin]
            r = fit_eval(tr, te, f"{name}/{coin}")
            if r:
                r["mode"] = "LOCO"; results.append(r)
                print(f"  {name}/{coin}: AUC-PRC={r['AUC_PRC']} (base {r['base_rate']}, "
                      f"lift x{r['lift']}), recall@1%={r['recall@1%']}, FAR={r['FAR@1%']} "
                      f"(양성 {r['n_pos']})")
            else:
                print(f"  {name}/{coin}: 양성 없음 → 스킵")

    # ── 2) Cross-coin (코인 통째 제외) ──
    print("\n" + "=" * 70); print("  2) Cross-coin (해당 코인을 학습에서 통째 제외)"); print("=" * 70)
    for name, s, e, coins in CRISES:
        s, e = pd.Timestamp(s), pd.Timestamp(e)
        for coin in coins:
            tr = d[(d["coin"] != coin) &
                   ((d["datetime"] < s - EMBARGO) | (d["datetime"] > e + EMBARGO))]
            te = d[(d["coin"] == coin) & (d["datetime"] >= s) & (d["datetime"] <= e)]
            r = fit_eval(tr, te, f"{name}/{coin}")
            if r:
                r["mode"] = "cross-coin"; results.append(r)
                print(f"  {name}/{coin} (코인 미학습): AUC-PRC={r['AUC_PRC']} "
                      f"(base {r['base_rate']}, lift x{r['lift']}), "
                      f"recall@1%={r['recall@1%']}, FAR={r['FAR@1%']} (양성 {r['n_pos']})")
            else:
                print(f"  {name}/{coin}: 양성 없음 → 스킵")

    pd.DataFrame(results).to_csv(os.path.join(ML_DIR, "multicoin_validation.csv"), index=False)
    print(f"\n저장: data/ml/multicoin_validation.csv")


if __name__ == "__main__":
    main()
