"""
11_regime_pooling_onset.py — 레짐별 그룹 pooling + onset 소급 + 평온연도 오경보

A) 레짐 그룹: 급성군(USDC/UST/BUSD/USDP/FDUSD/USDE)만 pooling — 만성군(DAI/TUSD) 제외
   → 음의 전이 해소 여부. 비교: 단독 vs 급성군 vs 전체 pooled
B) onset 소급: USDC 단독 onset walk-forward(2023) — h2와 일관 비교
C) 평온연도 오경보: USDC 단독 onset 모델, 2021/2024/2025/2026에서 1% 예산 경보 수
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
EMB = pd.Timedelta("24h")
BUDGET = 0.01
ACUTE = ["USDC", "UST", "BUSD", "USDP", "FDUSD", "USDE"]
FEATS = ["ret_1h", "ret_6h", "ret_24h", "vol_6h", "vol_24h", "vol_72h",
         "ma24_dev", "ma168_dev", "rsi_24h", "hl_spread", "upper_shadow", "lower_shadow",
         "volume_ratio", "volume_surge",
         "btc_ret_1h", "btc_ret_24h", "btc_vol_24h", "eth_ret_1h", "eth_ret_24h", "eth_vol_24h"]


def make_xgb(spw):
    return XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8,
                         colsample_bytree=0.8, scale_pos_weight=spw, eval_metric="logloss",
                         random_state=42, verbosity=0, n_jobs=-1)


def run(tr, te, label):
    imp = SimpleImputer(strategy="median")
    Xtr = imp.fit_transform(tr[FEATS]); ytr = tr[TARGET].astype(int).values
    Xte = imp.transform(te[FEATS]); yte = te[TARGET].astype(int).values
    if yte.sum() == 0 or ytr.sum() < 5:
        print(f"  {label}: 스킵"); return None
    m = make_xgb((ytr == 0).sum() / max(ytr.sum(), 1)).fit(Xtr, ytr)
    p_tr = m.predict_proba(Xtr)[:, 1]; p = m.predict_proba(Xte)[:, 1]
    thr = float(np.quantile(p_tr, 1 - BUDGET))
    ap = average_precision_score(yte, p); base = yte.mean()
    rc = recall_score(yte, (p >= thr).astype(int), zero_division=0)
    print(f"  {label}: AUC-PRC={ap:.3f} (base {base:.4f}, lift x{ap/base:.1f}), "
          f"recall@1%={rc:.3f} (양성 {int(yte.sum())})")
    return {"test": label, "AUC_PRC": round(ap, 3), "base": round(base, 4),
            "lift": round(ap / base, 1), "recall@1%": round(rc, 3), "n_pos": int(yte.sum())}


def main():
    df = pd.read_csv(os.path.join(PROC, "df_multicoin_hourly.csv"),
                     parse_dates=["datetime"], low_memory=False)
    df = df.replace([np.inf, -np.inf], np.nan)
    d = df[df["depeg"] == 0].dropna(subset=[TARGET]).reset_index(drop=True)
    results = []

    # ── A) 레짐 그룹 pooling ──
    print("=" * 66); print("A) 급성군 pooling (만성 DAI/TUSD 제외)"); print("=" * 66)
    tests = [
        ("SVB_2023/USDC", "2023-02-15", "2023-04-15", "USDC", False),
        ("UST_2022/UST(미학습)", "2022-05-05", "2022-05-20", "UST", True),
        ("USDe_2025/USDE(미학습)", "2025-10-08", "2025-10-18", "USDE", True),
        ("FDUSD_2025/FDUSD(미학습)", "2025-03-28", "2025-04-12", "FDUSD", True),
        ("COVID_2020/USDC", "2020-03-05", "2020-03-25", "USDC", False),
    ]
    for label, s, e, coin, holdout_coin in tests:
        s, e = pd.Timestamp(s), pd.Timestamp(e)
        base_tr = d[d["coin"].isin(ACUTE)]
        if holdout_coin:
            base_tr = base_tr[base_tr["coin"] != coin]
        tr = base_tr[(base_tr["datetime"] < s - EMB) | (base_tr["datetime"] > e + EMB)]
        te = d[(d["coin"] == coin) & (d["datetime"] >= s) & (d["datetime"] <= e)]
        r = run(tr, te, f"급성군 | {label}")
        if r: r["mode"] = "acute_pool"; results.append(r)

    # ── B) onset 소급: USDC 단독 walk-forward 2023 ──
    print("\n" + "=" * 66); print("B) USDC 단독 onset walk-forward (2023)"); print("=" * 66)
    u = d[d["coin"] == "USDC"]
    t0 = pd.Timestamp("2023-01-01")
    tr = u[u["datetime"] < t0 - EMB]
    te = u[(u["datetime"] >= t0) & (u["datetime"] < pd.Timestamp("2024-01-01"))]
    r = run(tr, te, "USDC단독 | WF_2023(onset)")
    if r: r["mode"] = "single_onset"; results.append(r)

    # ── C) 평온연도 오경보 (USDC 단독, SVB 제외 학습 모델) ──
    print("\n" + "=" * 66); print("C) 평온연도 오경보 (USDC 단독 onset 모델, 1% 예산)"); print("=" * 66)
    s, e = pd.Timestamp("2023-02-15"), pd.Timestamp("2023-04-15")
    tr = u[(u["datetime"] < s - EMB) | (u["datetime"] > e + EMB)]
    tr2 = tr[tr["datetime"] < pd.Timestamp("2023-01-01")]  # 평온연도 평가를 위해 2023 이전만 학습
    imp = SimpleImputer(strategy="median")
    Xtr = imp.fit_transform(tr2[FEATS]); ytr = tr2[TARGET].astype(int).values
    m = make_xgb((ytr == 0).sum() / max(ytr.sum(), 1)).fit(Xtr, ytr)
    thr = float(np.quantile(m.predict_proba(Xtr)[:, 1], 1 - BUDGET))
    for yr in [2024, 2025, 2026]:
        cal = u[(u["datetime"] >= pd.Timestamp(f"{yr}-01-01")) &
                (u["datetime"] < pd.Timestamp(f"{yr+1}-01-01"))]
        if len(cal) == 0: continue
        p = m.predict_proba(imp.transform(cal[FEATS]))[:, 1]
        n_alert = int((p >= thr).sum())
        print(f"  {yr}: {len(cal):,}시간 중 경보 {n_alert}건 ({n_alert/len(cal)*100:.2f}%) "
              f"— 실제 디페깅 {int(cal[TARGET].sum())}건")
    # 2021도 (학습에 포함 안 되게 별도 모델: 2022+만 학습이 필요하나 간이로 2020 제외 불가 → 참고치 표기)
    print("  (2021은 학습기간에 포함되어 out-of-sample 아님 → 제외)")

    pd.DataFrame(results).to_csv(os.path.join(ML_DIR, "regime_pooling_results.csv"), index=False)
    print(f"\n저장: data/ml/regime_pooling_results.csv")


if __name__ == "__main__":
    main()
