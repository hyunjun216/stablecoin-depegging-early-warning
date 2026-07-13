"""
h13_seed_ensemble.py — seed 앙상블 + 기하평균 실험

수상작 리버스엔지니어링 사이클 #2 적용 (2020 미래에셋 보험금 청구분류 1위 기법):
  ExtraTrees×3(seed만 다름)을 기하평균 (p1·p2·p3)^(1/3)으로 앙상블 → 극단 확률 보수화.
  여기서는 XGB×5 seed 앙상블을 동일 프로토콜(h2)로 검증한다.

비교 3군 (동일 데이터·피처·폴드, 결합 방식만 다름 = 애블레이션):
  1) single    : seed 42 단일 모델 (현 베이스라인)
  2) arithmean : 5-seed 산술평균
  3) geomean   : 5-seed 기하평균 (미래에셋 방식)

측정:
  - Walk-forward 연도별 + pooled AUC-PRC
  - Cross-event(COVID/UST/SVB) AUC-PRC, recall, FAR
  - seed 간 개별 AUC-PRC 표준편차 (단일모델 불안정성 → 앙상블 안정화 효과)

설정: 피처셋 A(코어), tau05, horizon=6h (헤드라인 스위트스팟)
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, recall_score
from xgboost import XGBClassifier

PROJECT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
PROC = os.path.join(PROJECT_DIR, "data", "processed")
ML_DIR = os.path.join(PROJECT_DIR, "data", "ml")

SEEDS = [42, 5187, 1217, 701, 2020]   # 42=기존 베이스라인, 나머지는 미래에셋 노트북 오마주
TAU_NAME, TAU, H = "tau05", 0.005, 6
TARGET = f"y_{TAU_NAME}_h{H}"
EMBARGO = pd.Timedelta("24h")
ALERT_BUDGET = 0.01
EPS = 1e-7                            # 기하평균에서 p=0이 곱을 죽이는 것 방지

DROP_PREFIX = ("depeg_", "y_", "ymax_")
DROP_EXACT = {"datetime", "seg", "open", "high", "low", "close", "volume",
              "quote_vol", "n_trades", "btc_close", "eth_close", "dev"}


def load():
    df = pd.read_csv(os.path.join(PROC, "df_usdc_hourly.csv"), parse_dates=["datetime"])
    return df.replace([np.inf, -np.inf], np.nan).sort_values("datetime").reset_index(drop=True)


def feature_cols(df):
    return [c for c in df.columns
            if c not in DROP_EXACT and not c.startswith(DROP_PREFIX)
            and not c.startswith("b_")]          # 피처셋 A(코어)만


def fit_seeds(Xtr, ytr, Xte):
    """seed별 XGB 5개 학습 → (n_te, n_seed) 확률 행렬 반환. imputer·spw는 공유."""
    imp = SimpleImputer(strategy="median")
    Xtr_i = imp.fit_transform(Xtr)
    Xte_i = imp.transform(Xte)
    n_pos, n_neg = int(ytr.sum()), int((ytr == 0).sum())
    spw = (n_neg / n_pos) if n_pos > 0 else 1
    probs = []
    for sd in SEEDS:
        m = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
                          eval_metric="logloss", random_state=sd, verbosity=0, n_jobs=-1)
        m.fit(Xtr_i, ytr)
        probs.append(m.predict_proba(Xte_i)[:, 1])
    return np.column_stack(probs)                # (n_te, 5)


def combine(P, mode):
    """seed 확률 행렬 → 단일 확률. single=seed42(0번째), arith=평균, geom=기하평균."""
    if mode == "single":
        return P[:, 0]
    if mode == "arithmean":
        return P.mean(axis=1)
    if mode == "geomean":
        return np.exp(np.log(np.clip(P, EPS, 1)).mean(axis=1))
    raise ValueError(mode)


def far(y, pred):
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    return fp / (fp + tn) if (fp + tn) > 0 else 0.0


def calib_thresholds(tr, feats):
    """train 시간순 꼬리 20%에서 결합방식별 경보예산 임계값 (h2.calibrate_threshold와 동일 사상)."""
    cut = int(len(tr) * 0.8)
    fit_part, val_part = tr.iloc[:cut], tr.iloc[cut:]
    if val_part[TARGET].notna().sum() < 50 or fit_part[TARGET].sum() < 3:
        fit_part, val_part = tr, tr              # 폴백: in-sample 분위수
    P = fit_seeds(fit_part[feats], fit_part[TARGET].astype(int), val_part[feats])
    return {m: float(np.quantile(combine(P, m), 1 - ALERT_BUDGET))
            for m in ["single", "arithmean", "geomean"]}


def eval_fold(P, yte, thr_by_mode, seed_stats=True):
    """한 폴드에서 3개 결합방식 + seed별 개별 AUC 산출."""
    out = {}
    for m in ["single", "arithmean", "geomean"]:
        p = combine(P, m)
        pred = (p >= thr_by_mode[m]).astype(int)
        out[m] = {"AUC_PRC": average_precision_score(yte, p),
                  "recall": recall_score(yte, pred, zero_division=0),
                  "FAR": far(yte, pred)}
    if seed_stats:
        seed_aps = [average_precision_score(yte, P[:, i]) for i in range(P.shape[1])]
        out["seed_ap_mean"] = float(np.mean(seed_aps))
        out["seed_ap_std"] = float(np.std(seed_aps))
    return out


def main():
    df = load()
    feats = feature_cols(df)
    print(f"데이터 {df.shape}, 피처 {len(feats)}개, target={TARGET}, seeds={SEEDS}\n")

    rows = []

    # ── 1. Walk-forward (연도별 확장윈도우, h2와 동일) ─────────────────────
    print("=" * 68)
    print("  [1] Walk-forward")
    print("=" * 68)
    pooled = {m: ([], []) for m in ["single", "arithmean", "geomean"]}
    for ty in [2021, 2022, 2023, 2024, 2025, 2026]:
        t0, t1 = pd.Timestamp(f"{ty}-01-01"), pd.Timestamp(f"{ty+1}-01-01")
        tr = df[df["datetime"] < (t0 - EMBARGO)].dropna(subset=[TARGET])
        te = df[(df["datetime"] >= t0) & (df["datetime"] < t1)].dropna(subset=[TARGET])
        if len(te) == 0 or tr[TARGET].sum() < 5 or te[TARGET].sum() == 0:
            continue
        P = fit_seeds(tr[feats], tr[TARGET].astype(int), te[feats])
        thr = calib_thresholds(tr, feats)
        yte = te[TARGET].astype(int).values
        r = eval_fold(P, yte, thr)
        print(f"  {ty} (n_pos={int(yte.sum())}): "
              f"single={r['single']['AUC_PRC']:.3f}  "
              f"arith={r['arithmean']['AUC_PRC']:.3f}  "
              f"geom={r['geomean']['AUC_PRC']:.3f}  "
              f"| seed AP {r['seed_ap_mean']:.3f}±{r['seed_ap_std']:.3f}")
        for m in ["single", "arithmean", "geomean"]:
            rows.append({"protocol": "walkforward", "fold": str(ty), "mode": m,
                         "n_pos": int(yte.sum()), **{k: round(v, 4) for k, v in r[m].items()},
                         "seed_ap_std": round(r["seed_ap_std"], 4)})
            pooled[m][0].append(yte)
            pooled[m][1].append(combine(P, m))

    print("\n  [pooled AUC-PRC]")
    for m in ["single", "arithmean", "geomean"]:
        Y = np.concatenate(pooled[m][0]); Pp = np.concatenate(pooled[m][1])
        ap = average_precision_score(Y, Pp)
        print(f"    {m:10s}: {ap:.4f}")
        rows.append({"protocol": "walkforward", "fold": "pooled", "mode": m,
                     "n_pos": int(Y.sum()), "AUC_PRC": round(ap, 4),
                     "recall": None, "FAR": None, "seed_ap_std": None})

    # ── 2. Cross-event 홀드아웃 (h2와 동일 구간) ──────────────────────────
    print("\n" + "=" * 68)
    print("  [2] Cross-event 홀드아웃")
    print("=" * 68)
    events = [("COVID_2020", "2020-03-05", "2020-03-25"),
              ("UST_2022", "2022-05-05", "2022-05-20"),
              ("SVB_2023", "2023-02-15", "2023-04-15")]
    for nm, st, en in events:
        s, e = pd.Timestamp(st), pd.Timestamp(en)
        te = df[(df["datetime"] >= s) & (df["datetime"] <= e)].dropna(subset=[TARGET])
        tr = df[(df["datetime"] < s - EMBARGO) | (df["datetime"] > e + EMBARGO)].dropna(subset=[TARGET])
        if len(te) == 0 or te[TARGET].sum() == 0:
            print(f"  {nm}: 양성 없음 → 스킵")
            continue
        P = fit_seeds(tr[feats], tr[TARGET].astype(int), te[feats])
        thr = calib_thresholds(tr, feats)
        yte = te[TARGET].astype(int).values
        r = eval_fold(P, yte, thr)
        print(f"  {nm} (n_pos={int(yte.sum())}):")
        for m in ["single", "arithmean", "geomean"]:
            print(f"    {m:10s}: AUC-PRC={r[m]['AUC_PRC']:.3f}  recall={r[m]['recall']:.3f}  FAR={r[m]['FAR']:.4f}")
            rows.append({"protocol": "crossevent", "fold": nm, "mode": m,
                         "n_pos": int(yte.sum()), **{k: round(v, 4) for k, v in r[m].items()},
                         "seed_ap_std": round(r["seed_ap_std"], 4)})
        print(f"    (seed 개별 AP: {r['seed_ap_mean']:.3f} ± {r['seed_ap_std']:.3f})")

    out = pd.DataFrame(rows)
    path = os.path.join(ML_DIR, "hourly_seed_ensemble.csv")
    out.to_csv(path, index=False)
    print(f"\n저장: data/ml/hourly_seed_ensemble.csv ({len(out)}행)")


if __name__ == "__main__":
    main()
