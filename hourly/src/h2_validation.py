"""
h2_validation.py — 시계열 검증 프로토콜 (XGBoost)

두 축:
  1) Walk-forward: 연도별 확장윈도우(expanding), embargo=24h, fold별 AUC-PRC/recall/FAR + pooled
  2) Cross-event: 특정 위기구간 홀드아웃 → AUC-PRC/recall/FAR + lead time
     - SVB 2023 (헤드라인, 단 Binance 결측으로 pre-onset 제한)
     - UST 2022-05 (클린), COVID 2020-03 (클린)

애블레이션: A(코어) vs A+B(일별 ffill) 동일 프로토콜 비교.
타겟: tau05 메인, 시계 1h/6h/24h. dev/close 등 타겟기반 컬럼은 피처 제외(누수 방지).
"""

import pandas as pd
import numpy as np
import os
import sys
import warnings
warnings.filterwarnings("ignore")

from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, recall_score, precision_score
from xgboost import XGBClassifier

PROJECT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
PROC = os.path.join(PROJECT_DIR, "data", "processed")
ML_DIR = os.path.join(PROJECT_DIR, "data", "ml")
os.makedirs(ML_DIR, exist_ok=True)

RANDOM_STATE = 42
# 사용법: python h2_validation.py [tau05|tau10]  (기본 tau05)
TAU_NAME = sys.argv[1] if len(sys.argv) > 1 else "tau05"
TAU = {"tau05": 0.005, "tau10": 0.01}[TAU_NAME]
HORIZONS = [1, 6, 24]
EMBARGO = pd.Timedelta("24h")     # 라벨 전방참조(최대 24h) 겹침 차단

# 피처에서 제외할 컬럼 (타겟 기반·원본레벨·식별자)
DROP_PREFIX = ("depeg_", "y_", "ymax_")
DROP_EXACT = {"datetime", "seg", "open", "high", "low", "close", "volume",
              "quote_vol", "n_trades", "btc_close", "eth_close", "dev"}


def fbeta(y, p, beta=2.0):
    pr = precision_score(y, p, zero_division=0)
    rc = recall_score(y, p, zero_division=0)
    return 0.0 if pr + rc == 0 else (1 + beta**2) * pr * rc / (beta**2 * pr + rc)


def load():
    df = pd.read_csv(os.path.join(PROC, "df_usdc_hourly.csv"), parse_dates=["datetime"])
    df = df.replace([np.inf, -np.inf], np.nan).sort_values("datetime").reset_index(drop=True)
    return df


def feature_cols(df, use_b):
    cols = [c for c in df.columns
            if c not in DROP_EXACT and not c.startswith(DROP_PREFIX)]
    if not use_b:
        cols = [c for c in cols if not c.startswith("b_")]
    return cols


ALERT_BUDGET = 0.01               # 오경보 예산 1% (운영점: 시간의 상위 1%만 경보)


def _xgb(spw):
    return XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
                         eval_metric="logloss", random_state=RANDOM_STATE,
                         verbosity=0, n_jobs=-1)


def fit_predict(Xtr, ytr, Xte):
    imp = SimpleImputer(strategy="median")
    Xtr = imp.fit_transform(Xtr)
    Xte = imp.transform(Xte)
    n_pos, n_neg = int(ytr.sum()), int((ytr == 0).sum())
    spw = (n_neg / n_pos) if n_pos > 0 else 1
    m = _xgb(spw)
    m.fit(Xtr, ytr)
    return m.predict_proba(Xte)[:, 1], imp, m


def calibrate_threshold(tr, feats, target, alert_rate=ALERT_BUDGET):
    """시간순 검증꼬리(train 마지막 20%)에서 경보예산 분위수로 임계값 결정.
       과적합된 in-sample 임계값 대신, 최근 정상데이터의 (1-alert_rate) 분위수 사용."""
    n = len(tr)
    cut = int(n * 0.8)
    fit_part, val_part = tr.iloc[:cut], tr.iloc[cut:]
    if val_part[target].notna().sum() < 50 or fit_part[target].sum() < 3:
        # 검증꼬리 부족 → 전체 train in-sample 분위수로 폴백
        p, _, _ = fit_predict(tr[feats], tr[target].astype(int), tr[feats])
        return float(np.quantile(p, 1 - alert_rate))
    pv, _, _ = fit_predict(fit_part[feats], fit_part[target].astype(int), val_part[feats])
    return float(np.quantile(pv, 1 - alert_rate))


def far(y, pred):
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    return fp / (fp + tn) if (fp + tn) > 0 else 0.0


# ── 1. Walk-forward (연도별 확장윈도우) ──────────────────────────────────────
def walk_forward(df, feats, target):
    years = [2021, 2022, 2023, 2024, 2025, 2026]
    rows, pooled_y, pooled_p = [], [], []
    for ty in years:
        t0 = pd.Timestamp(f"{ty}-01-01")
        t1 = pd.Timestamp(f"{ty+1}-01-01")
        tr = df[df["datetime"] < (t0 - EMBARGO)].dropna(subset=[target])
        te = df[(df["datetime"] >= t0) & (df["datetime"] < t1)].dropna(subset=[target])
        if len(te) == 0 or tr[target].sum() < 5:
            continue
        p, _, _ = fit_predict(tr[feats], tr[target].astype(int), te[feats])
        yte = te[target].astype(int).values
        npos = int(yte.sum())
        if npos > 0:
            ap = average_precision_score(yte, p)
            thr = calibrate_threshold(tr, feats, target)
            pred = (p >= thr).astype(int)
            rows.append({"test_year": ty, "n_test": len(te), "n_pos": npos,
                         "AUC_PRC": round(ap, 3), "recall": round(recall_score(yte, pred, zero_division=0), 3),
                         "FAR": round(far(yte, pred), 4), "thr": round(thr, 3)})
            pooled_y.append(yte); pooled_p.append(p)
        else:
            rows.append({"test_year": ty, "n_test": len(te), "n_pos": 0,
                         "AUC_PRC": None, "recall": None, "FAR": None, "thr": None})
    pooled_ap = None
    if pooled_y:
        Y = np.concatenate(pooled_y); P = np.concatenate(pooled_p)
        pooled_ap = average_precision_score(Y, P)
    return pd.DataFrame(rows), pooled_ap


# ── 2. Cross-event 홀드아웃 ──────────────────────────────────────────────────
def cross_event(df, feats, target, start, end, name):
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    te = df[(df["datetime"] >= s) & (df["datetime"] <= e)].dropna(subset=[target])
    tr = df[(df["datetime"] < s - EMBARGO) | (df["datetime"] > e + EMBARGO)].dropna(subset=[target])
    if len(te) == 0 or te[target].sum() == 0:
        return None
    thr = calibrate_threshold(tr, feats, target)
    _, imp, m = fit_predict(tr[feats], tr[target].astype(int), tr[feats])
    p = m.predict_proba(imp.transform(te[feats]))[:, 1]
    yte = te[target].astype(int).values
    pred = (p >= thr).astype(int)
    ap = average_precision_score(yte, p)

    # lead time: 가격이 처음 페그 이탈(|close-1|>TAU)한 시각 대비 첫 경보 시각
    te = te.reset_index(drop=True)
    onset_idx = te.index[(te["close"] - 1).abs() > TAU]
    lead = None
    if len(onset_idx) > 0:
        onset_t = te.loc[onset_idx[0], "datetime"]
        alert_idx = te.index[p >= thr]
        if len(alert_idx) > 0:
            first_alert_t = te.loc[alert_idx[0], "datetime"]
            lead = (onset_t - first_alert_t).total_seconds() / 3600.0  # +면 사전경보
    return {"event": name, "window": f"{start}~{end}", "n_test": len(te),
            "n_pos": int(yte.sum()), "AUC_PRC": round(ap, 3),
            "recall": round(recall_score(yte, pred, zero_division=0), 3),
            "FAR": round(far(yte, pred), 4), "lead_time_h": None if lead is None else round(lead, 1),
            "thr": round(thr, 3)}


def main():
    df = load()
    print(f"데이터: {df.shape}, {df['datetime'].min()} ~ {df['datetime'].max()}\n")

    events = [
        ("COVID_2020", "2020-03-05", "2020-03-25"),
        ("UST_2022",   "2022-05-05", "2022-05-20"),
        ("SVB_2023",   "2023-02-15", "2023-04-15"),
    ]

    all_wf, all_ce = [], []
    for use_b in [False, True]:
        tag = "A+B" if use_b else "A"
        feats = feature_cols(df, use_b)
        print("=" * 64)
        print(f"  피처셋 {tag} ({len(feats)}개)")
        print("=" * 64)
        for H in HORIZONS:
            target = f"y_{TAU_NAME}_h{H}"
            print(f"\n── target={target} ──")
            wf, pooled = walk_forward(df, feats, target)
            print("  [Walk-forward] 연도별:")
            print(wf.to_string(index=False))
            print(f"  pooled AUC-PRC = {None if pooled is None else round(pooled,3)}")
            wf["featset"], wf["horizon"] = tag, H
            all_wf.append(wf)

            print("  [Cross-event]:")
            for nm, st, en in events:
                r = cross_event(df, feats, target, st, en, nm)
                if r:
                    r["featset"], r["horizon"] = tag, H
                    all_ce.append(r)
                    print(f"    {nm}: AUC-PRC={r['AUC_PRC']}, recall={r['recall']}, "
                          f"FAR={r['FAR']}, lead={r['lead_time_h']}h (n_pos={r['n_pos']})")

    suffix = "" if TAU_NAME == "tau05" else f"_{TAU_NAME}"
    pd.concat(all_wf).to_csv(os.path.join(ML_DIR, f"hourly_walkforward{suffix}.csv"), index=False)
    pd.DataFrame(all_ce).to_csv(os.path.join(ML_DIR, f"hourly_crossevent{suffix}.csv"), index=False)
    print(f"\n저장: data/ml/hourly_walkforward{suffix}.csv, hourly_crossevent{suffix}.csv")


if __name__ == "__main__":
    main()
