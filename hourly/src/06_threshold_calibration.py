"""
06_threshold_calibration.py — 경보 임계값 보정 (문헌 표준 방법)

문제: XGB 원확률이 희소사건이라 작고 고정 임계값이 자의적
      (arXiv:2512.00916: 저확률서 F1/AUPRC 최적임계값이 0/1로 퇴화 — 알려진 현상).
해법:
  1) sigmoid(Platt) 확률 캘리브레이션 — 단조변환이라 랭킹(AUC-PRC) 보존하며 확률값 해석가능화
     (isotonic은 양성 소수라 계단함수로 랭킹을 깨뜨려 제외)
  2) 경보예산(alert budget) 기반 이중 임계값 (CALIBURN류 운영점):
       - 주의(Caution): train 전체분포 상위 5% (평균적으로 5% 시간만 주의)
       - 경보(Alert):   상위 1%
     → 음성 분위수(0으로 퇴화)가 아니라 '경보 예산'에서 역산 → 비퇴화·해석가능
  3) SVB out-of-sample 3단계 평가 + lead time

최적 구성: A(코어) / h6 / XGBoost.
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
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score, brier_score_loss, recall_score
from xgboost import XGBClassifier

PROJECT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
PROC = os.path.join(PROJECT_DIR, "data", "processed")
ML_DIR = os.path.join(PROJECT_DIR, "data", "ml")
OUT = os.path.join(PROJECT_DIR, "outputs", "ml")
os.makedirs(OUT, exist_ok=True)

TARGET = "y_tau05_h6"
BUDGET_CAUTION = 0.05   # 주의: 시간의 상위 5%
BUDGET_ALERT = 0.01     # 경보: 상위 1%
DROP_PREFIX = ("depeg_", "y_", "ymax_")
DROP_EXACT = {"datetime", "seg", "open", "high", "low", "close", "volume",
              "quote_vol", "n_trades", "btc_close", "eth_close", "dev"}
EMBARGO = pd.Timedelta("24h")


def feats_A(df):
    return [c for c in df.columns if c not in DROP_EXACT
            and not c.startswith(DROP_PREFIX) and not c.startswith("b_")]


def make_xgb(spw):
    return XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8,
                         colsample_bytree=0.8, scale_pos_weight=spw, eval_metric="logloss",
                         random_state=42, verbosity=0, n_jobs=-1)


def far(y, pred):
    tn = int(((pred == 0) & (y == 0)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    return fp / (fp + tn) if (fp + tn) > 0 else 0.0


def main():
    df = pd.read_csv(os.path.join(PROC, "df_usdc_hourly.csv"), parse_dates=["datetime"])
    df = df.replace([np.inf, -np.inf], np.nan).sort_values("datetime").reset_index(drop=True)
    feats = feats_A(df)
    print(f"A 피처 {len(feats)}개, target={TARGET}\n")

    s, e = pd.Timestamp("2023-02-15"), pd.Timestamp("2023-04-15")
    tr = df[(df["datetime"] < s - EMBARGO) | (df["datetime"] > e + EMBARGO)].dropna(subset=[TARGET])
    te = df[(df["datetime"] >= s) & (df["datetime"] <= e)].dropna(subset=[TARGET]).reset_index(drop=True)

    imp = SimpleImputer(strategy="median")
    Xtr = imp.fit_transform(tr[feats]); ytr = tr[TARGET].astype(int).values
    Xte = imp.transform(te[feats]); yte = te[TARGET].astype(int).values
    spw = (ytr == 0).sum() / max(ytr.sum(), 1)

    base = make_xgb(spw).fit(Xtr, ytr)
    p_raw_tr = base.predict_proba(Xtr)[:, 1]
    p_raw_te = base.predict_proba(Xte)[:, 1]

    # sigmoid(Platt) 캘리브레이션 — 단조라 랭킹 보존
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    try:
        cal = CalibratedClassifierCV(estimator=make_xgb(spw), method="sigmoid", cv=skf)
    except TypeError:
        cal = CalibratedClassifierCV(base_estimator=make_xgb(spw), method="sigmoid", cv=skf)
    cal.fit(Xtr, ytr)
    p_cal_tr = cal.predict_proba(Xtr)[:, 1]
    p_cal_te = cal.predict_proba(Xte)[:, 1]

    print("[캘리브레이션 품질] SVB test")
    print(f"  Brier  원={brier_score_loss(yte, p_raw_te):.5f}  sigmoid={brier_score_loss(yte, p_cal_te):.5f} (낮을수록↑)")
    print(f"  AUC-PRC 원={average_precision_score(yte, p_raw_te):.3f}  sigmoid={average_precision_score(yte, p_cal_te):.3f} (랭킹 보존 확인)\n")

    # 경보예산 기반 이중 임계값 (캘리브레이션 확률의 train 전체분포 분위수)
    t_caution = float(np.quantile(p_cal_tr, 1 - BUDGET_CAUTION))
    t_alert = float(np.quantile(p_cal_tr, 1 - BUDGET_ALERT))
    print(f"[운영점] 경보예산 기반 (sigmoid 확률, train 전체분포 분위수)")
    print(f"  주의(Caution): P >= {t_caution:.5f}  (예산 상위 {BUDGET_CAUTION:.0%})")
    print(f"  경보(Alert)  : P >= {t_alert:.5f}  (예산 상위 {BUDGET_ALERT:.0%})\n")

    caution = p_cal_te >= t_caution
    alert = p_cal_te >= t_alert
    print(f"[SVB out-of-sample 평가] (양성 {int(yte.sum())}시간 / 전체 {len(yte)})")
    for name, mask, thr in [("주의+", caution, t_caution), ("경보", alert, t_alert)]:
        print(f"  {name:5s}(P>={thr:.4f}): recall={recall_score(yte, mask.astype(int), zero_division=0):.3f}, "
              f"실제 오경보율={far(yte, mask.astype(int)):.3f}, 발령 {int(mask.sum())}시간")

    onset = te.index[(te["close"] - 1).abs() > 0.005]
    if len(onset) and caution.any():
        lead = (te.loc[onset[0], "datetime"] - te.loc[np.where(caution)[0][0], "datetime"]).total_seconds() / 3600
        print(f"  lead time(첫 주의→첫 이탈): {lead:+.0f}h (Binance 결측으로 SVB는 위기 도중 시작)")

    # 3단계 타임라인
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1.3]})
    t = te["datetime"]
    ax1.plot(t, te["close"], color="#2C3E50", lw=1.3, label="USDC/USDT close")
    ax1.axhline(0.995, color="#E67E22", ls="--", lw=0.8, label="peg ±0.5%")
    ax1.axhline(1.005, color="#E67E22", ls="--", lw=0.8)
    av = alert.values if hasattr(alert, "values") else alert
    cv = (caution & ~alert).values if hasattr(caution, "values") else (caution & ~alert)
    for i in np.where(av)[0]:
        ax1.axvspan(t.iloc[i], t.iloc[min(i+1, len(t)-1)], color="#E74C3C", alpha=0.18)
    for i in np.where(cv)[0]:
        ax1.axvspan(t.iloc[i], t.iloc[min(i+1, len(t)-1)], color="#F39C12", alpha=0.15)
    ax1.set_ylabel("USDC price")
    ax1.set_title("SVB out-of-sample — 3-level alert (sigmoid calibration + alert-budget thresholds) | A/h6",
                  fontsize=12, fontweight="bold")
    from matplotlib.patches import Patch
    handles = ax1.get_legend_handles_labels()[0] + [
        Patch(facecolor="#E74C3C", alpha=0.18, label="Alert"),
        Patch(facecolor="#F39C12", alpha=0.15, label="Caution")]
    ax1.legend(handles=handles, loc="lower right", fontsize=8)

    ax2.fill_between(t, p_cal_te, alpha=0.3, color="#3498DB")
    ax2.plot(t, p_cal_te, color="#2980B9", lw=1.0, label="calibrated P(depeg<=6h)")
    ax2.axhline(t_caution, color="#F39C12", ls="--", lw=1.1, label=f"Caution {t_caution:.5f}")
    ax2.axhline(t_alert, color="#E74C3C", ls="--", lw=1.1, label=f"Alert {t_alert:.5f}")
    ax2.set_ylabel("calibrated P")
    ax2.set_ylim(0, max(float(p_cal_te.max()) * 1.15, t_alert * 1.5))
    ax2.legend(loc="upper right", fontsize=8)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %Hh"))
    plt.xticks(rotation=45); plt.tight_layout()
    path = os.path.join(OUT, "hourly_calibrated_3level_SVB.png")
    plt.savefig(path, dpi=140, bbox_inches="tight"); plt.close()
    print(f"\n시각화: {path}")

    pd.DataFrame([{
        "config": "A/h6/XGB+sigmoid",
        "t_caution": round(t_caution, 4), "t_alert": round(t_alert, 4),
        "recall_caution": round(recall_score(yte, caution.astype(int), zero_division=0), 3),
        "recall_alert": round(recall_score(yte, alert.astype(int), zero_division=0), 3),
        "far_caution": round(far(yte, caution.astype(int)), 3),
        "far_alert": round(far(yte, alert.astype(int)), 3),
        "brier_raw": round(brier_score_loss(yte, p_raw_te), 5),
        "brier_cal": round(brier_score_loss(yte, p_cal_te), 5),
        "aucpr": round(average_precision_score(yte, p_cal_te), 3),
    }]).to_csv(os.path.join(ML_DIR, "hourly_calibrated_thresholds.csv"), index=False)
    print("저장: data/ml/hourly_calibrated_thresholds.csv")


if __name__ == "__main__":
    main()
