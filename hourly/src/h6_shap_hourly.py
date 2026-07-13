"""
h6_shap_hourly.py — hourly 최적모델(A/h6/XGB)의 SHAP 해석

무엇이 경보를 울리는가:
  1) 전체 데이터 학습 모델의 SHAP 중요도 (v2 s4와 동일 방식)
  2) SVB 홀드아웃 모델로 SVB 구간 SHAP → "위기 때 어떤 신호가 점화됐나"
v2(일별) SHAP와 비교: v2 1위=vol_7d(변동성), 거시(us_10y, dxy)가 3~10위.
hourly는 A(가격·거래량·유동성)만이므로 미시신호 간 순위가 관심사.
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import shap
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier, DMatrix


def shap_values_xgb(model, X):
    """XGBoost 내장 TreeSHAP (shap.TreeExplainer의 신버전 호환문제 우회, 동일 알고리즘)"""
    contribs = model.get_booster().predict(DMatrix(X), pred_contribs=True)
    return contribs[:, :-1]  # 마지막 열 = bias 제외

PROJECT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
PROC = os.path.join(PROJECT_DIR, "data", "processed")
ML_DIR = os.path.join(PROJECT_DIR, "data", "ml")
OUT = os.path.join(PROJECT_DIR, "outputs", "ml")

TARGET = "y_tau05_h6"
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


def main():
    df = pd.read_csv(os.path.join(PROC, "df_usdc_hourly.csv"), parse_dates=["datetime"])
    df = df.replace([np.inf, -np.inf], np.nan).sort_values("datetime").reset_index(drop=True)
    feats = feats_A(df)
    d = df.dropna(subset=[TARGET])
    print(f"A 피처 {len(feats)}개, target={TARGET}, n={len(d)} (양성 {int(d[TARGET].sum())})\n")

    # ── 1) 전체 데이터 모델 SHAP ──
    imp = SimpleImputer(strategy="median")
    X = imp.fit_transform(d[feats]); y = d[TARGET].astype(int).values
    spw = (y == 0).sum() / max(y.sum(), 1)
    model = make_xgb(spw).fit(X, y)

    sv = shap_values_xgb(model, X)
    imp_full = pd.Series(np.abs(sv).mean(axis=0), index=feats).sort_values(ascending=False)
    print("[전체 기간] SHAP 중요도 Top 15:")
    print(imp_full.head(15).round(4).to_string())

    # beeswarm (top 15)
    plt.figure()
    shap.summary_plot(sv, pd.DataFrame(X, columns=feats), max_display=15, show=False)
    plt.title("SHAP — hourly A/h6 XGB (full period)", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "hourly_shap_full.png"), dpi=140, bbox_inches="tight")
    plt.close()

    # ── 2) SVB 홀드아웃 모델로 SVB 구간 SHAP ──
    s, e = pd.Timestamp("2023-02-15"), pd.Timestamp("2023-04-15")
    tr = d[(d["datetime"] < s - EMBARGO) | (d["datetime"] > e + EMBARGO)]
    te = d[(d["datetime"] >= s) & (d["datetime"] <= e)]
    imp2 = SimpleImputer(strategy="median")
    Xtr = imp2.fit_transform(tr[feats]); ytr = tr[TARGET].astype(int).values
    Xte = imp2.transform(te[feats])
    m2 = make_xgb((ytr == 0).sum() / max(ytr.sum(), 1)).fit(Xtr, ytr)
    sv_te = shap_values_xgb(m2, Xte)
    imp_svb = pd.Series(np.abs(sv_te).mean(axis=0), index=feats).sort_values(ascending=False)
    print("\n[SVB 구간(홀드아웃 모델)] SHAP 중요도 Top 10:")
    print(imp_svb.head(10).round(4).to_string())

    plt.figure()
    shap.summary_plot(sv_te, pd.DataFrame(Xte, columns=feats), max_display=12, show=False)
    plt.title("SHAP — SVB window (model trained excluding SVB)", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "hourly_shap_svb.png"), dpi=140, bbox_inches="tight")
    plt.close()

    # 저장
    out = pd.DataFrame({"feature": imp_full.index,
                        "shap_full": imp_full.values,
                        "shap_svb": imp_svb.reindex(imp_full.index).values})
    out.to_csv(os.path.join(ML_DIR, "hourly_shap_importance.csv"), index=False)
    print(f"\n저장: data/ml/hourly_shap_importance.csv")
    print(f"시각화: outputs/ml/hourly_shap_full.png, hourly_shap_svb.png")


if __name__ == "__main__":
    main()
