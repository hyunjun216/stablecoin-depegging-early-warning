"""
h3_lstm.py — LSTM 시퀀스 모델 vs XGBoost (A/h6)

직전 k=24시간 피처 시퀀스로 y_tau05_h6 예측.
동일 split(SVB cross-event, 2023 walk-forward)에서 XGB와 AUC-PRC 비교.
문헌: 불균형 크래시 탐지에서 RNN이 트리 능가 보고(arXiv:2406.07888) → 검증.
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier

torch.manual_seed(42)
np.random.seed(42)

PROJECT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
PROC = os.path.join(PROJECT_DIR, "data", "processed")
ML_DIR = os.path.join(PROJECT_DIR, "data", "ml")

TARGET = "y_tau05_h6"
LOOKBACK = 24
DROP_PREFIX = ("depeg_", "y_", "ymax_")
DROP_EXACT = {"datetime", "seg", "open", "high", "low", "close", "volume",
              "quote_vol", "n_trades", "btc_close", "eth_close", "dev"}
EMBARGO = pd.Timedelta("24h")


def feats_A(df):
    return [c for c in df.columns if c not in DROP_EXACT
            and not c.startswith(DROP_PREFIX) and not c.startswith("b_")]


def build_sequences(df, feats, mask):
    """mask=True 행을 타겟으로, 직전 LOOKBACK 시간을 시퀀스로. 세그먼트 내부만."""
    X, y, idx = [], [], []
    arr = df[feats].values
    segs = df["seg"].values
    yv = df[TARGET].values
    pos = df.groupby("seg").cumcount().values
    m = mask.values
    for i in np.where(m)[0]:
        if pos[i] < LOOKBACK or np.isnan(yv[i]):
            continue
        seq = arr[i - LOOKBACK + 1: i + 1]
        if segs[i - LOOKBACK + 1] != segs[i]:      # 세그먼트 경계 넘으면 제외
            continue
        X.append(seq); y.append(yv[i]); idx.append(i)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), np.array(idx)


class LSTMClf(nn.Module):
    def __init__(self, n_feat, hidden=32):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, hidden, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(hidden, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


def train_lstm(Xtr, ytr, n_feat, epochs=15):
    dev = "cpu"
    model = LSTMClf(n_feat).to(dev)
    n_pos = max(ytr.sum(), 1); n_neg = (ytr == 0).sum()
    pos_w = torch.tensor([n_neg / n_pos], dtype=torch.float32)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    Xt = torch.tensor(Xtr); yt = torch.tensor(ytr)
    bs = 256
    for ep in range(epochs):
        perm = torch.randperm(len(Xt))
        model.train()
        for b in range(0, len(Xt), bs):
            bi = perm[b:b + bs]
            opt.zero_grad()
            out = model(Xt[bi])
            loss = lossf(out, yt[bi])
            loss.backward(); opt.step()
    return model


def predict_lstm(model, X):
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.tensor(X))).numpy()


def prep(df, feats, tr_mask, te_mask):
    """train으로 impute+scale 학습 후 시퀀스 생성"""
    imp = SimpleImputer(strategy="median").fit(df.loc[tr_mask, feats])
    sc = StandardScaler().fit(imp.transform(df.loc[tr_mask, feats]))
    dsc = df.copy()
    dsc[feats] = sc.transform(imp.transform(df[feats].values))
    Xtr, ytr, _ = build_sequences(dsc, feats, tr_mask)
    Xte, yte, _ = build_sequences(dsc, feats, te_mask)
    return Xtr, ytr, Xte, yte


def xgb_ap(df, feats, tr_mask, te_mask):
    """동일 split에서 XGB(단일행) AUC-PRC"""
    tr = df[tr_mask].dropna(subset=[TARGET]); te = df[te_mask].dropna(subset=[TARGET])
    imp = SimpleImputer(strategy="median")
    Xtr = imp.fit_transform(tr[feats]); Xte = imp.transform(te[feats])
    ytr = tr[TARGET].astype(int)
    spw = (ytr == 0).sum() / max(int(ytr.sum()), 1)
    m = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8,
                      colsample_bytree=0.8, scale_pos_weight=spw, eval_metric="logloss",
                      random_state=42, verbosity=0, n_jobs=-1).fit(Xtr, ytr)
    p = m.predict_proba(Xte)[:, 1]
    return average_precision_score(te[TARGET].astype(int), p)


def run_split(df, feats, tr_mask, te_mask, name):
    Xtr, ytr, Xte, yte = prep(df, feats, tr_mask, te_mask)
    if len(Xte) == 0 or yte.sum() == 0 or ytr.sum() < 5:
        print(f"  {name}: 스킵(양성부족)"); return None
    model = train_lstm(Xtr, ytr, len(feats))
    p = predict_lstm(model, Xte)
    ap_lstm = average_precision_score(yte, p)
    ap_xgb = xgb_ap(df, feats, tr_mask, te_mask)
    print(f"  {name}: LSTM AUC-PRC={ap_lstm:.3f} | XGB={ap_xgb:.3f} "
          f"(train seq={len(Xtr)}, test seq={len(Xte)}, test 양성={int(yte.sum())})")
    return {"split": name, "AUC_PRC_LSTM": round(ap_lstm, 3), "AUC_PRC_XGB": round(ap_xgb, 3),
            "n_test": len(Xte), "n_pos": int(yte.sum())}


def main():
    df = pd.read_csv(os.path.join(PROC, "df_usdc_hourly.csv"), parse_dates=["datetime"])
    df = df.replace([np.inf, -np.inf], np.nan).sort_values("datetime").reset_index(drop=True)
    feats = feats_A(df)
    print(f"A 피처 {len(feats)}개, lookback={LOOKBACK}h, target={TARGET}\n")

    results = []
    # 1) SVB cross-event
    s, e = pd.Timestamp("2023-02-15"), pd.Timestamp("2023-04-15")
    te_mask = (df["datetime"] >= s) & (df["datetime"] <= e)
    tr_mask = (df["datetime"] < s - EMBARGO) | (df["datetime"] > e + EMBARGO)
    r = run_split(df, feats, tr_mask, te_mask, "SVB_cross_event")
    if r: results.append(r)

    # 2) 2023 walk-forward fold (2023 이전으로 학습)
    t0 = pd.Timestamp("2023-01-01")
    tr_mask = df["datetime"] < (t0 - EMBARGO)
    te_mask = (df["datetime"] >= t0) & (df["datetime"] < pd.Timestamp("2024-01-01"))
    r = run_split(df, feats, tr_mask, te_mask, "WF_2023")
    if r: results.append(r)

    if results:
        pd.DataFrame(results).to_csv(os.path.join(ML_DIR, "hourly_lstm_vs_xgb.csv"), index=False)
        print(f"\n저장: data/ml/hourly_lstm_vs_xgb.csv")


if __name__ == "__main__":
    main()
