# 시간단위(hourly) 디벨롭 — USDC 시계열 검증 (2026-07-08)

## 목적
v2(일별)의 최대 약점 = **시계열 검증 불가**(USDC 양성 33건 중 29건이 2020년 집중 → TimeSeriesSplit·Purged K-Fold 전부 F2≈0). 이를 hourly 재구축으로 해소하고, "과거 위기로 학습 → 새 위기로 검증"이라는 문헌 표준(Curve/BOCD)을 실제로 구현.

## 데이터
- **소스**: Binance klines 1h (USDCUSDT, BTCUSDT, ETHUSDT), 2020-01-01~2026-07, 무료·무인증
- **USDT-준달러 기준**: USDCUSDT는 USDT-quote. SVB 기간 Coinbase USDT-USD가 0.999~1.015로 안정 확인 → USDCUSDT 하락 = USDC 하락으로 해석 정당
- ⚠️ **한계 1 — BUSD 자동전환 결측**: 2022-09-26~2023-03-11(166일) USDCUSDT 거래중단. 공교롭게 SVB 시작 시점에 재개 → **SVB pre-onset lead time 측정 불가**(데이터가 위기 도중 시작). 결측은 평온기라 모델엔 무해
- ⚠️ **한계 2 — 나쁜 wick**: 시간봉 저가에 가짜 프린트(2021-12-04 저가 0.20, 2024-01-03 저가 0.76, 종가는 정상) → 타겟을 **종가 기준**으로 정의해 강건화

## 타겟 (`h1_build_hourly.py`)
- `dev = |close − 1|`, 디페깅 = dev > τ + 2-of-3 연속시간 persistence
- **다중 시계 전방 라벨**: `y^H = 1` iff (t, t+H] 구간 디페깅 발생, H ∈ {1,6,24}h. 연속 보조: (t,t+H] max(dev)
- **τ=±0.5%(tau05) 메인**: 양성이 **2020(94)/2022(35)/2023(89)** 3개년 분산 → walk-forward 성립. τ=±1%는 2020이 0이라 robustness용
- 세그먼트 분할로 166일 결측을 rolling이 넘나들지 않게 처리

### 드러난 디페깅 에피소드 (종가 ±0.5%)
| 시기 | 사건 | 시간 | 최저 종가 |
|------|------|------|----------|
| 2020-03-13~19 | COVID Black Thursday | 21h | 0.9890 |
| 2022-05-12 | UST/Terra 붕괴 전이 | 12h | 1.0050(프리미엄) |
| 2023-03-11~15 | SVB | 58h | 0.9128 |
> hourly 재구축으로 일별 ±1%에선 흐릿했던 **2022 UST 전이 에피소드**가 드러남

## 검증 (`h2_validation.py`)
- **피처**: A(코어 hourly 네이티브 24개: 수익률·변동성·RSI·MA이탈·shadow·거래량·BTC/ETH) / B(일별 매크로·심리 ffill, 애블레이션). 타겟기반(dev/close) 컬럼은 누수 방지로 제외
- **Walk-forward**: 연도별 확장윈도우, embargo=24h. **Cross-event**: 위기구간 홀드아웃 → AUC-PRC·recall·FAR·lead time
- **운영점**: 임계값은 절대확률이 작아(base rate 0.2%) 자의적 → **AUC-PRC(임계값 무관)를 주지표**로. recall/FAR는 1% 경보예산 기준 참고치

### 핵심 결과 (SVB cross-event, 2023 학습 제외)
| 피처셋 | 시계 | AUC-PRC | recall | FAR |
|--------|------|---------|--------|-----|
| **A** | **6h** | **0.880** | 0.855 | 0.049 |
| A | 1h | 0.775 | 0.707 | 0.029 |
| A | 24h | 0.661 | 0.697 | 0.048 |
| A+B | 6h | 0.829 | 0.867 | 0.088 |
| A+B | 24h | 0.428 | 0.270 | 0.014 |

Walk-forward 2023 fold: A/h6 **AUC-PRC 0.881, recall 0.771, FAR 0.14%**.

## 결론
1. **시계열 검증이 작동한다** — 2023을 학습에서 빼고도 SVB를 out-of-sample AUC-PRC 0.88로 포착. v2의 leakage 약점을 구조적으로 제거(핵심 성과)
2. **Lean(A) > 매크로증강(A+B)** — h6·h24에서 A 우세. 일별 매크로는 hourly에서 성능 저하(레짐-암기/노이즈). v2 텍스트-피처 "검증 후 제거" 서사와 동일 구조로 재확인
3. **6시간 시계가 스위트스팟** — h1은 쉽지만 덜 조기, h24는 어려움. h6이 AUC-PRC·조기성 균형 최적
4. **정직한 한계**: SVB pre-onset lead time은 Binance 결측으로 0h(측정불가). 2022 UST는 0.5% 프리미엄 blip이라 walk-forward fold에서 약함(AUC-PRC 0.08~0.25)

## LSTM 비교 (`h3_lstm.py`)
직전 24시간 피처 시퀀스 LSTM(PyTorch) vs XGBoost, 동일 split.

| Split | LSTM | XGB |
|-------|------|-----|
| WF_2023 (순수 walk-forward) | 0.891 | 0.881 |
| SVB cross-event (위기 홀드아웃) | **0.454** | **0.880** |

- 순수 시간분할에선 LSTM ≈ XGB(대등, 문헌의 RNN>트리 주장 약하게 성립)
- **위기 통째 홀드아웃에선 LSTM 붕괴(0.454), XGB 견고(0.880)** — 총 양성 ~90건뿐인 희소레짐에서 LSTM이 과적합해 out-of-distribution 일반화 실패
- **결론: XGBoost 채택.** "딥러닝이 무조건 낫다"를 반증 — v2 정직검증 서사와 일치. (지원서 LSTM 언급은 "시도했으나 데이터 규모상 트리가 견고"로 정직하게)

## 경보 임계값 보정 (`h5_threshold_calibration.py`)
문제: XGB 원확률이 희소사건이라 작아 고정 임계값이 자의적 (arXiv:2512.00916 — 저확률서 F1/AUPRC 최적임계값이 0/1로 퇴화하는 알려진 현상). 문헌 표준(CALIBURN류 운영 보정)으로 해결:
- **sigmoid(Platt) 캘리브레이션**: 단조변환이라 랭킹(AUC-PRC 0.88→0.86) 보존하며 확률값 해석가능화. (isotonic은 양성 소수라 계단함수로 랭킹 붕괴 0.88→0.48 → 제외)
- **경보예산(alert budget) 이중 임계값**: 음성 분위수(0 퇴화) 대신 **train 전체분포 상위 %**로 역산 → 주의=상위 5%, 경보=상위 1%. 임계값을 "절대확률"이 아니라 **"위험 상위 %"**로 표현하는 게 직관적(희귀사건이라 절대확률 0.0002 수준으로 촘촘).

### SVB out-of-sample 3단계 성능
| 단계 | recall | 실제 오경보율 | 성격 |
|------|--------|--------------|------|
| **주의(Caution, 상위 5%)** | 0.976 | 0.115 | 거의 다 포착, 조기·민감 |
| **경보(Alert, 상위 1%)** | 0.723 | 0.018 | 고신뢰·저오경보 |

→ 자의적 임계값 문제 해소. **작동하는 3단계 조기경보 완성** (`outputs/ml/hourly_calibrated_3level_SVB.png`).

### 주의 예산 5% 유지의 실증 근거 (`h15_caution_budget.py`, 2026-07-09 확정)
"주의를 조이면 오탐↓ 미탐↑?" 트레이드오프를 walk-forward OOS로 실측:

| 예산 | 2022(UST) recall | UST 사전주의 | 평온 3년 점등 |
|------|------|------|------|
| **5% (채택)** | **0.333** | **4h 전** | 133h (0.6%) |
| 3% | 0 (전멸) | 없음 | 10h |
| 2% | 0 | 없음 | 4h |

5%→3%만 조여도 UST 전이 탐지가 통째로 소실(0.5% 프리미엄 수준의 미묘한 신호는 상위 5%엔 들지만 3%엔 못 듦). **주의 점등 = 민감층의 보험료**이며, 평온 3년 133h의 대가로 UST급 미묘 위기 탐지 + 4h 사전 경보를 확보 → recall 중시(F2) 설계 철학과 일치, 5%/1% 유지 확정.

## Robustness: τ=±1% 재검증 (`h2_validation.py tau10`, 2026-07-09)
"±0.5% 선택이 결과를 만든 것 아니냐"는 반론 검증. 데이터는 2026-07-09 기준 재수집(53,020행, 타겟분포 동일).
- **신호는 ±1%에서도 존재**: SVB h6 AUC-PRC 0.502(구간 base rate ~5% 대비 **10배 리프트**), UST 0.54~0.56 → 결과가 임계값 선택의 인공물이 아님
- 단 ±1%에선 **2020 에피소드가 소실**(hourly 종가가 1% 미달) → 학습가능 위기 감소로 절대성능 하락. **±0.5% = "신호 존재 + 학습·검증 가능성"의 균형점**이라는 채택 논리 강화
- 참고: 재수집 데이터로 tau05 수치 미세변동(SVB h6 0.880→0.867), 방향 동일. 결과: `hourly_crossevent_tau10.csv` 등

## SHAP 해석 (`h6_shap_hourly.py`)
무엇이 경보를 울리는가 (XGBoost 내장 TreeSHAP):
- **전체 기간 Top**: btc_vol(BTC 거래량), vol_6h(USDC 단기변동성), hl_spread(일중 스프레드), lower_shadow, vol_24h
- **SVB 구간(홀드아웃 모델, 누수 없는 뷰) Top**: **vol_24h(USDC 24h 변동성) 1위**, btc/eth_vol_24h(암호시장 전반 변동성) 2·3위, hl_spread, vol_72h — beeswarm에서 vol_24h 고값이 예측을 강하게 상승
- **v2(일별)와 일관**: v2 SHAP 1위도 vol_7d(변동성) — **해상도 불문 "변동성·유동성이 디페깅 최강 선행신호"**. 위기 시엔 BTC/ETH 변동성(시장 전반 스트레스)이 함께 점화 → 매크로 피처 없이도 전이 경로가 미시신호로 포착됨
- ⚠️ 정직한 주의: 전체기간 1위 btc_vol은 수준(level) 변수라 레짐 프록시 성격 가능. 누수 없는 SVB 홀드아웃 뷰에서는 USDC 자체 변동성이 1위 — 이쪽이 신뢰할 해석
- 산출: `hourly_shap_importance.csv`, `outputs/ml/hourly_shap_{full,svb}.png`

## seed 앙상블 + 기하평균 (`h13_seed_ensemble.py`, 2026-07-09)
수상작 리버스엔지니어링 #2(2020 미래에셋 보험금 청구분류 1위: ExtraTrees×3 seed 기하평균) 기법을 XGB×5 seed로 이식.
동일 프로토콜(h2)에서 결합 방식만 애블레이션: single(seed42) vs 산술평균 vs 기하평균.

| 지표 | single | 산술평균 | 기하평균 |
|------|--------|---------|---------|
| WF pooled AUC-PRC | 0.626 | **0.657** | 0.655 |
| WF 2022 fold | 0.080 | **0.131** | 0.128 |
| SVB AUC-PRC / recall | 0.867 / 0.867 | **0.882 / 0.880** | 0.879 / 0.880 |
| UST AUC-PRC | 0.480 | 0.433 | 0.431 |

핵심 발견 — **"seed 운" 효과가 실재하고, 앙상블이 이를 제거**:
- 2022 fold에서 seed42는 불운(0.080 vs seed 평균 0.125±0.034), UST에서는 행운(0.480 vs 평균 0.434±0.039)
  → 단일 seed 보고 성능에는 ±0.04 수준의 운이 섞임. UST의 single 우위는 개선이 아니라 행운의 흔적
- 앙상블은 기대성능에 수렴 → **보고 수치의 신뢰성·재현성 확보** (pooled +0.031, SVB +0.015)
- 산술 vs 기하: 사실상 동률(차이 ≤0.003). 기하평균의 보수성은 COVID FAR에서만 미세하게 드러남(0.594 vs 0.605)
  → **채택: 5-seed 산술평균** (단순함 우선). 기하평균은 극단확률 보수화가 필요한 상황용 옵션으로 기록
- 산출: `data/ml/hourly_seed_ensemble.csv`

## 산출물
- 데이터: `data/processed/df_usdc_hourly.csv` (53,020행 × 61열, 2026-07-09 재수집)
- 코드: `data/collect/collect_binance_hourly.py`, `notebooks/src/{h1_build_hourly,h2_validation,h3_lstm,h4_early_warning_hourly,h5_threshold_calibration,h6_shap_hourly,h13_seed_ensemble}.py`
- 결과: `data/ml/{hourly_walkforward,hourly_crossevent,hourly_lstm_vs_xgb,hourly_calibrated_thresholds,hourly_shap_importance}.csv` (+`*_tau10.csv`)
- 시각화: `outputs/ml/hourly_early_warning_{SVB,COVID,UST}_*.png`, `hourly_calibrated_3level_SVB.png`

## 다음 (미완)
- 코인 확장(DAI·UST·USDe 등) — 별도 hourly 소스 확보 후. 멀티코인이면 이벤트가 시간축에 더 분산돼 walk-forward 강건성↑, cross-event도 다수 위기로 검증 가능
- (심화 옵션) EVT/POT 꼬리 모델링 또는 BOCD 변화점 탐지로 도메인 표준 정합 강화
