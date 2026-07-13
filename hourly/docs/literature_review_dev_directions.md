# 디벨롭 방향 문헌 조사 — 스테이블코인 디페깅 분석은 어떻게 이어지는가

> 작성: 2026-07-08 (arXiv 탐색 기반)
> 목적: v2 완료 이후 디벨롭 방향 결정. 특히 "양성 샘플의 시간 편중 때문에 시계열 검증 불가" 문제를 다른 연구들이 어떻게 우회하는지 확인

## 핵심 논문 3편

### 1. Cintra & Holloway (2023), "Detecting Depegs: Towards Safer Passive Liquidity Provision on Curve Finance" (arXiv:2306.10612)
- **방법**: 가격·거래 데이터 기반 지표 + Bayesian Online Changepoint Detection (BOCD). 비지도·온라인 방식이라 양성 라벨 수에 의존하지 않음
- **검증 방식이 핵심**: **2022년 UST 붕괴 데이터로 학습 → 2023년 3월 USDC(SVB) 디페깅으로 out-of-sample 테스트**
- **평가 지표**: fold 기반 F-score가 아니라 (a) **lead time** — USDC가 $0.99 아래로 떨어지기 약 5시간 전(3/10 21:00 UTC) 경보, (b) **17개월 테스트 구간의 오경보 건수**
- **데이터**: 시간 단위(intraday), Curve 13개 StableSwap 풀, 2022~2023
- 시사점: 우리가 고민한 "이벤트 스터디 검증"이 실제로 이 도메인의 표준 관행. cross-event(다른 위기로 학습→새 위기로 테스트) + lead time + 오경보율 조합

### 2. "Stability Anchors and Risk Amplifiers: Tail Spillovers Across Stablecoin Designs" (arXiv:2602.18820)
- **데이터 설계**: 11개 스테이블코인(fiat/crypto-collateralized/algorithmic), 일별 2020-12~2025-11 패널 + **분·시간 단위 이벤트 스터디 4건** 병행
- **타겟**: 이진 디페깅이 아니라 **연속 페그 이탈(bp 단위 deviation)** — QVAR로 분위(5/50/95%)별 스필오버 추정
- **이벤트 목록** (멀티코인 확장 시 그대로 활용 가능):
  | 코인 | 시점 | 최대 이탈 | 유형 |
  |------|------|----------|------|
  | LUSD | 2022-03 | −23.3% | crypto-collat. |
  | UST | 2022-05 | −37.7% | algorithmic |
  | FRAX | 2022-06 | +15.2% | algorithmic |
  | USDC | 2023-03 | −5.9% | fiat |
  | DAI | 2023-03 | −4.8% | fiat(전이) |
  | sUSD | 2025-04 | −22.6% | algorithmic |
  | USDe | 2025-10 | −32.0% | crypto-collat. |
- 시사점: 코인을 넓히면 이벤트가 2022~2025에 자연 분산 → 시간 분할 검증 성립. "일별 패널(구조 분석) + 고빈도(이벤트 검증)" 이중 설계가 최신 관행

### 3. IPB University (2024), "Classification Modeling with RNN-Based, RF, XGBoost for Imbalanced Data: Early Crash Detection in ASEAN-5" (arXiv:2406.07888)
- 희귀 사건(시장 크래시) 이진 분류 — 우리 프로젝트와 구조 동일
- **타겟**: 고정 임계값이 아니라 **VaR 분위(5%/2.5%/1%) 기반** → 양성 비율을 시나리오로 조절, 양성이 특정 시기에 몰리지 않음
- **검증**: **expanding-window 시계열 CV** (2010~2011 학습→2012~2013 검증 … 최종 test 2020~2023)
- **지표**: hit rate, false alarm rate, balanced accuracy, AUC-PRC
- **결과**: Simple RNN > LSTM/GRU > RF/XGBoost (불균형 처리: SMOTE-ENN)
- 시사점: 연속/분위 기반 타겟으로 바꾸면 시계열 CV가 성립. LSTM 계열 확장(비타민 지원서 방향)의 직접 근거

## 보조 논문

- **arXiv:2205.06338** — Multivariate Hawkes process로 디페깅 이벤트 동학 모델링. 이진 분류 대신 point process(이벤트 강도 λ) 관점. 자기·상호 여기(mutually-exciting) 구조로 "디페깅이 디페깅을 부르는" 군집성 포착
- **arXiv:2301.00509** — time-varying DAR 모델로 Tether 페그 안정성 측정. 연속 타겟 + 국소 정상성 접근
- **arXiv:2606.07442** — SVB/USDC 디페깅을 온체인 트랜잭션 데이터로 이벤트 스터디 (고빈도·행동 반응 분석)
- **arXiv:2506.17622** — SoK: 95개 스테이블코인, 44개 보안 사건 정리 (멀티코인 확장 시 코인 선정·사건 라벨링 참고)

## 결론 — 문헌이 가리키는 디벨롭 패턴

1. **검증은 fold 지표가 아니라 이벤트 기반**: "과거 위기로 학습 → 새 위기로 테스트 + lead time + 오경보율" (Curve 논문이 정확히 우리 프로젝트의 다음 버전 모습)
2. **타겟을 연속화/분위화**: bp deviation, VaR 분위 임계값 → 양성 희소성 문제 완화 + 시계열 CV 성립
3. **멀티코인 패널 + 고빈도 이벤트 스터디 이중 설계**: 이벤트가 시간축에 분산되어 temporal split 가능, 메커니즘 유형(fiat/crypto/algo)별 비교라는 새 연구 질문 확보
4. 모델 확장 근거: RNN/LSTM(불균형 크래시 탐지에서 트리 모델 능가 보고), BOCD(비지도 온라인), Hawkes(이벤트 동학)
