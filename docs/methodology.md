# Methodology

최종 분석 파이프라인은 7단계로 정리됩니다.

## 1. Target Definition

Typical Price를 `(High + Low + Close) / 3`으로 계산합니다. `|TP - 1| > 0.01`이면 디페깅 후보로 보고, 3일 중 2일 이상 조건을 만족하는 2-of-3 filter를 적용해 최종 디페깅 라벨을 정의합니다. 모델 타깃은 다음 날 디페깅 여부입니다.

## 2. Feature Engineering

시장 가격, 변동성, 거래량, 공급량, 시가총액, 거시경제, 온체인, DeFi, 검색량 변수를 구성했습니다. 후보 변수는 218개였고, 최종 모델에는 60개 변수를 사용했습니다.

## 3. Text Feature Experiment

뉴스 데이터에서 키워드 사전 기반 변수, LDA 토픽 변수, FinBERT 임베딩 기반 변수를 생성했습니다. 텍스트 변수를 포함한 모델과 제외한 모델의 성능을 비교했고, 최종 F2-score 개선이 없어 최종 모델에서는 제외했습니다.

## 4. Imbalanced Data Handling

디페깅 이벤트는 드문 이벤트이므로 F2-score와 Recall을 중심으로 평가했습니다. 여러 oversampling 및 weighting 전략을 비교한 결과, 최종 발표 기준으로 USDC는 WeightOnly, DAI는 ADASYN을 사용했습니다.

## 5. Model Training and Comparison

Random Forest, XGBoost, LightGBM, SVM 등 후보 모델을 비교했습니다. 최종 모델은 두 코인 모두 XGBoost입니다. 성능 평가는 F2-score와 Recall을 중심으로 해석했습니다.

## 6. SHAP Interpretation

SHAP을 사용해 최종 모델의 주요 변수와 개별 예측 근거를 해석했습니다. 전역 중요도는 모델이 반복적으로 참조한 위험 요인을 보여주고, 개별 관측치 해석은 특정 날짜의 경보 발생 원인을 설명하는 데 사용했습니다.

## 7. Three-Stage Early Warning System

모델의 `P(depeg)`를 Normal, Caution, Alert 세 단계로 변환했습니다. Alert는 실제 디페깅을 놓치지 않는 방향으로 Recall과 F2-score를 중시해 설정했습니다.

