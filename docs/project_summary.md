# Project Summary

## Summary

| 항목 | 내용 |
|---|---|
| 주제 | USDC·DAI 스테이블코인 디페깅 조기경보 모델 |
| 분석 기간 | 2020-01-01 to 2026-03-25 |
| 예측 목표 | 다음 날 디페깅 여부 |
| 타깃 정의 | Typical Price 기준 1% 이상 이탈 + 2-of-3 filter |
| 후보 변수 | 218개 |
| 최종 변수 | 60개 |
| 최종 모델 | XGBoost |
| 불균형 처리 | USDC WeightOnly, DAI ADASYN |
| 주요 지표 | F2-score, Recall |

