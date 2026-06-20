# Project Summary

이 저장소는 BAF 26-1 팀 프로젝트인 USDC·DAI 스테이블코인 디페깅 조기경보 모델을 개인 포트폴리오 제출용으로 정리한 버전입니다.

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

## Portfolio Scope

원본 팀 프로젝트 전체가 아니라 최종 발표 기준과 직접 연결되는 코드, 문서, figure만 포함했습니다. API key가 포함된 수집 코드, 백업 파일, 중간 산출물, 원자료 전체, pkl 모델 파일은 제외했습니다.

