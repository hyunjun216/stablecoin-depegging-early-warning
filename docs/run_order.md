# Run Order

이 문서는 새 포트폴리오 레포 기준 실행 순서입니다. 원자료 전체가 포함되어 있지 않으므로 전체 재실행은 not fully re-executed 상태입니다.

## 0. Environment

```bash
pip install -r requirements.txt
```

## 1. Prepare Data

```bash
python src/01_preprocess.py
```

Reproduction note: `data/raw/` 원자료가 필요합니다. API key가 필요한 원본 수집 스크립트는 이 레포에서 제외했습니다.

## 2. Select Features

```bash
python src/02_prepare_features.py
```

Reproduction note: 최종 발표 기준은 후보 변수 218개, 최종 변수 60개입니다.

## 3. Generate Text Features

```bash
python src/03_text_features_experiment.py
```

Reproduction note: 뉴스 원문과 FinBERT 파생변수 생성 환경이 필요합니다. 텍스트 변수는 최종 모델에서는 제외되었습니다.

## 4. Compare Imbalance Methods

```bash
python src/04_imbalance_experiment.py
```

Final setting: USDC WeightOnly, DAI ADASYN.

## 5. Compare Models

```bash
python src/05_model_comparison.py
```

Final setting: XGBoost for USDC and DAI.

## 6. SHAP Analysis

```bash
python src/06_shap_analysis.py
```

Reproduction note: SHAP 분석은 해석 목적입니다. full-data fit 결과를 일반화 성능으로 해석하지 않습니다.

## 7. Early Warning System

```bash
python src/07_early_warning.py
```

Output concept: Normal, Caution, Alert.

## API Key Note

원본 프로젝트에서 CoinGecko API key가 하드코딩된 수집 스크립트가 발견되어 이 레포에는 포함하지 않았습니다. 필요한 경우 다음처럼 환경변수를 사용합니다.

```bash
set COINGECKO_API_KEY=your_key_here
```

```python
import os

api_key = os.environ.get("COINGECKO_API_KEY")
```
