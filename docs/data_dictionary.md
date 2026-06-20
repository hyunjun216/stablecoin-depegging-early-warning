# Data Dictionary

원자료 전체는 이 저장소에 포함하지 않습니다. 아래는 재현 또는 면접 설명에 필요한 데이터 유형과 대표 변수입니다.

## Dataset Layers

| Layer | 설명 |
|---|---|
| Raw market data | USDC, DAI, USDT, BTC, ETH의 OHLCV, volume, market cap, supply |
| Macro data | DXY, VIX, rates, credit spread, liquidity indicators |
| On-chain/DeFi data | stablecoin supply, MakerDAO collateral, Curve imbalance, Aave/DeFi TVL |
| Attention data | Google Trends search intensity |
| Text data | news title/text, keyword counts, LDA topics, FinBERT-derived features |
| Modeling data | 218 candidate variables, 60 selected variables, next-day depeg target |

## Key Variables

| 그룹 | 대표 변수 | 의미 |
|---|---|---|
| Price | `high_USDC`, `low_DAI`, `close_DAI` | 디페깅 및 변동성 계산의 기초 가격 |
| Target | `typical_price`, `depeg`, `Y` | Typical Price 기반 디페깅 및 다음 날 타깃 |
| Volatility | `return_1d`, `vol_7d`, `vol_30d` | 단기/중기 가격 변동성 |
| Liquidity | `volume_ratio`, `turnover_rate` | 거래량 및 유동성 변화 |
| Supply | `supply_change`, `mint_intensity` | 발행/소각 및 공급량 변화 |
| Macro | `dxy`, `vix`, `federal_funds_rate` | 거시 위험 환경 |
| DeFi | `curve_imbalance`, `maker_collateral`, `lending_tvl` | 온체인 및 DeFi 구조 변수 |
| Text | `risk_keyword_total`, `topic_*`, `finbert_*` | 텍스트 기반 위험 신호 후보 |

## Data Availability Note

GitHub 저장소에는 원자료 전체를 올리지 않습니다. 데이터 크기, 외부 API 약관, 재배포 가능성, 개인정보 또는 저작권 이슈가 있을 수 있기 때문입니다. 대신 데이터 출처, 변수 유형, 실행 순서를 문서화했습니다.

