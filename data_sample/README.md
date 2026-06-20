# Data Sample

이 폴더에는 원자료 전체를 포함하지 않습니다.

## Why Raw Data Is Not Included

- 외부 API와 웹 스크래핑 데이터의 재배포 가능성이 명확하지 않습니다.
- 뉴스 원문에는 저작권 이슈가 있을 수 있습니다.
- 원자료와 중간 산출물의 크기가 GitHub 포트폴리오 저장소에 적합하지 않습니다.
- API key가 필요한 수집 단계는 공개 저장소에 포함하지 않는 것이 안전합니다.

## Data Sources

- Market data: CoinMarketCap, CoinGecko, yfinance
- Macro data: FRED, Yahoo Finance
- On-chain and DeFi data: DefiLlama, MakerDAO/Curve/Aave related public data
- Attention data: Google Trends
- Text data: Reddit, CoinTelegraph, CoinDesk 등 공개 뉴스/게시글 기반 수집

## Variable Types

- Price and volatility variables
- Liquidity and volume variables
- Supply and market cap variables
- Macro risk variables
- On-chain and DeFi variables
- Search trend variables
- Keyword, LDA, and FinBERT-derived text variables

## Reproduction Note

전체 재실행을 위해서는 원본 `data/raw/`, `data/processed/`, `data/ml/` 구조가 필요합니다. 이 포트폴리오 레포는 분석 흐름과 핵심 코드 검토를 위한 정리본입니다.

