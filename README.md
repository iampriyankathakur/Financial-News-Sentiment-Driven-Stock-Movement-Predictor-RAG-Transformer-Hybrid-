# Financial News Sentiment Driven Stock Movement Predictor

## 📌 Problem Statement
Financial markets react rapidly to informational signals embedded in textual news streams. Traditional quantitative models rely primarily on historical price movements and fail to incorporate semantic market sentiment derived from financial narratives.

This project proposes a hybrid predictive framework that integrates:

- Financial News Sentiment (FinBERT)
- Retrieval-Augmented Contextual News Matching
- Technical Market Indicators
- Transformer Based Feature Fusion

to predict next-day stock price directional movement.


## 🧠 Methodology

### 1. News Sentiment Extraction
We use FinBERT (domain-adapted BERT model trained on financial corpora) to compute entity-specific sentiment scores from financial news headlines and descriptions.

Output Sentiment Vector:
- Positive
- Neutral
- Negative


### 2. Retrieval-Augmented Generation (RAG)

News embeddings are generated using:

SentenceTransformer → FAISS Vector Database

For each prediction day:

Relevant historical news context is retrieved using semantic similarity search.

This helps incorporate:
- Thematic similarity
- Market regime behaviour
- Narrative recurrence


### 3. Market Technical Indicators

Extracted using TA-Lib style indicators:

- RSI
- MACD
- 10 Day SMA
- Volatility
- Momentum Index


### 4. Feature Fusion

Final Feature Vector:

| Feature Type | Description |
|-------------|-------------|
| Sentiment Score | FinBERT Output |
| RSI | Momentum Oscillator |
| MACD | Trend Indicator |
| SMA | Price Mean |
| Volatility | Risk Proxy |

### 5. Transformer Classifier

Fully connected deep fusion model used to classify:

- 0 → Bearish Movement
- 1 → Bullish Movement

Loss Function:
Cross Entropy Loss

Optimizer:
Adam


## 📊 Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score


## 🔍 Explainable AI

SHAP values are computed to:

- Interpret prediction drivers
- Identify feature dominance
- Understand sentiment vs technical influence


## 📈 Backtesting

Strategy simulated using:

Signal(t) → Position(t+1)

Cumulative returns compared against:

- Buy & Hold Strategy


## 📌 Future Work

- Temporal Attention Transformer
- Cross-Asset Sentiment Spillover
- LLM Based Narrative Reasoning Layer
- Event Driven Volatility Modelling


## ▶️ Run

```bash
pip install -r requirements.txt
streamlit run app/dashboard.py
