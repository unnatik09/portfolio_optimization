# 📊 Predictive Portfolio Optimization using ML + Sentiment (NIFTY 30)

This project builds a predictive portfolio allocation engine for the top 30 NIFTY stocks by:
- Extracting technical & market regime features
- Predicting short-term returns using a blended ML model (LGBM + Ridge)
- Optimizing portfolio weights using `PyPortfolioOpt` to maximize Sharpe ratio
- Enhancing allocation with **daily sentiment overlay** via FinBERT
- Comparing performance against Equal-Weighted and NIFTY50 index

---

## 🧠 Model Summary

| Model Component    | Description                                      |
|--------------------|--------------------------------------------------|
| Features           | Technical indicators (RSI, MACD, ADX, BBW, etc.), Market regime, Volume shifts |
| Targets            | 5-day forward returns for each stock             |
| ML Models          | Ensemble: 70% LightGBM + 30% Ridge Regression    |
| Optimizer          | Efficient Frontier (Max Sharpe via PyPortfolioOpt) |
| Sentiment Overlay  | FinBERT on last 30 days of news headlines        |
| Evaluation         | R² (train/test), Sharpe ratio, cumulative returns |
| Backtests          | ML Portfolio vs Equal-Weight vs NIFTY Index      |

---

## 🔍 ML Model R² Performance

| Metric         | Average (30 Stocks) |
|----------------|---------------------|
| Train R²       | ~0.73               |
| Test R²        | ~0.55               |

(See full breakdown in notebook.)

---

## 💼 Portfolio Performance (5-Year Backtest)

| Metric              | Value             |
|---------------------|------------------:|
| **Cumulative Return** | 357.22%         |
| **Annual Return**     | 41.17%          |
| **Volatility**        | 19.78%          |
| **Sharpe Ratio**      | 2.08            |
| **Max Drawdown**      | -30.92%         |

---

## 📰 Impact of Sentiment (Last 30 Days Only)

| Metric         | Base Portfolio | With Sentiment |
|----------------|----------------|----------------|
| Sharpe Ratio   | **1.83**       | **1.91**       |
| Performance    | Adjusted only for last 30 days | — |

> ✅ Sentiment-adjusted weights slightly outperform in Sharpe ratio, especially under volatile conditions.

---

## 📈 Plots Included

- Rolling Sharpe Ratio
- Cumulative Returns (ML vs Equal-Weight vs NIFTY)
- Feature Exploratory Analysis (RSI, MACD, Volatility, etc.)
- Market Regime Heatmaps
- Correlation Heatmaps (sector-wise)
- Portfolio Allocation Pie Chart

---

## 📦 Files

| File                        | Description                            |
|-----------------------------|----------------------------------------|
| `final_notebook.ipynb`      | All code: feature engineering, training, optimization, sentiment |
| `daily_returns.csv`         | Historical daily returns for 30 stocks |
| `weights_YYYY-MM-DD.csv`    | Final optimized weights (saved daily)  |
| `allocation_YYYY-MM-DD.csv` | ₹1L allocation based on sentiment-adjusted weights |
| `portfolio_comparison.png`  | Cumulative returns chart               |

---

## 🚀 Future Improvements

- Add sector exposure control during optimization
- Incorporate macroeconomic indicators (CPI, GDP, etc.)
- Collect daily sentiment for longer horizon backtests
- Try reinforcement learning for dynamic rebalancing

---

## 📚 Acknowledgements

- [`yiyanghkust/finbert-tone`](https://huggingface.co/yiyanghkust/finbert-tone) for sentiment
- `yfinance`, `scikit-learn`, `LightGBM`, `PyPortfolioOpt`, `NewsAPI`
