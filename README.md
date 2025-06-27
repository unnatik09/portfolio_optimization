# portfolio_optimization
📈 Predictive Portfolio Optimizer using LGBM + Ridge regression, enhanced with FinBERT sentiment overlay and Sharpe-maximizing allocation via PyPortfolioOpt. Includes comparative backtesting, technical feature engineering, and daily rebalancing.
<pre><code>## 📘 Predictive Portfolio Optimization with Sentiment Overlay

---

### 📌 Overview

This project implements a **machine learning-powered portfolio optimizer** using predicted 5-day returns from financial and technical signals, enhanced by a **news sentiment overlay**.  
It allocates weights to the top 30 NIFTY 50 stocks using **LGBM + Ridge regression ensemble**, optimizing for **Sharpe Ratio** via the `PyPortfolioOpt` library.

---

### 🔧 Pipeline Steps

1. **Feature Engineering**  
   - Extracted technical indicators (RSI, MACD, ADX, CCI, BB Width, etc.)
   - Computed rolling volatility, returns, and market regime (bull/bear/neutral)

2. **Target Construction**  
   - Calculated 5-day forward returns for each stock (future price % change)

3. **Modeling**  
   - Trained a `MultiOutputRegressor` ensemble:  
     `y_pred = 0.7 × LGBM + 0.3 × Ridge`
   - LGBM captured nonlinearities efficiently, Ridge reduced overfitting.

4. **Sentiment Analysis**  
   - Fetched last 30 days of headlines per stock using `NewsAPI`
   - Scored using `FinBERT` (PyTorch)  
   - Adjusted predicted returns as:  
     μ<sub>sent</sub> = μ<sub>pred</sub> × (1 + α × sentiment), with α = 0.5

5. **Portfolio Optimization**  
   - Used `EfficientFrontier` (PyPortfolioOpt) to maximize Sharpe ratio  
   - Performed optimization both with and without sentiment overlay  
   - Simulated and compared cumulative returns and metrics

---

### 📊 Performance Metrics *(example — update later)*

| Metric                | Base Portfolio | With Sentiment |
|-----------------------|----------------|----------------|
| Cumulative Return     | 17.1%          | **19.3%**      |
| Annualized Return     | 41.2%          | 44.0%          |
| Annualized Volatility | 19.8%          | 19.5%          |
| Sharpe Ratio          | 2.08           | **2.23**       |
| Max Drawdown          | -30.9%         | -28.6%         |

---

### 📈 Visualizations

- 📉 Technical Feature Line Plots (RSI, MACD, ADX, Volatility)
- 🧊 Market Regime Heatmap
- 🔥 Sector-wise Correlation Heatmaps
- ⚖️ Portfolio Allocation (Pie Chart)
- 📈 Cumulative Return Comparison (ML vs Equal vs NIFTY)
- 📰 Sentiment-based Portfolio Comparison (Last 30 Days)
- 📊 30-Day Rolling Sharpe Ratio

---

### 💡 Modeling Notes

- **LGBM + Ridge** performed better than LGBM or Ridge alone  
- **PCA** increased overfitting, hence excluded  
- **Optuna** tuning gave very high train R² (~1.0) but low test R² (~0.4–0.5), indicating overfit  
- **MLP** underperformed due to the **non-temporal**, tabular nature of data  
- **LSTM/Transformer** models were not suitable here  
- Final ensemble generalizes well with balanced bias-variance

---

### 🆕 What’s Unique in This Project?

- ✅ **Sentiment Overlay Integration** — FinBERT-enhanced predictions  
- ✅ **Comparative Backtesting** with baseline portfolios  
- ✅ **Sector-wise Correlation Analysis**  
- ✅ **Rolling Sharpe Visualization**  
- ✅ **End-to-End Rebalancing Pipeline**, suitable for daily automation

---

### 📂 Files

| File                        | Purpose                               |
|-----------------------------|----------------------------------------|
| `main_notebook.ipynb`       | Full pipeline from data to simulation |
| `weights_YYYY-MM-DD.csv`    | Sentiment-adjusted weights (daily)    |
| `allocation_YYYY-MM-DD.csv` | ₹1L Allocation based on weights       |
| `daily_returns.csv`         | Historical returns (used for backtest)|
</code></pre>
