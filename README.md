# 📈 Crypto Time Series Dashboard
**Temporal, Geographic, and Spatiotemporal Analysis of Cryptocurrencies (2020–2025)**

This project implements an **interactive Streamlit dashboard** that analyzes the behavior of major cryptocurrencies across three analytical dimensions: **temporal**, **geographic**, and **spatiotemporal**, using historical price and global search trend data.

It explores hypotheses such as the **Random Walk**, **Volatility Clustering**, and the **Influence of Public Interest on Price Volatility**.

---

## 🧩 Contents
1. [Main Features](#main-features)
2. [Architecture and Libraries](#architecture-and-libraries)
3. [File Structure](#file-structure)
4. [Program Flow](#program-flow)
5. [Statistical Analysis](#statistical-analysis)
6. [Visualizations](#visualizations)
7. [Dashboard Sections](#dashboard-sections)
8. [Running the Project](#running-the-project)
9. [Environment Requirements](#environment-requirements)
10. [Analytical Conclusions](#analytical-conclusions)

---

## 🚀 Main Features
- **Evaluation of the Random Walk Hypothesis** using ADF tests.
- **Autocorrelation (ACF)** analysis on prices, returns, and volatility.
- **Global geographic visualization** of cryptocurrency interest (Google Trends).
- **Spatiotemporal analysis** of interest vs. volatility using CCF.
- **Modular dashboard** with interactive tabs for each analytical section.
- **Full compatibility** with Plotly, Seaborn, and Matplotlib for rich visualizations.

---

## 🧠 Architecture and Libraries
The project integrates data analysis, visualization, and efficient caching.

| Category | Main Libraries |
|-----------|----------------|
| Interactive visualization | `plotly`, `streamlit` |
| Statistical analysis | `numpy`, `pandas`, `statsmodels` |
| Graphs and plots | `matplotlib`, `seaborn` |
| Geographic data | `pycountry`, `plotly.express` |
| Performance optimization | `@st.cache_data`, `@lru_cache` |

---

## 📁 File Structure
```
📦 CryptoDashboard
 ┣ 📜 app.py                          # Main dashboard code
 ┣ 📊 crypto_prices_2020_2025_daily.csv   # Historical crypto prices
 ┣ 📊 trends.csv                      # Global search trends (Google Trends)
 ┣ 📊 merged_interest_country.csv     # Combined interest–volatility dataset
 ┣ 🧾 README.md                       # Project documentation
 ┗ 📸 /assets/                        # Optional images or resources
```

---

## 🔄 Program Flow
1. **Data Loading** (`load_data`, `load_trends`, `st_load_interest`)  
   Imports and normalizes CSV data with proper date and country code formatting.

2. **Statistical Processing**  
   - Logarithmic returns (`calculate_returns`).
   - Stationarity tests (`adf_test`).
   - Volatility and Sharpe ratio calculations.

3. **Visualization Generation**  
   - Bar charts, line plots, choropleths, and correlation heatmaps.
   - Specialized functions for ACF, CCF, and volatility dynamics.

4. **Streamlit Rendering**  
   - Five analytical tabs: Overview, Time Series, Geographic, Spatiotemporal, Insights.

---

## 📊 Statistical Analysis
- **ADF (Augmented Dickey-Fuller)**: Tests for stationarity of price and return series.  
- **ACF (Autocorrelation Function)**: Measures time dependencies.  
- **CCF (Cross-Correlation Function)**: Tests whether public interest leads price volatility.  
- **Sharpe Ratio**: Evaluates risk-adjusted return per cryptocurrency.

---

## 📈 Visualizations
| Type | Library | Description |
|------|----------|-------------|
| Prices & returns | Plotly | Time series of closing prices and log returns |
| ACF | Plotly | Autocorrelation of prices, returns, and squared returns |
| Choropleth (static) | Plotly Express | Average global interest map |
| Choropleth (animated) | Plotly Express | Monthly evolution of interest (2020–2025) |
| Top 10 bars | Plotly / Go | Top countries by average interest |
| Scatter interest–volatility | Seaborn | Correlation between search attention and volatility |
| Correlation heatmap | Seaborn | Co-movements in interest among countries |

---

## 🗂️ Dashboard Sections

### 1. 📊 Overview
- Price and return stationarity.  
- p-values (ADF tests).  
- Volatility and Sharpe metrics.

### 2. 📈 Time Series
- Price and return charts.  
- ACF of prices, returns, and volatility.

### 3. 🌍 Geographic
- Global interest map (choropleth).  
- Temporal evolution animation.  
- Interest by country and Top 10 ranking.

### 4. ⏳ Spatiotemporal
- Scatter of interest vs. volatility.  
- Lead-lag CCF correlations.  
- Heatmap of cross-country interest correlations.

### 5. 💡 Insights
- Random walk and predictability analysis.  
- Volatility clustering effects.  
- Return distribution asymmetries.  
- Geographic and spatiotemporal insights summary.

---

## ⚙️ Running the Project
### 1. Clone the repository
```bash
git clone https://github.com/user/crypto-timeseries-dashboard.git
cd crypto-timeseries-dashboard
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the dashboard
```bash
streamlit run app.py
```

### 4. Access the app
Visit:  
```
http://localhost:8501
```

---

## 🧾 Environment Requirements
- **Python 3.10+**
- Required packages:
  ```bash
  streamlit pandas numpy matplotlib seaborn statsmodels plotly pycountry
  ```
- Required data files:
  - `crypto_prices_2020_2025_daily.csv`
  - `trends.csv`
  - `merged_interest_country.csv`

---

## 📍 Analytical Conclusions
- Most cryptocurrencies **follow a Random Walk**, with non-stationary prices.  
- **Returns are stationary**, supporting efficient market behavior.  
- **Volatility clustering** indicates high-variance periods grouped in time.  
- **Public interest (Google Trends)** leads price volatility by **2–5 days**, suggesting predictive potential.  
- **Interest synchronization** between the US, UK, Germany, and Japan implies **global behavioral alignment**.  

---

## 🧩 Author & License
Developed by **Francisco Chan, Valeria Ramírez, Esther Apaza, Carlos Helguera, Daniel Valdes, Rogelio Novelo**.  
Open for academic and research purposes.  
© 2025 — All rights reserved.
