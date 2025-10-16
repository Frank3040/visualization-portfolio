import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import plotly.express as px
import plotly.graph_objects as go
from functools import lru_cache
try:
    import pycountry
except Exception:
    pycountry = None

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import ccf


def load_data(filepath):
    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(['Symbol', 'Date'])
    return df

def calculate_returns(prices):
    return np.log(prices / prices.shift(1))

def adf_test(series, name="Series"):
    from statsmodels.tsa.stattools import adfuller
    result = adfuller(series.dropna())
    return {
        'Test Statistic': result[0],
        'p-value': result[1],
        'Is Stationary': result[1] < 0.05
    }

def create_acf_plot_improved(series, title, nlags=40, color='#0066cc'):
    from statsmodels.tsa.stattools import acf
    import plotly.graph_objects as go
    series_clean = series.dropna()
    acf_values = acf(series_clean, nlags=nlags)
    conf_interval = 1.96 / np.sqrt(len(series_clean))
    fig = go.Figure()
    for i in range(len(acf_values)):
        fig.add_trace(go.Scatter(
            x=[i, i], y=[0, acf_values[i]],
            mode='lines', line=dict(color=color, width=2),
            showlegend=False,
            hovertemplate=f'Lag {i}<br>ACF: {acf_values[i]:.4f}<extra></extra>'
        ))
    fig.add_trace(go.Scatter(
        x=list(range(len(acf_values))), y=acf_values,
        mode='markers', marker=dict(size=6, color=color),
        showlegend=False
    ))
    fig.add_hline(y=conf_interval, line_dash="dash", line_color="red")
    fig.add_hline(y=-conf_interval, line_dash="dash", line_color="red")
    fig.add_hrect(y0=-conf_interval, y1=conf_interval, fillcolor="rgba(255,0,0,0.1)", layer="below")
    fig.update_layout(
        title=title, xaxis_title="Lag", yaxis_title="Autocorrelation",
        height=300, margin=dict(t=40, b=40, l=40, r=40)
    )
    return fig

def plot_price_and_returns(df_crypto, crypto_name):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05)
    fig.add_trace(go.Scatter(x=df_crypto['Date'], y=df_crypto['Close'], name='Price'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_crypto['Date'], y=df_crypto['Returns'], name='Returns'), row=2, col=1)
    fig.update_layout(height=500, showlegend=False, title=f"{crypto_name} - Price & Returns")
    return fig

def plot_pvalues_summary(df):
    cryptos = df['Symbol'].unique()
    results = []
    for crypto in cryptos:
        df_crypto = df[df['Symbol'] == crypto].copy()
        df_crypto['Returns'] = calculate_returns(df_crypto['Close'])
        adf_prices = adf_test(df_crypto['Close'])
        adf_returns = adf_test(df_crypto['Returns'].dropna())
        results.append({
            'Crypto': crypto,
            'Prices p-value': adf_prices['p-value'],
            'Returns p-value': adf_returns['p-value'],
            'Prices Stationary': adf_prices['Is Stationary'],
            'Returns Stationary': adf_returns['Is Stationary'],
            'Volatility': df_crypto['Returns'].std(),
            'Sharpe': (df_crypto['Returns'].mean() / df_crypto['Returns'].std() * np.sqrt(252)) if df_crypto['Returns'].std() != 0 else 0
        })
    return pd.DataFrame(results)


@lru_cache(None)
def iso2_to_iso3(iso2: str) -> str | None:
    if not isinstance(iso2, str) or len(iso2) != 2:
        return None
    iso2 = iso2.upper()
    if pycountry is not None:
        try:
            return pycountry.countries.get(alpha_2=iso2).alpha_3
        except Exception:
            pass
    fallback = {
        "US":"USA","MX":"MEX","BR":"BRA","AR":"ARG","CL":"CHL","CO":"COL","PE":"PER","VE":"VEN",
        "GB":"GBR","DE":"DEU","FR":"FRA","ES":"ESP","IT":"ITA","PT":"PRT","NL":"NLD","BE":"BEL",
        "JP":"JPN","CN":"CHN","IN":"IND","KR":"KOR","RU":"RUS","TR":"TUR","AU":"AUS","CA":"CAN",
    }
    return fallback.get(iso2)

@lru_cache(None)
def iso2_to_name(iso2: str) -> str | None:
    if not isinstance(iso2, str) or len(iso2) != 2:
        return None
    iso2 = iso2.upper()
    if pycountry is not None:
        try:
            return pycountry.countries.get(alpha_2=iso2).name
        except Exception:
            pass
    fallback = {
        "US":"United States of America","MX":"Mexico","BR":"Brazil","AR":"Argentina","CL":"Chile",
        "CO":"Colombia","PE":"Peru","VE":"Venezuela","GB":"United Kingdom","DE":"Germany","FR":"France",
        "ES":"Spain","IT":"Italy","PT":"Portugal","NL":"Netherlands","BE":"Belgium","JP":"Japan",
        "CN":"China","IN":"India","KR":"South Korea","RU":"Russia","TR":"Türkiye","AU":"Australia","CA":"Canada"
    }
    return fallback.get(iso2)

@st.cache_data
def load_trends():
    df = pd.read_csv("trends.csv", parse_dates=["date"])
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()
    df["iso3"] = df["country"].map(iso2_to_iso3)
    df["country_name"] = df["country"].map(iso2_to_name)
    df = df.dropna(subset=["iso3", "country_name"])
    return df

def _flag_url_iso2(iso2: str, size=40) -> str:
    return f"https://flagcdn.com/w{size}/{iso2.lower()}.png"

@st.cache_data
def st_load_interest():
    df = pd.read_csv("merged_interest_country.csv")
    df["date"] = pd.to_datetime(df["date"], errors="coerce", infer_datetime_format=True)
    df = df.dropna(subset=["date"])
    df["date"] = df["date"].dt.floor("d")
    if "country_name" not in df.columns:
        df["country_name"] = df["country"].map(iso2_to_name)
    return df

def merge_data(interest_df, prices_df, country="US", keyword="Bitcoin", symbol=None):
    df_i = interest_df[(interest_df["country"] == country) & (interest_df["keyword"] == keyword)].copy()
    if symbol is None:
        candidates = prices_df["Symbol"].unique().tolist()
        symbol = keyword if keyword in candidates else (candidates[0] if candidates else None)
    df_p = prices_df[prices_df["Symbol"] == symbol].copy()
    if "Date" in df_p.columns:
        df_p["Date"] = pd.to_datetime(df_p["Date"], utc=True, errors="coerce", infer_datetime_format=True)
        df_p["Date"] = df_p["Date"].dt.tz_localize(None).dt.floor("d")
    merged = pd.merge(df_i, df_p, left_on="date", right_on="Date", how="inner")
    if {"High","Low"}.issubset(merged.columns):
        merged["Volatility"] = merged["High"] - merged["Low"]
    else:
        merged["Volatility"] = merged["Close"].diff().abs().fillna(0)
    return merged

def fig_interest_vs_volatility(merged, country_name=""):
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=120)
    sns.scatterplot(ax=ax, data=merged, x="interest", y="Volatility")
    kw = merged["keyword"].iloc[0] if "keyword" in merged.columns and not merged.empty else ""
    ax.set_title(f"Interest vs Volatility ({country_name or merged.get('country',[None])[0]} - {kw})")
    ax.set_xlabel("Geographic interest (Google Trends)")
    ax.set_ylabel("Daily volatility (High - Low)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

def fig_lead_lag_ccf(merged, lags=30):
    x = merged["interest"].astype(float) - merged["interest"].astype(float).mean()
    y = merged["Volatility"].astype(float) - merged["Volatility"].astype(float).mean()
    ccf_vals = ccf(x, y)
    ccf_vals = ccf_vals[: max(1, lags)]
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=120)
    ax.stem(range(len(ccf_vals)), ccf_vals, use_line_collection=True)
    ax.set_title("Lead/Lag (CCF): Interest → Volatility")
    ax.set_xlabel("Lag (days) — positive: interest leads volatility")
    ax.set_ylabel("CCF")
    ax.axhline(0, color="black", linewidth=1)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

def fig_country_correlation(interest_df, keyword="Bitcoin", top_n=12):
    dfk = interest_df[interest_df["keyword"] == keyword].copy()
    pivot = (
        dfk.pivot_table(index="date", columns="country", values="interest", aggfunc="mean")
        .dropna(how="all", axis=1)
        .fillna(method="ffill")
        .fillna(method="bfill")
        .fillna(0)
    )
    var_rank = pivot.var().sort_values(ascending=False).head(top_n).index
    pivot = pivot[var_rank]
    corr = pivot.corr()
    fig, ax = plt.subplots(figsize=(8, 6), dpi=120)
    sns.heatmap(corr, cmap="coolwarm", annot=True, fmt=".2f", ax=ax, square=True, cbar_kws={"shrink": .8})
    ax.set_title(f"Country interest correlation — {keyword}")
    plt.tight_layout()
    return fig


st.set_page_config(page_title="Crypto Time Series Dashboard", layout="wide")

@st.cache_data
def load_and_process_data():
    df = load_data('crypto_prices_2020_2025_daily.csv')
    summary_df = plot_pvalues_summary(df)
    return df, summary_df

try:
    df, summary_df = load_and_process_data()
    cryptos = df['Symbol'].unique()
except FileNotFoundError:
    st.error("❌ File 'crypto_prices_2020_2025_daily.csv' not found. Make sure it exists in the working directory.")
    st.stop()

st.sidebar.title("⚙️ Controls")
selected_crypto = st.sidebar.selectbox("Select Cryptocurrency", cryptos)
show_all = st.sidebar.checkbox("Show All Cryptos in Overview", value=True)

tab_overview, tab_ts, tab_geo, tab_spatiotemporal, tab_insights = st.tabs(
    ["📊 Overview", "📈 Time Series", "🌍 Geographic", "⏳ Spatiotemporal", "💡 Insights"]
)

with tab_overview:
    st.header("Executive Overview")
    st.markdown("""
    > **Key Question**: *Are cryptocurrency prices predictable?*  
    This dashboard evaluates the **random walk hypothesis** and **volatility dynamics** across major cryptocurrencies.
    """)
    if show_all:
        cols = st.columns(len(cryptos))
        for i, crypto in enumerate(cryptos):
            row = summary_df[summary_df['Crypto'] == crypto].iloc[0]
            with cols[i]:
                st.metric(
                    label=f"{crypto}",
                    value="✅ Stationary" if row['Prices Stationary'] else "❌ Random Walk",
                    delta=f"σ={row['Volatility']:.4f}",
                    delta_color="off"
                )
        fig = px.bar(
            summary_df.melt(id_vars='Crypto', value_vars=['Prices p-value', 'Returns p-value'], var_name='Series', value_name='p-value'),
            x='Crypto', y='p-value', color='Series', barmode='group',
            title="ADF Test p-values (α = 0.05)",
            color_discrete_sequence=["#3498db", "#e74c3c"]
        )
        fig.add_hline(y=0.05, line_dash="dash", line_color="red", annotation_text="Significance Threshold")
        st.plotly_chart(fig, use_container_width=True)
    else:
        row = summary_df[summary_df['Crypto'] == selected_crypto].iloc[0]
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Price Stationarity", "✅ Yes" if row['Prices Stationary'] else "❌ No")
        col2.metric("Return Stationarity", "✅ Yes" if row['Returns Stationary'] else "❌ No")
        col3.metric("Annualized Volatility", f"{row['Volatility']*np.sqrt(252):.2%}")
        col4.metric("Sharpe Ratio", f"{row['Sharpe']:.2f}")

with tab_ts:
    st.header(f"Time Series Analysis: {selected_crypto}")
    df_crypto = df[df['Symbol'] == selected_crypto].copy()
    df_crypto['Returns'] = calculate_returns(df_crypto['Close'])
    fig_price = plot_price_and_returns(df_crypto, selected_crypto)
    st.plotly_chart(fig_price, use_container_width=True)
    st.subheader("Autocorrelation Analysis")
    col1, col2, col3 = st.columns(3)
    with col1:
        fig_acf_price = create_acf_plot_improved(df_crypto['Close'], "ACF: Prices", color='#0066cc')
        st.plotly_chart(fig_acf_price, use_container_width=True)
    with col2:
        fig_acf_ret = create_acf_plot_improved(df_crypto['Returns'], "ACF: Returns", color='#ff6600')
        st.plotly_chart(fig_acf_ret, use_container_width=True)
    with col3:
        fig_acf_vol = create_acf_plot_improved(df_crypto['Returns']**2, "ACF: Squared Returns (Volatility)", color='#00aa00')
        st.plotly_chart(fig_acf_vol, use_container_width=True)

with tab_geo:
    st.header("Geographic Analysis")
    try:
        geo_df = load_trends()
    except FileNotFoundError:
        st.error("❌ File 'trends.csv' not found in the project folder.")
        st.stop()
    st.subheader("Average Global Cryptocurrency Interest from 2020 to 2025")
    avg_country = (
        geo_df.groupby("iso3", as_index=False)["interest"].mean()
        .rename(columns={"interest": "avg_interest"})
    )
    fig_static = px.choropleth(
        avg_country,
        locations="iso3",
        color="avg_interest",
        color_continuous_scale="YlOrBr",
        projection="natural earth",
        title=""
    )
    fig_static.update_geos(
        bgcolor='rgba(0,0,0,0)',
        landcolor="rgba(50,50,50,0.30)",
        oceancolor="rgba(0,0,0,0)",
        showframe=False,
        showcoastlines=True,
        coastlinecolor="gray"
    )
    fig_static.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=10, r=10, b=10, l=10),
        coloraxis_colorbar=dict(title="Avg. Interest")
    )
    st.plotly_chart(fig_static, use_container_width=True)
    st.subheader("Evolution of Average Interest in Cryptocurrencies")
    monthly_country = (
        geo_df.groupby(["month","iso3"], as_index=False)["interest"].mean()
        .rename(columns={"interest":"avg_interest"})
    )
    monthly_country["month_label"] = monthly_country["month"].dt.strftime("%b-%Y")
    fig_anim = px.choropleth(
        monthly_country,
        locations="iso3",
        color="avg_interest",
        animation_frame="month_label",
        color_continuous_scale="YlOrBr",
        range_color=(0, monthly_country["avg_interest"].max()),
        projection="natural earth",
        title=""
    )
    fig_anim.update_geos(
        bgcolor='rgba(0,0,0,0)',
        landcolor="rgba(50,50,50,0.30)",
        oceancolor="rgba(0,0,0,0)",
        showframe=False,
        showcoastlines=True,
        coastlinecolor="gray"
    )
    fig_anim.update_layout(updatemenus=[])
    if getattr(fig_anim.layout, "sliders", None) and len(fig_anim.layout.sliders) > 0:
        fig_anim.layout.sliders[0].update(currentvalue=dict(visible=False, prefix=""))
    fig_anim.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=10, r=10, b=10, l=10),
        coloraxis_colorbar=dict(title="Avg. Interest")
    )
    st.plotly_chart(fig_anim, use_container_width=True)
    st.subheader("Interest in Cryptocurrencies by Country")
    top_keywords = ["Bitcoin", "Cardano", "Dogecoin", "Ethereum", "Ripple"]
    geo_top = geo_df[geo_df["keyword"].isin(top_keywords)].copy()
    countries_opts = sorted(geo_df["country_name"].unique())
    country_sel_name = st.selectbox("Country", countries_opts, index=0, key="geo_country")
    ts = (
        geo_top[geo_top["country_name"] == country_sel_name]
        .groupby(["month", "keyword"], as_index=False)["interest"].mean()
        .rename(columns={"interest":"avg_interest"})
        .sort_values("month")
    )
    fig_lines = px.line(
        ts, x="month", y="avg_interest", color="keyword",
        line_shape="linear", markers=False,
        labels={"month": "Date", "avg_interest": "Average Interest", "keyword": "Cryptocurrencies"},
        title=f"Interest in Cryptocurrencies in {country_sel_name}"
    )
    fig_lines.update_layout(margin=dict(t=60, r=10, b=10, l=10))
    st.plotly_chart(fig_lines, use_container_width=True)
    st.subheader("Top 10 Countries by Average Interest in Cryptocurrencies (2020–2025)")
    top10 = (
        geo_df.groupby(["country","country_name"], as_index=False)["interest"].mean()
        .rename(columns={"interest":"avg_interest"})
        .sort_values("avg_interest", ascending=False).head(10)
    )
    fig_top10 = go.Figure(
        data=[go.Bar(x=top10["country_name"], y=top10["avg_interest"], marker=dict(color="#CC6A00"))]
    )
    fig_top10.update_traces(text=[f"{v:.2f}" for v in top10["avg_interest"]], textposition="outside")
    fig_top10.update_layout(
        xaxis_title="Country", yaxis_title="Average Interest",
        margin=dict(t=60, r=20, b=60, l=60),
        yaxis=dict(range=[0, top10["avg_interest"].max() * 1.25]),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    for iso2, name, val in zip(top10["country"], top10["country_name"], top10["avg_interest"]):
        fig_top10.add_layout_image(
            dict(
                source=_flag_url_iso2(iso2, size=40),
                xref="x", yref="y", x=name, y=val * 0.5,
                xanchor="center", yanchor="middle",
                sizex=0.6, sizey=val * 0.4,
                sizing="contain", layer="above", opacity=1.0
            )
        )
    st.plotly_chart(fig_top10, use_container_width=True)

with tab_spatiotemporal:
    st.header("Spatiotemporal: Geographic Interest ↔ Price Volatility")
    try:
        interest_df = st_load_interest()
    except FileNotFoundError:
        st.error("❌ File 'merged_interest_country.csv' not found in the project folder.")
        st.stop()
    colA, colB, colC, colD = st.columns([1, 1, 1, 1])
    with colA:
        countries = sorted(interest_df["country"].dropna().unique().tolist())
        country_sel = st.selectbox("Country (ISO2)", countries, index=0, key="st_country")
        country_name = iso2_to_name(country_sel) or country_sel
    with colB:
        keywords = sorted(interest_df["keyword"].dropna().unique().tolist())
        default_kw_idx = keywords.index("Bitcoin") if "Bitcoin" in keywords else 0
        keyword_sel = st.selectbox("Keyword", keywords, index=default_kw_idx, key="st_keyword")
    with colC:
        symbols = sorted(df["Symbol"].dropna().unique().tolist())
        default_sym = keyword_sel if keyword_sel in symbols else (symbols[0] if symbols else None)
        idx_sym = symbols.index(default_sym) if default_sym in symbols else 0
        symbol_sel = st.selectbox("Symbol (prices)", symbols, index=idx_sym, key="st_symbol")
    with colD:
        lags = st.slider("Lags (CCF)", min_value=5, max_value=60, value=30, step=5)

    merged = merge_data(interest_df, df, country=country_sel, keyword=keyword_sel, symbol=symbol_sel)
    if merged.empty:
        st.warning("No overlapping dates for this combination. Try another country/keyword/symbol.")
    else:
        with st.expander("Quick preview of the merge (first rows)"):
            cols_to_show = [c for c in ["date","country","keyword","interest","Symbol","Date","Open","High","Low","Close"] if c in merged.columns]
            st.dataframe(merged[cols_to_show].head(20), use_container_width=True)

        st.subheader("Interest vs Volatility")
        fig1 = fig_interest_vs_volatility(merged, country_name=country_name)
        st.pyplot(fig1, use_container_width=True)

        st.subheader("Lead/Lag — Does interest precede volatility?")
        fig2 = fig_lead_lag_ccf(merged, lags=lags)
        st.pyplot(fig2, use_container_width=True)
        st.caption("Positive lags suggest changes in interest may precede changes in volatility.")

        st.subheader("Country Interest Correlation")
        topn = st.slider("Number of countries (highest variance) for the heatmap", 6, 20, 12, 1)
        fig3 = fig_country_correlation(interest_df, keyword=keyword_sel, top_n=topn)
        st.pyplot(fig3, use_container_width=True)

        st.info("If you see CCF spikes at +k lags, consider position-sizing or options rules k days after interest spikes.")

with tab_insights:
    st.header("Key Insights & Implications")
    row = summary_df[summary_df['Crypto'] == selected_crypto].iloc[0]
    is_random_walk = not row['Prices Stationary']
    if is_random_walk:
        st.success("✅ Prices follow a Random Walk (p > 0.05)")
        st.markdown("""
        - Past prices do not predict future prices.
        - Technical analysis is unlikely to yield consistent alpha.
        - Focus on long-term fundamentals or on-chain metrics instead.
        """)
    else:
        st.warning("⚠️ Prices show stationarity – potential predictability (rare for cryptos).")
    st.subheader("Volatility Clustering")
    st.markdown("""
    - Squared returns exhibit significant autocorrelation → volatility clusters in time.
    - While you can't predict direction, you can anticipate high-volatility periods.
    - Consider volatility-based position sizing or options during calm periods.
    """)
    st.subheader("Return Distribution")
    st.markdown("""
    - Crypto returns often show fat tails and negative skew.
    - Risk management is critical: large downside moves are more likely than under Gaussian assumptions.
    """)
    st.subheader("Geographic")
    st.markdown("""
    - The global map highlights the highest average interest in **Japan**, **South Korea**, and **Germany**, followed by **Brazil, the United Kingdom, the United States, and Mexico**.  
    - Worldwide attention peaked between **2021 and 2022**, aligning with the crypto market boom.  
    - Bitcoin and Ethereum dominate searches; Dogecoin shows short spikes tied to viral events.  
    - The Top 10 ranking mixes technological powers with emerging economies, suggesting attention follows both digital development and speculation cycles.
    """)
    
    st.subheader("Spatiotemporal")
    st.markdown("""
    - The **cross-correlation (CCF)** shows that in most countries, **spikes in public interest lead market volatility by around 2–5 days**.  
    - This suggests that **Google Trends data can act as an early signal** of upcoming turbulence in crypto prices.  
    - The **correlation heatmap** indicates that attention patterns are globally synchronized — especially between the **US, UK, Germany, and Japan**.  
    - Overall, **social attention drives volatility**, not direction: when interest surges, short-term uncertainty increases.
    """)


st.markdown("---")
st.caption(f"Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M')} | Data: crypto_prices_2020_2025_daily.csv, trends.csv & merged_interest_country.csv")
