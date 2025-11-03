#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import matplotlib.pyplot as plt
from typing import List
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler
import circlify
from plotly.colors import sample_colorscale
import math

# =========================
# PAGE CONFIG & STYLE
# =========================
st.set_page_config(
    page_title="World Population — Analytics",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="collapsed",
)
# Colores de la página
PRIMARY = "#0B3C5D"
PAGE_BG = "#0e1117"
TEXT_LIGHT = "#e2e8f0"
# Fondo global
st.markdown(
    f"""
    <style>
      html, body, .stApp {{
        background-color: {PAGE_BG} !important;
      }}
      .block-container {{
        padding-top: 1.2rem;
      }}
      .stTabs [data-baseweb="tab"] {{
        font-weight:600;
        color: "{PRIMARY}";
      }}
    </style>
    """,
    unsafe_allow_html=True,
)
# Evitar transparencia en el dropdown de Plotly
st.markdown(
    """
    <style>
    .plotly .updatemenu-bg { fill: #1e1e1e !important; stroke: #3b82f6 !important; }
    .plotly .updatemenu-item-rect { fill: #1e1e1e !important; stroke: #1f2937 !important; }
    .plotly .updatemenu-item-rect:hover { fill: #3b82f6 !important; stroke: #3b82f6 !important; }
    .plotly .updatemenu-item-text { fill: #f8fafc !important; }
    .plotly .updatemenu-button, .plotly .updatemenu-button rect { fill: #1e1e1e !important; stroke: #3b82f6 !important; }
    .plotly .updatemenu-button:hover rect { fill: #3b82f6 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)
# =========================
# VIRIDIS PERSONALIZADA (para GDP)
# =========================
COLOR_PALETTE = {
    "color_continuous_scale": "Viridis",
    "Viridis": [
        "#440154", "#482777", "#3E4989", "#31688E", "#26828E",
        "#1F9E89", "#35B779", "#6DCD59", "#B4DE2C", "#FDE725"
    ],
    "root_color": "lightgray"
}
# =========================
# DATA HANDLING
# =========================
REQUIRED_COLS = ["Country_Name", "Country_Code", "Continent", "Population", "GDP"]
@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)
def validate_columns(df: pd.DataFrame, required: List[str]) -> List[str]:
    return [c for c in required if c not in df.columns]
def clean_and_enrich(df: pd.DataFrame) -> pd.DataFrame:
    df["Population"] = pd.to_numeric(df["Population"], errors="coerce")
    df["GDP"] = pd.to_numeric(df["GDP"], errors="coerce")
    df = df.dropna(subset=["Population", "GDP", "Continent", "Country_Name"])
    df = df[df["Population"] > 0].copy()
    df["Population_adj"] = df["Population"] ** 0.72
    med_gdp = df["GDP"].replace(0, np.nan).median()
    df["GDP"] = df["GDP"].replace(0, np.nan).fillna(med_gdp)
    df["GDP_per_capita"] = df["GDP"] / df["Population"]
    return df
# =========================
# TREEMAP (VIRIDIS)
# =========================
def build_treemap(df: pd.DataFrame, height: int = 900) -> go.Figure:
    COLOR_SCALE = COLOR_PALETTE["Viridis"]
    def treemap_trace(data: pd.DataFrame, parent_label: str) -> go.Treemap:
        fig_tmp = px.treemap(
            data,
            path=["ContinentView", "Country_Name"],
            values="Population_adj",
            color="GDP",
            color_continuous_scale=COLOR_SCALE,
            hover_data={"Country_Code": True, "Population": ":,", "GDP": ":,.2f"},
        )
        tr = fig_tmp.data[0]
        tr.name = parent_label
        tr.visible = False
        tr.textinfo = "label"
        tr.textfont = dict(size=14, color="#f8fafc")
        tr.tiling = dict(pad=2, packing="squarify")
        return tr
    traces = []
    continent_sets = [
        (["North America", "South America"], "America (NA+SA)"),
        (["Africa"], "Africa"),
        (["Asia"], "Asia"),
        (["Europe"], "Europe"),
        (["Oceania"], "Oceania"),
    ]
    for conts, label in continent_sets:
        subset = df[df["Continent"].isin(conts)].copy()
        subset["ContinentView"] = label.split()[0]
        traces.append(treemap_trace(subset, label))
    agg = (
        df.groupby("Continent", as_index=False)
          .agg(Population=("Population", "sum"), GDP=("GDP", "median"))
    )
    fig_tmp_agg = px.treemap(
        agg,
        path=["Continent"],
        values="Population",
        color="GDP",
        color_continuous_scale=COLOR_SCALE,
        hover_data={"Population": ":,", "GDP": ":,.2f"},
    )
    tr_agg = fig_tmp_agg.data[0]
    tr_agg.name = "All Continents "
    tr_agg.visible = False
    tr_agg.textinfo = "label"
    tr_agg.textfont = dict(size=14, color="#f8fafc")
    tr_agg.tiling = dict(pad=2, packing="squarify")
    traces.append(tr_agg)
    fig = go.Figure(traces)
    for i in range(len(fig.data)):
        fig.data[i].visible = (i == 0)
    labels = [
        "North and South America ", "Africa", "Asia", "Europe", "Oceania", "All Continents",
    ]
    buttons = []
    for i, label in enumerate(labels):
        vis = [False] * len(traces); vis[i] = True
        buttons.append(
            dict(
                method="update",
                label=label,
                args=[
                    {"visible": vis},
                    {"title": {"text": f"World Population Treemap — {label.strip()} ",
                               "y": 0.98, "x": 0.5, "xanchor": "center", "yanchor": "top"}},
                ],
            )
        )
    fig.update_layout(
        height=height,
        margin=dict(t=80, l=6, r=6, b=6),
        title=dict(
            text="World Population Treemap — America (North and South) ",
            y=0.98, x=0.5, xanchor="center", yanchor="top",
            font=dict(color=TEXT_LIGHT),
        ),
        updatemenus=[dict(
            type="dropdown", direction="down",
            x=0.01, y=1.06, xanchor="left", yanchor="top",
            showactive=True, buttons=buttons,
            bgcolor="#1e1e1e", bordercolor="#3b82f6",
            font=dict(color="#f8fafc"),
        )],
        coloraxis=dict(
            colorscale=COLOR_SCALE,
            cmin=float(df["GDP"].min()),
            cmax=float(df["GDP"].max()),
            colorbar=dict(
                title=dict(text="GDP per capita", font=dict(color="#f8fafc")),
                tickfont=dict(color="#f8fafc"),
                outlinecolor="#334155",
            ),
        ),
        uniformtext=dict(minsize=10, mode="show"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig
# =========================
# SUNBURST (CIRCULAR TREEMAP - PLOTLY)
# =========================
def build_sunburst(df: pd.DataFrame) -> go.Figure:
    fig = px.sunburst(
        df,
        path=["Continent", "Country_Name"],
        values="Population",
        color="GDP_per_capita",
        color_continuous_scale="Viridis",
        hover_data={"Country_Code": True, "Population": ":,", "GDP_per_capita": ":,.0f", "GDP": ":,.0f"},
        title="Circular Treemap (Sunburst) — Population (size) • GDP per Capita (color)",
    )
    fig.update_layout(
        margin=dict(t=80, l=6, r=6, b=6),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT_LIGHT),
        coloraxis_colorbar=dict(title="GDP per Capita", tickfont=dict(color=TEXT_LIGHT)),
    )
    return fig
# =========================
# CIRCULAR TREEMAP (CIRCLE PACKING) — 100% PLOTLY
# =========================
def build_circle_packing_plotly(df: pd.DataFrame) -> go.Figure:
    """
    Size = Population • Color = GDP per capita (log-mapped to Viridis)
    Dibuja TODO con Plotly (shapes + anotaciones + colorbar).
    """

    # --- datos limpios
    d = df.copy()
    d["Population"] = pd.to_numeric(d["Population"], errors="coerce")
    d["GDP"] = pd.to_numeric(d["GDP"], errors="coerce")
    d = d.dropna(subset=["Population", "GDP", "Continent", "Country_Name"])
    d = d[d["Population"] > 0].copy()
    d["GDP_per_capita"] = d["GDP"] / d["Population"]

    # --- jerarquía para circlify (usa 'datum' en TODOS los niveles)
    def _sum_pop(cont):
        return float(d.loc[d["Continent"] == cont, "Population"].sum())

    hierarchy = [
        dict(
            id="World",
            datum=float(d["Population"].sum()),
            children=[
                dict(
                    id=cont,
                    datum=_sum_pop(cont),
                    children=[
                        dict(
                            id=row["Country_Name"],
                            datum=float(row["Population"]),
                            gdp_pc=float(row["GDP_per_capita"]),
                        )
                        for _, row in d[d["Continent"] == cont].iterrows()
                        if float(row["Population"]) > 0
                    ],
                )
                for cont in sorted(d["Continent"].unique())
            ],
        )
    ]

    # --- layout de círculos
    circles = circlify.circlify(hierarchy, show_enclosure=True)

    # --- escala de color (log)
    eps = 1e-6
    vmin = max(float(d["GDP_per_capita"].min()), eps)
    vmax = float(d["GDP_per_capita"].max())
    def _to_unit_log(val: float) -> float:
        # normaliza log10 a [0,1]
        val = max(val, vmin)
        return (math.log10(val) - math.log10(vmin)) / (math.log10(vmax) - math.log10(vmin))

    # Colores y estilo
    viridis = COLOR_PALETTE["Viridis"]
    EDGE = "#223042"
    CONT_FILL = "#1d2a3b"
    CONT_ALPHA = 0.08

    shapes = []
    annotations = []

    for c in circles:
        x, y, r = c.x, c.y, c.r
        ex = getattr(c, "ex", None)

        # nivel 0: borde del mundo
        if c.level == 0:
            shapes.append(dict(
                type="circle", xref="x", yref="y",
                x0=x - r, x1=x + r, y0=y - r, y1=y + r,
                line=dict(color=EDGE, width=1.2), fillcolor="rgba(0,0,0,0)"
            ))
            continue

        is_leaf = bool(ex) and ("gdp_pc" in ex)
        is_continent = bool(ex) and (ex.get("children") is not None) and ("gdp_pc" not in ex)
        is_root = bool(ex) and (ex.get("id") == "World") and not is_leaf

        if is_root:
            shapes.append(dict(
                type="circle", xref="x", yref="y",
                x0=x - r, x1=x + r, y0=y - r, y1=y + r,
                line=dict(color=EDGE, width=1.4), fillcolor="rgba(0,0,0,0)"
            ))
            continue

        if is_continent:
            shapes.append(dict(
                type="circle", xref="x", yref="y",
                x0=x - r, x1=x + r, y0=y - r, y1=y + r,
                line=dict(color=EDGE, width=2.0),
                fillcolor=f"rgba(29,42,59,{CONT_ALPHA})"
            ))
            name = ex.get("id", "")
            if name:
                # umbral bajo + fallback para continentes pequeños (Oceania)
                if r > 0.10:
                    annotations.append(dict(
                        x=x, y=y, text=name, showarrow=False,
                        font=dict(size=12, color="#94a3b8"), xanchor="center", yanchor="middle"
                    ))
                else:
                    annotations.append(dict(
                        x=x, y=y - r*0.15, text=name, showarrow=False,
                        font=dict(size=10, color="#94a3b8"), xanchor="center", yanchor="top"
                    ))
            continue

        if is_leaf:
            g = float(ex["gdp_pc"])
            u = max(min(_to_unit_log(g), 1.0), 0.0)  # 0..1
            fill = sample_colorscale(viridis, [u])[0]  # hex color
            shapes.append(dict(
                type="circle", xref="x", yref="y",
                x0=x - r, x1=x + r, y0=y - r, y1=y + r,
                line=dict(color="#0b2545", width=0.6),
                fillcolor=fill
            ))

    # --- traza "dummy" para mostrar la colorbar (hack habitual en Plotly)
    colorbar_trace = go.Scatter(
        x=[None], y=[None],
        mode="markers",
        marker=dict(
            size=0.1,  # invisible
            color=[vmin, vmax],
            colorscale=viridis,
            showscale=True,
            cmin=vmin, cmax=vmax,
            colorbar=dict(
                title=dict(text="GDP per Capita<br>(log scale)", font=dict(color=TEXT_LIGHT)),
                ticks="outside",
                tickformat=".2e",
                tickfont=dict(color=TEXT_LIGHT),
            ),
        ),
        hoverinfo="skip",
        showlegend=False,
    )

    fig = go.Figure(data=[colorbar_trace])
    fig.update_layout(
        title=dict(
            text="Circular Treemap (Circle Packing)<br><sup>Size = Population · Color = GDP per Capita</sup>",
            x=0.5, xanchor="center",
            font=dict(color=TEXT_LIGHT)
        ),
        shapes=shapes,
        annotations=annotations,
        xaxis=dict(visible=False, range=[-1.05, 1.05], scaleratio=1),
        yaxis=dict(visible=False, range=[-1.05, 1.05], scaleanchor="x"),
        margin=dict(t=80, l=10, r=80, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig
# =========================
# COMPARATIVE ANALYSIS CHARTS
# =========================
def build_bubble_scatter(df: pd.DataFrame) -> go.Figure:
    fig = px.scatter(
        df, x="Population", y="GDP_per_capita", size="GDP", color="Continent",
        hover_name="Country_Name", size_max=40, log_x=True,
        labels={"Population": "Population (log scale)", "GDP_per_capita": "GDP per Capita",
                "GDP": "GDP", "Continent": "Continent"},
        title="GDP per Capita vs Population (Bubble size ~ GDP)", template="plotly_white",
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT_LIGHT),
        legend_title_text="Continent", margin=dict(t=60, l=10, r=10, b=10),
    )
    return fig
def build_violin(df: pd.DataFrame) -> go.Figure:
    fig = px.violin(
        df, x="Continent", y="GDP_per_capita", box=True, points=False,
        hover_data=["Country_Name", "Country_Code"],
        labels={"GDP_per_capita": "GDP per Capita", "Continent": "Continent"},
        title="Distribution of GDP per Capita by Continent", template="plotly_white",
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT_LIGHT),
        margin=dict(t=60, l=10, r=10, b=10),
    )
    return fig
# =========================
# MAIN APP
# =========================
st.title("World Population — Analytics ")
st.caption("Explora composición y métricas por continente y país.")
csv_path = "countries_per_region.csv"
try:
    df_raw = load_csv(csv_path)
except Exception as e:
    st.error(f"No se pudo leer `{csv_path}`.\n\n{e}")
    st.stop()
missing = validate_columns(df_raw, REQUIRED_COLS)
if missing:
    st.error(f"Faltan columnas en el CSV: {missing}")
    st.stop()
df = clean_and_enrich(df_raw)
# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Treemap", "Dendrogram", "Sunburst Chart", "Circular Treemap", "Comparative Analysis"]
)
with tab1:
    st.subheader("Treemap")
    fig_treemap = build_treemap(df, height=900)
    st.plotly_chart(fig_treemap, use_container_width=True, config={"displayModeBar": False})
with tab2:
    st.subheader("Dendrograma (Clustering jerárquico)")
    df_d = df.copy()
    if "Continent" in df_d.columns:
        continents = sorted(df_d["Continent"].dropna().unique().tolist())
        cont_sel = st.multiselect("Filtrar por continente", continents, default=continents)
        df_d = df_d[df_d["Continent"].isin(cont_sel)]
    num_cols = df_d.select_dtypes(include=["number"]).columns.tolist()
    pref = [c for c in ["Population", "GDP"] if c in num_cols]
    feat_sel = st.multiselect("Variables para clustering", num_cols, default=(pref if pref else num_cols))
    method = st.selectbox("Método de enlace", ["ward", "average", "complete", "single"], index=0)
    metric = st.selectbox("Métrica de distancia", ["euclidean", "cityblock", "cosine"], index=0, help="Para 'ward' se usa siempre euclidean.")
    if len(feat_sel) < 1:
        st.info("Selecciona al menos una variable numérica para generar el dendrograma.")
    else:
        X = df_d[feat_sel].to_numpy(dtype=float)
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        try:
            if method == "ward":
                Z = linkage(Xs, method="ward")
            else:
                Z = linkage(Xs, method=method, metric=metric)
        except Exception as e:
            st.error(f"Error calculando el dendrograma: {e}")
            Z = None
        if Z is not None:
            orientacion = st.radio(
                "Orientación del dendrograma",
                ["Vertical (arriba)", "Horizontal (derecha)"],
                index=0,
                help="Cambia la disposición de la gráfica."
            )
            labels = (
                df_d["Country_Name"].astype(str).tolist()
                if "Country_Name" in df_d.columns
                else df_d.index.astype(str).tolist()
            )
            is_vertical = orientacion.startswith("Vertical")
            orient = "top" if is_vertical else "right"
            if is_vertical:
                fig_w, fig_h = (12, 6)
                leaf_rot = 90
                pad_kwargs = dict(bottom=0.30)
            else:
                fig_w, fig_h = (10, max(6, 0.22 * len(labels)))
                leaf_rot = 0
                pad_kwargs = dict(left=0.30)
            fig = plt.figure(figsize=(fig_w, fig_h))
            dendrogram(
                Z,
                labels=labels,
                orientation=orient,
                leaf_rotation=leaf_rot,
                leaf_font_size=8,
                distance_sort="ascending",
                show_leaf_counts=False
            )
            plt.subplots_adjust(**pad_kwargs)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            with st.expander("Detalles del clustering"):
                st.write(f"Observaciones: {Xs.shape[0]} | Variables: {Xs.shape[1]}")
                st.write(f"Método: **{method}** | Métrica: **{'euclidean' if method=='ward' else metric}**")
with tab3:
    st.subheader("Sunburst Chart")
    fig_sun = build_sunburst(df)
    st.plotly_chart(fig_sun, use_container_width=True, config={"displayModeBar": False})
with tab4:
    st.subheader("Circular Treemap (Circle Packing)")
    try:
        fig_cp = build_circle_packing_plotly(df)
        st.plotly_chart(fig_cp, use_container_width=True, config={"displayModeBar": False})
    except Exception as e:
        st.error(f"No se pudo renderizar el Circle Packing: {e}")
with tab5:
    st.subheader("Comparative Analysis (New Visualizations)")
    fig_bubble = build_bubble_scatter(df)
    st.plotly_chart(fig_bubble, use_container_width=True, config={"displayModeBar": False})
    st.markdown("**Visualization 1 — Bubble Scatter**")
    st.markdown(
        "- **a) Best suited for:** Explorar relaciones y clusters entre tamaño poblacional y riqueza por persona; el tamaño de burbuja añade la magnitud de GDP.\n"
        "- **b) Key insight:** Países con poblaciones muy grandes suelen concentrarse en niveles bajos/medios de GDP per capita, mientras que países con menor población pueden mostrar niveles muy altos; también se ven disparidades y outliers por continente.\n"
        "- **c) Limitation:** El solapamiento de burbujas puede ocultar países pequeños y dificulta lecturas precisas basadas en áreas."
    )
    st.markdown("---")
    fig_violin = build_violin(df)
    st.plotly_chart(fig_violin, use_container_width=True, config={"displayModeBar": False})
    st.markdown("**Visualization 2 — Violin Plot**")
    st.markdown(
        "- **a) Best suited for:** Comparar la forma de la distribución, la dispersión y la tendencia central de GDP per capita entre continentes.\n"
        "- **b) Key insight:** Violines más anchos indican mayor dispersión (economías heterogéneas); violines más estrechos sugieren uniformidad en niveles de riqueza por persona.\n"
        "- **c) Limitation:** Se pierde el detalle a nivel país; con tamaños de muestra pequeños por continente la forma puede inducir a error."
    )