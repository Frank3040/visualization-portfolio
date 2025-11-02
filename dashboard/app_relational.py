# app.py
from pathlib import Path
import math
import numpy as np
import pandas as pd
import streamlit as st
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

# =========================
# Page configuration
# =========================
st.set_page_config(
    page_title="Network Analysis (Facebook)",
    page_icon="🕸️",
    layout="wide",
)

# =========================
# CSV loading utilities
# =========================
@st.cache_data(show_spinner=False)
def load_csv_from_path(path_str: str) -> pd.DataFrame:
    path = Path(path_str).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path)

@st.cache_data(show_spinner=False)
def load_csv_from_upload(uploaded_file) -> pd.DataFrame:
    return pd.read_csv(uploaded_file)

def csv_loader_ui(default_name: str = "") -> pd.DataFrame | None:
    with st.expander("📁 Load data (CSV)", expanded=True):
        cols = st.columns([2, 1.2, 1.2])
        with cols[0]:
            file_name = st.text_input(
                "CSV name (same folder as the project)",
                value=default_name,
                placeholder="e.g., top50_neighbors_betweenness.csv",
            )
        with cols[1]:
            load_local = st.button("Load local CSV")
        with cols[2]:
            uploaded = st.file_uploader("…or upload CSV", type=["csv"], label_visibility="collapsed")

        df = None
        if load_local and file_name.strip():
            try:
                df = load_csv_from_path(file_name.strip())
                st.success(f"✅ Loaded local file: {file_name}")
            except Exception as e:
                st.error(f"❌ Load error: {e}")

        if uploaded is not None:
            try:
                df = load_csv_from_upload(uploaded)
                st.success("✅ Loaded from uploaded file")
            except Exception as e:
                st.error(f"❌ Load error: {e}")

        if df is not None:
            st.caption("Preview (first rows)")
            st.dataframe(df.head(20), width="stretch")

        return df

def show_basic_bar_if_columns(df: pd.DataFrame, x_col: str, y_col: str, title: str):
    if df is None:
        return
    if x_col in df.columns and y_col in df.columns:
        st.write(f"**Quick chart:** {title}")
        st.bar_chart(df.set_index(x_col)[y_col], width="stretch")
    else:
        st.info(f"For the quick chart, make sure you have columns **{x_col}** and **{y_col}**.")

# =========================
# Load graph from .txt (edge list)
# =========================
@st.cache_resource(show_spinner=False)
def load_graph_from_edgelist(path_str: str) -> nx.Graph:
    """
    Expected line format: "u v" (IDs separated by a space).
    """
    path = Path(path_str).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    G = nx.read_edgelist(path, create_using=nx.Graph(), nodetype=int)
    return G

def graph_loader_ui(default_name: str = "facebook_combined.txt", key_prefix: str = "home") -> nx.Graph | None:
    with st.expander("🧩 Load graph (.txt edge list)", expanded=True):
        c1, c2, c3 = st.columns([2, 1.2, 1.2])
        with c1:
            txt_name = st.text_input(
                ".txt name (same folder as app.py)",
                value=default_name,
                placeholder="facebook_combined.txt",
                key=f"{key_prefix}_txt_name",
            )
        with c2:
            load_local = st.button("Load local .txt", key=f"{key_prefix}_btn_local")
        with c3:
            uploaded = st.file_uploader("…or upload .txt", type=["txt"], label_visibility="collapsed", key=f"{key_prefix}_uploader")

        G = None
        if load_local and txt_name.strip():
            try:
                G = load_graph_from_edgelist(txt_name.strip())
                st.session_state["G"] = G
                st.success(
                    f"✅ Graph loaded: {txt_name} "
                    f"({G.number_of_nodes():,} nodes / {G.number_of_edges():,} edges)"
                )
            except Exception as e:
                st.error(f"❌ Load error: {e}")

        if uploaded is not None:
            try:
                tmp_path = Path(
                    st.session_state.get(f"_{key_prefix}_tmp_edgelist_path", f"{key_prefix}_uploaded_edgelist.txt")
                ).resolve()
                tmp_path.write_bytes(uploaded.getvalue())
                st.session_state[f"_{key_prefix}_tmp_edgelist_path"] = str(tmp_path)
                G = load_graph_from_edgelist(str(tmp_path))
                st.session_state["G"] = G
                st.success(
                    f"✅ Graph loaded from uploaded file "
                    f"({G.number_of_nodes():,} nodes / {G.number_of_edges():,} edges)"
                )
            except Exception as e:
                st.error(f"❌ Load error: {e}")

        return st.session_state.get("G", G)

# =========================
# Sidebar / Navigation
# =========================
st.sidebar.title("🧭 Navigation")
section = st.sidebar.radio("Go to:", ["Home", "Degree", "Betweenness", "Closeness"], index=0)
st.sidebar.markdown("---")
st.sidebar.caption("Authors: Los cuyos team")

# =========================
# Fixed summary values (if not computed)
# =========================
DEFAULT_SUMMARY = {
    "Nodes": 4039,
    "Edges": 88234,
    "Average Degree": 43.69,
    "Density": 0.010820,
    "Connected Components": 1,
    "Diameter": 8,
    "Avg Shortest Path Length": 3.69,
}

# =========================
# Common Top-N helper
# =========================
def get_top_nodes(centrality_dict, metric_name, top_n=10):
    top_items = sorted(centrality_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
    df = pd.DataFrame(top_items, columns=['Node ID', metric_name])
    df.index = range(1, len(df) + 1)
    return df

# =========================
# DEGREE (functions)
# =========================
def plot_top10_degree(centrality_dict, metric_name='Degree Centrality', savepath=None):
    df = get_top_nodes(centrality_dict, metric_name, top_n=10).reset_index(drop=True)
    df_for_plot = df.copy()
    df_for_plot['Node ID'] = df_for_plot['Node ID'].astype(str)
    df_for_plot = df_for_plot.iloc[::-1]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(df_for_plot['Node ID'], df_for_plot[metric_name], color='steelblue')
    ax.set_xlabel(metric_name)
    ax.set_ylabel("Node ID")
    ax.set_title("Top 10 Nodes by Degree Centrality")
    fig.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")

    return df, fig

def top_k_neighbors_of_top_degree(G: nx.Graph, degree_centrality: dict, k=50, output_file="top50_neighbors_degree.csv"):
    top_node = max(degree_centrality, key=degree_centrality.get)
    if isinstance(G, nx.DiGraph):
        neighbors = set(G.predecessors(top_node)) | set(G.successors(top_node))
    else:
        neighbors = set(G.neighbors(top_node))
    data = [{"Center Node": top_node, "Neighbor": n, "Degree Centrality": degree_centrality.get(n, 0.0)} for n in neighbors]
    df = pd.DataFrame(data).sort_values(by="Degree Centrality", ascending=False).head(k)
    df.index = range(1, len(df) + 1)
    subG = G.subgraph([top_node] + df["Neighbor"].tolist()).copy()
    try:
        df.to_csv(output_file, index=False, encoding="utf-8")
    except Exception:
        pass
    return df, subG

def plot_star_by_degree(
    df_neighbors: pd.DataFrame,
    value_col: str = "Degree Centrality",
    center_col: str = "Center Node",
    neighbor_col: str = "Neighbor",
    title: str = "Node with Highest Degree and Its 50 Strongest Direct Connections",
    radius: float = 6.0,
    start_angle_deg: float = 90.0,
    cmap_name: str = "plasma",
    arrows: bool = True,
    neighbor_size_min: int = 450,
    neighbor_size_max: int = 1200,
    center_size: int = 2600
):
    if df_neighbors is None or df_neighbors.empty:
        return None
    center = df_neighbors[center_col].iloc[0]
    neighbors = df_neighbors[neighbor_col].tolist()
    vals = df_neighbors[value_col].astype(float).values

    vmin, vmax = float(np.nanmin(vals)), float(np.nanmax(vals))
    if vmin == vmax:
        vmin = 0.0
    norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
    cmap = matplotlib.colormaps.get_cmap(cmap_name)

    pos = {center: (0.0, 0.0)}
    n = max(1, len(neighbors))
    order = np.argsort(-vals)
    angles = np.linspace(0, 2 * math.pi, n, endpoint=False) + math.radians(start_angle_deg)
    for rank, idx in enumerate(order):
        node = neighbors[idx]
        ang = angles[rank]
        pos[node] = (radius * math.cos(ang), radius * math.sin(ang))

    fig, ax = plt.subplots(figsize=(12, 8))
    for nb in neighbors:
        x0, y0 = pos[center]
        x1, y1 = pos[nb]
        ax.plot([x0, x1], [y0, y1], color="#2458FF", linewidth=1.3, alpha=0.9)
        if arrows:
            dx, dy = (x1 - x0), (y1 - y0)
            ax.arrow(x0, y0, dx * 0.97, dy * 0.97,
                     length_includes_head=True, head_width=0.18, head_length=0.35,
                     fc="#2458FF", ec="#2458FF", alpha=0.9)

    nb_colors = [cmap(norm(v)) for v in vals]
    nb_sizes = [neighbor_size_min + (neighbor_size_max - neighbor_size_min) * norm(v) for v in vals]
    for i, nb in enumerate(neighbors):
        ax.scatter(*pos[nb], s=nb_sizes[i], c=[nb_colors[i]], edgecolors="#224", linewidths=0.9, zorder=3)
        ax.text(pos[nb][0], pos[nb][1], str(nb), ha="center", va="center",
                fontsize=9, fontweight="bold", color="black", zorder=4)

    ax.scatter(*pos[center], s=center_size, c="#0E2A7B", edgecolors="black", linewidths=2.0, zorder=5)
    ax.text(pos[center][0], pos[center][1], str(center), ha="center", va="center",
            fontsize=12, fontweight="bold", color="white", zorder=6)

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label(value_col, rotation=90)

    ax.set_title(title, fontsize=14)
    ax.set_axis_off()
    ax.set_aspect('equal', adjustable='box')
    m = radius + 1.5
    ax.set_xlim(-m, m); ax.set_ylim(-m, m)
    fig.tight_layout()
    return fig

def ensure_degree_centrality(G: nx.Graph) -> dict:
    if "G" not in st.session_state or G is None:
        return {}
    cache_key = f"degcent_{G.number_of_nodes()}_{G.number_of_edges()}"
    if cache_key in st.session_state:
        return st.session_state[cache_key]
    degc = nx.degree_centrality(G)
    st.session_state[cache_key] = degc
    return degc

# =========================
# BETWEENNESS (functions)
# =========================
def plot_top10_bars(centrality_dict, metric_name, title=None, savepath=None):
    df = get_top_nodes(centrality_dict, metric_name, top_n=10).reset_index(drop=True)
    df_for_plot = df.copy()
    df_for_plot['Node ID'] = df_for_plot['Node ID'].astype(str)
    df_for_plot = df_for_plot.iloc[::-1]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(df_for_plot['Node ID'], df_for_plot[metric_name], color='skyblue', edgecolor='black')
    ax.set_xlabel(metric_name)
    ax.set_ylabel("Node ID")
    ax.set_title(title or f"Top 10 by {metric_name}")
    fig.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")

    return df, fig

def top_k_neighbors_of_top_betweenness(G, betweenness_centrality, k=50, output_file="top50_neighbors_betweenness.csv"):
    top_node = max(betweenness_centrality, key=betweenness_centrality.get)
    if isinstance(G, nx.DiGraph):
        neighbors = set(G.predecessors(top_node)) | set(G.successors(top_node))
    else:
        neighbors = set(G.neighbors(top_node))
    data = [{"Center Node": top_node, "Neighbor": n, "Betweenness": betweenness_centrality.get(n, 0.0)} for n in neighbors]
    df = pd.DataFrame(data).sort_values(by="Betweenness", ascending=False).head(k)
    df.index = range(1, len(df) + 1)
    subG = G.subgraph([top_node] + df["Neighbor"].tolist()).copy()
    try:
        df.to_csv(output_file, index=False, encoding="utf-8")
    except Exception:
        pass
    return df, subG

def plot_star_by_betweenness(
    df_neighbors: pd.DataFrame,
    value_col: str = "Betweenness",
    center_col: str = "Center Node",
    neighbor_col: str = "Neighbor",
    title: str = "Node with Highest Betweenness and Its 50 Strongest Direct Connections",
    radius: float = 6.0,
    start_angle_deg: float = 90.0,
    cmap_name: str = "plasma",
    arrows: bool = False,
    neighbor_size_min: int = 420,
    neighbor_size_max: int = 1200,
    center_size: int = 3000
):
    if df_neighbors is None or df_neighbors.empty:
        return None

    center = df_neighbors[center_col].iloc[0]
    neighbors = df_neighbors[neighbor_col].tolist()
    vals = df_neighbors[value_col].astype(float).values
    vmin, vmax = float(np.nanmin(vals)), float(np.nanmax(vals))
    if vmin == vmax:
        vmin = 0.0
    norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
    cmap = matplotlib.colormaps.get_cmap(cmap_name)

    pos = {center: (0.0, 0.0)}
    n = max(1, len(neighbors))
    order = np.argsort(-vals)
    angles = np.linspace(0, 2 * math.pi, n, endpoint=False) + math.radians(start_angle_deg)
    for rank, idx in enumerate(order):
        node = neighbors[idx]
        ang = angles[rank]
        pos[node] = (radius * math.cos(ang), radius * math.sin(ang))

    fig, ax = plt.subplots(figsize=(12, 8))
    for nb in neighbors:
        x0, y0 = pos[center]
        x1, y1 = pos[nb]
        ax.plot([x0, x1], [y0, y1], color="#2458FF", linewidth=1.3, alpha=0.9)
        if arrows:
            dx, dy = (x1 - x0), (y1 - y0)
            ax.arrow(x0, y0, dx * 0.97, dy * 0.97,
                     length_includes_head=True, head_width=0.18, head_length=0.35,
                     fc="#2458FF", ec="#2458FF", alpha=0.9)

    nb_colors = [cmap(norm(v)) for v in vals]
    nb_sizes = [neighbor_size_min + (neighbor_size_max - neighbor_size_min) * norm(v) for v in vals]
    for i, nb in enumerate(neighbors):
        ax.scatter(*pos[nb], s=nb_sizes[i], c=[nb_colors[i]], edgecolors="#224", linewidths=0.9, zorder=3)
        ax.text(pos[nb][0], pos[nb][1], str(nb), ha="center", va="center",
                fontsize=9, fontweight="bold", color="black", zorder=4)

    ax.scatter(*pos[center], s=center_size, c="#0E2A7B", edgecolors="black", linewidths=2.0, zorder=5)
    ax.text(pos[center][0], pos[center][1], str(center), ha="center", va="center",
            fontsize=12, fontweight="bold", color="white", zorder=6)

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label(value_col, rotation=90)

    ax.set_title(title, fontsize=14)
    ax.set_axis_off()
    ax.set_aspect('equal', adjustable='box')
    m = radius + 1.5
    ax.set_xlim(-m, m); ax.set_ylim(-m, m)
    fig.tight_layout()
    return fig

def ensure_betweenness_centrality(G: nx.Graph, approx_k: int | None = 256) -> dict:
    """
    Computes normalized betweenness centrality.
    - approx_k: if int -> sample-based approximation with k nodes (fast for large graphs).
                if None -> exact computation (can be slow).
    Cached by (nodes, edges, approx_k).
    """
    if "G" not in st.session_state or G is None:
        return {}
    cache_key = f"betcent_{G.number_of_nodes()}_{G.number_of_edges()}_{approx_k}"
    if cache_key in st.session_state:
        return st.session_state[cache_key]
    if approx_k is None:
        bet = nx.betweenness_centrality(G, normalized=True)
    else:
        bet = nx.betweenness_centrality(G, k=approx_k, normalized=True, seed=42)
    st.session_state[cache_key] = bet
    return bet

# =========================
# CLOSENESS (functions)
# =========================
def plot_top10_bars_closeness(centrality_dict, metric_name, title=None, savepath=None):
    df = get_top_nodes(centrality_dict, metric_name, top_n=10).reset_index(drop=True)
    df_for_plot = df.copy()
    df_for_plot['Node ID'] = df_for_plot['Node ID'].astype(str)
    df_for_plot = df_for_plot.iloc[::-1]  # highest at top

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(df_for_plot['Node ID'], df_for_plot[metric_name])
    ax.set_xlabel(metric_name)
    ax.set_ylabel("Node ID")
    ax.set_title(title or f"Top 10 by {metric_name}")
    fig.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")

    return df, fig

def top_k_neighbors_of_top_closeness(
    G,
    k=50,
    output_file="top50_neighbors_closeness.csv",
    closeness_centrality=None
):
    if closeness_centrality is None:
        closeness_centrality = nx.closeness_centrality(G)
    top_node = max(closeness_centrality, key=closeness_centrality.get)
    if isinstance(G, nx.DiGraph):
        neighbors = set(G.predecessors(top_node)) | set(G.successors(top_node))
    else:
        neighbors = set(G.neighbors(top_node))
    df = pd.DataFrame(
        [{"Center Node": top_node,
          "Neighbor": n,
          "Closeness": float(closeness_centrality.get(n, 0.0))}
         for n in neighbors]
    ).sort_values("Closeness", ascending=False).head(k)
    subG = G.subgraph([top_node] + df["Neighbor"].tolist()).copy()
    df_to_save = df[["Center Node", "Neighbor", "Closeness"]]
    try:
        df_to_save.to_csv(output_file, index=False, encoding="utf-8")
    except Exception:
        pass
    return df_to_save, subG

def plot_star_by_closeness(
    df_neighbors: pd.DataFrame,
    value_col: str = "Closeness",
    center_col: str = "Center Node",
    neighbor_col: str = "Neighbor",
    title: str = "Node with Highest Closeness and Its 50 Strongest Direct Connections",
    radius: float = 6.0,
    start_angle_deg: float = 90.0,
    cmap_name: str = "plasma",
    arrows: bool = False,
    neighbor_size_min: int = 420,
    neighbor_size_max: int = 1200,
    center_size: int = 3000
):
    if df_neighbors is None or df_neighbors.empty:
        return None
    center = df_neighbors[center_col].iloc[0]
    neighbors = df_neighbors[neighbor_col].tolist()
    vals = df_neighbors[value_col].astype(float).values
    vmin, vmax = float(np.nanmin(vals)), float(np.nanmax(vals))
    if vmin == vmax:
        vmin = 0.0
    norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
    cmap = matplotlib.colormaps.get_cmap(cmap_name)
    pos = {center: (0.0, 0.0)}
    n = max(1, len(neighbors))
    order = np.argsort(-vals)
    angles = np.linspace(0, 2 * math.pi, n, endpoint=False) + math.radians(start_angle_deg)
    for rank, idx in enumerate(order):
        node = neighbors[idx]
        ang = angles[rank]
        pos[node] = (radius * math.cos(ang), radius * math.sin(ang))
    fig, ax = plt.subplots(figsize=(12, 8))
    for nb in neighbors:
        x0, y0 = pos[center]
        x1, y1 = pos[nb]
        ax.plot([x0, x1], [y0, y1], color="#2458FF", linewidth=1.3, alpha=0.9)
        if arrows:
            dx, dy = (x1 - x0), (y1 - y0)
            ax.arrow(
                x0, y0, dx * 0.97, dy * 0.97,
                length_includes_head=True,
                head_width=0.18, head_length=0.35,
                fc="#2458FF", ec="#2458FF", alpha=0.9
            )
    nb_colors = [cmap(norm(v)) for v in vals]
    nb_sizes = [neighbor_size_min + (neighbor_size_max - neighbor_size_min) * norm(v) for v in vals]
    for i, nb in enumerate(neighbors):
        ax.scatter(*pos[nb], s=nb_sizes[i], c=[nb_colors[i]], edgecolors="#224", linewidths=0.9, zorder=3)
        ax.text(pos[nb][0], pos[nb][1], str(nb), ha="center", va="center",
                fontsize=9, fontweight="bold", color="black", zorder=4)
    ax.scatter(*pos[center], s=center_size, c="#0E2A7B", edgecolors="black", linewidths=2.0, zorder=5)
    ax.text(pos[center][0], pos[center][1], str(center), ha="center", va="center",
            fontsize=12, fontweight="bold", color="white", zorder=6)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label(value_col, rotation=90)
    ax.set_title(title, fontsize=14)
    ax.set_axis_off()
    ax.set_aspect('equal', adjustable='box')
    m = radius + 1.5
    ax.set_xlim(-m, m)
    ax.set_ylim(-m, m)
    fig.tight_layout()
    return fig

def ensure_closeness_centrality(G: nx.Graph) -> dict:
    """Exact closeness (usually lighter than betweenness). Cached by (nodes, edges)."""
    if "G" not in st.session_state or G is None:
        return {}
    cache_key = f"closcent_{G.number_of_nodes()}_{G.number_of_edges()}"
    if cache_key in st.session_state:
        return st.session_state[cache_key]
    clos = nx.closeness_centrality(G)
    st.session_state[cache_key] = clos
    return clos

# =========================
# HOME
# =========================
if section == "Home":
    st.title("🕸️ Social Network Analysis (Facebook)")
    st.markdown(
        """
        This network represents Facebook friendship ties within a local community.
        Each **node** is a person and each **edge** is a confirmed friendship.
        We observe a **giant component** (the whole network is connected) and short distances:
        on average, any pair of users is separated by **~3.7 steps**, and the **diameter** is **8**.
        The **density** ≈ **1.08%** suggests subcommunities, with **hubs** and **bridge nodes** speeding up diffusion.
        """
    )

    # Load .txt graph (optional)
    G = graph_loader_ui(default_name="facebook_combined.txt", key_prefix="home")

    # KPIs
    if G is not None and G.number_of_nodes() > 0:
        nodes = G.number_of_nodes()
        edges = G.number_of_edges()
        avg_degree = (2 * edges / nodes) if nodes > 0 else 0.0
        density = DEFAULT_SUMMARY["Density"]
        comps = list(nx.connected_components(G)) if not nx.is_directed(G) else list(nx.weakly_connected_components(G))
        connected_components = len(comps)

        k1, k2, k3, k4 = st.columns(4)
        with k1: st.metric("Nodes", f"{nodes:,}")
        with k2: st.metric("Edges", f"{edges:,}")
        with k3: st.metric("Average Degree", f"{avg_degree:.2f}")
        with k4: st.metric("Density", f"{density:.6f}")

        k5, k6, k7 = st.columns(3)
        with k5: st.metric("Connected Components", f"{connected_components}")
        with k6: st.metric("Diameter", f"{DEFAULT_SUMMARY['Diameter']}")
        with k7: st.metric("Avg Shortest Path Length", f"{DEFAULT_SUMMARY['Avg Shortest Path Length']:.2f}")
    else:
        k1, k2, k3, k4 = st.columns(4)
        with k1: st.metric("Nodes", f"{DEFAULT_SUMMARY['Nodes']:,}")
        with k2: st.metric("Edges", f"{DEFAULT_SUMMARY['Edges']:,}")
        with k3: st.metric("Average Degree", f"{DEFAULT_SUMMARY['Average Degree']:.2f}")
        with k4: st.metric("Density", f"{DEFAULT_SUMMARY['Density']:.6f}")

        k5, k6, k7 = st.columns(3)
        with k5: st.metric("Connected Components", f"{DEFAULT_SUMMARY['Connected Components']}")
        with k6: st.metric("Diameter", f"{DEFAULT_SUMMARY['Diameter']}")
        with k7: st.metric("Avg Shortest Path Length", f"{DEFAULT_SUMMARY['Avg Shortest Path Length']:.2f}")

    st.markdown("### 📋 Network Topology Summary")
    summary_df = pd.DataFrame(
        {
            "Metric": [
                "Nodes","Edges","Average Degree","Density","Connected","Diameter","Avg Shortest Path Length",
            ],
            "Value": [
                f"{st.session_state.get('G').number_of_nodes():,}" if st.session_state.get('G') else f"{DEFAULT_SUMMARY['Nodes']:,}",
                f"{st.session_state.get('G').number_of_edges():,}" if st.session_state.get('G') else f"{DEFAULT_SUMMARY['Edges']:,}",
                f"{(2*st.session_state['G'].number_of_edges()/st.session_state['G'].number_of_nodes()):.2f}" if st.session_state.get('G') else f"{DEFAULT_SUMMARY['Average Degree']:.2f}",
                f"{DEFAULT_SUMMARY['Density']:.6f}",
                f"{DEFAULT_SUMMARY['Connected Components']}",
                f"{DEFAULT_SUMMARY['Diameter']}",
                f"{DEFAULT_SUMMARY['Avg Shortest Path Length']:.2f}",
            ],
        }
    )
    st.dataframe(summary_df, width="stretch")
    st.info("💡 Load your `facebook_combined.txt` above to see the graph’s actual figures.")

# =========================
# DEGREE
# =========================
elif section == "Degree":
    st.title("📈 Degree / Degree Centrality")

    # Reuse graph or allow loading
    G = st.session_state.get("G")
    if G is None:
        st.warning("No graph loaded from Home. You can load your `.txt` below.")
        G = graph_loader_ui(default_name="facebook_combined.txt", key_prefix="degree")
    if G is None:
        st.stop()

    with st.expander("⚙️ Visualization parameters (Degree)", expanded=True):
        k_neighbors = st.slider("Neighbors to show for the highest-degree node", 5, 100, 50, 5)
        cmap_name = st.selectbox("Colormap", ["plasma", "viridis", "magma", "cividis", "inferno"], 0)
        radius = st.slider("Star layout radius", 3.0, 12.0, 6.0, 0.5)
        arrows = st.checkbox("Show arrows", value=False)
        neighbor_size_min = st.slider("Min neighbor size", 200, 1000, 420, 20)
        neighbor_size_max = st.slider("Max neighbor size", 800, 3000, 1200, 50)
        center_size = st.slider("Center node size", 1000, 6000, 3000, 100)

    degree_centrality = ensure_degree_centrality(G)
    if not degree_centrality:
        st.error("Could not compute degree centrality.")
        st.stop()

    st.subheader("Top 10 Nodes by Degree Centrality")
    try:
        df_degree_top10, fig_top10 = plot_top10_degree(degree_centrality, metric_name='Degree Centrality')
        st.pyplot(fig_top10)
        st.dataframe(df_degree_top10, width="stretch")
    except Exception as e:
        st.error(f"Error while plotting Top 10: {e}")

    st.subheader(f"Highest-Degree node and its {k_neighbors} strongest direct connections")
    try:
        df_neighbors, subG = top_k_neighbors_of_top_degree(G, degree_centrality, k=k_neighbors, output_file="top_neighbors_degree.csv")
        st.caption("Neighbors table (sorted by neighbor Degree Centrality):")
        st.dataframe(df_neighbors, width="stretch")

        fig_star = plot_star_by_degree(
            df_neighbors,
            value_col="Degree Centrality",
            title=f"Node with Highest Degree and Its {k_neighbors} Strongest Direct Connections",
            radius=radius, start_angle_deg=90, cmap_name=cmap_name, arrows=arrows,
            neighbor_size_min=neighbor_size_min, neighbor_size_max=neighbor_size_max, center_size=center_size
        )
        if fig_star is not None:
            st.pyplot(fig_star)
        else:
            st.info("No data to draw the star.")
    except Exception as e:
        st.error(f"Error while generating the star: {e}")

    st.markdown("---")
    st.write("Optional: load a CSV with precomputed results.")
    df = csv_loader_ui(default_name="degree_centrality_results.csv")
    show_basic_bar_if_columns(df, x_col="Node ID", y_col="Degree Centrality", title="Top by Degree Centrality (CSV)")

# =========================
# BETWEENNESS
# =========================
elif section == "Betweenness":
    st.title("🪢 Betweenness Centrality")

    # Reuse graph or allow loading
    G = st.session_state.get("G")
    if G is None:
        st.warning("No graph loaded from Home. You can load your `.txt` below.")
        G = graph_loader_ui(default_name="facebook_combined.txt", key_prefix="betweenness")
    if G is None:
        st.stop()

    with st.expander("⚙️ Computation & visualization parameters (Betweenness)", expanded=True):
        use_approx = st.checkbox("Use approximation (recommended for large graphs)", value=True,
                                 help="Uses node sampling (k) to speed up the computation.")
        approx_k = st.slider("Sampling k (approximate betweenness)", 16, 1024, 256, 16, disabled=not use_approx)
        k_neighbors = st.slider("Neighbors to show for the highest-betweenness node", 5, 100, 50, 5)
        cmap_name = st.selectbox("Colormap", ["plasma", "viridis", "magma", "cividis", "inferno"], 0)
        radius = st.slider("Star layout radius", 3.0, 12.0, 6.0, 0.5)
        arrows = st.checkbox("Show arrows", value=False)
        neighbor_size_min = st.slider("Min neighbor size", 200, 1000, 420, 20)
        neighbor_size_max = st.slider("Max neighbor size", 800, 3000, 1200, 50)
        center_size = st.slider("Center node size", 1000, 6000, 3000, 100)

    betweenness_centrality = ensure_betweenness_centrality(G, approx_k=approx_k if use_approx else None)
    if not betweenness_centrality:
        st.error("Could not compute betweenness centrality.")
        st.stop()

    st.subheader("Top 10 Nodes by Betweenness Centrality")
    try:
        df_bet_top10, fig_bars = plot_top10_bars(
            betweenness_centrality,
            'Betweenness Centrality',
            title="Top 10 Nodes by Betweenness Centrality",
            savepath=None
        )
        st.pyplot(fig_bars)
        st.dataframe(df_bet_top10, width="stretch")
    except Exception as e:
        st.error(f"Error while plotting Top 10: {e}")

    st.subheader(f"Highest-Betweenness node and its {k_neighbors} strongest direct connections")
    try:
        df_neighbors_all, _ = top_k_neighbors_of_top_betweenness(G, betweenness_centrality, k=10_000, output_file="top50_neighbors_betweenness_full.csv")
        df_neighbors = df_neighbors_all.head(k_neighbors).copy()

        st.caption("Neighbors table (sorted by neighbor Betweenness):")
        st.dataframe(df_neighbors, width="stretch")

        fig_star = plot_star_by_betweenness(
            df_neighbors,
            value_col="Betweenness",
            title=f"Node with Highest Betweenness and Its {k_neighbors} Strongest Direct Connections",
            radius=radius, start_angle_deg=90, cmap_name=cmap_name, arrows=arrows,
            neighbor_size_min=neighbor_size_min, neighbor_size_max=neighbor_size_max, center_size=center_size
        )
        if fig_star is not None:
            st.pyplot(fig_star)
        else:
            st.info("No data to draw the star.")
    except Exception as e:
        st.error(f"Error while generating the star: {e}")

    st.markdown("---")
    st.write("Optional: load a CSV with precomputed results.")
    df = csv_loader_ui(default_name="top50_neighbors_betweenness.csv")
    show_basic_bar_if_columns(df, x_col="Node ID", y_col="Betweenness Centrality", title="Top by Betweenness Centrality (CSV)")

# =========================
# CLOSENESS
# =========================
elif section == "Closeness":
    st.title("📍 Closeness Centrality")

    # Reuse graph or allow loading
    G = st.session_state.get("G")
    if G is None:
        st.warning("No graph loaded from Home. You can load your `.txt` below.")
        G = graph_loader_ui(default_name="facebook_combined.txt", key_prefix="closeness")
    if G is None:
        st.stop()

    with st.expander("⚙️ Visualization parameters (Closeness)", expanded=True):
        k_neighbors = st.slider("Neighbors to show for the highest-closeness node", 5, 100, 50, 5)
        cmap_name = st.selectbox("Colormap", ["plasma", "viridis", "magma", "cividis", "inferno"], 0)
        radius = st.slider("Star layout radius", 3.0, 12.0, 6.0, 0.5)
        arrows = st.checkbox("Show arrows", value=False)
        neighbor_size_min = st.slider("Min neighbor size", 200, 1000, 420, 20)
        neighbor_size_max = st.slider("Max neighbor size", 800, 3000, 1200, 50)
        center_size = st.slider("Center node size", 1000, 6000, 3000, 100)

    closeness_centrality = ensure_closeness_centrality(G)
    if not closeness_centrality:
        st.error("Could not compute closeness centrality.")
        st.stop()

    st.subheader("Top 10 Nodes by Closeness Centrality")
    try:
        df_close_top10, fig_close = plot_top10_bars_closeness(
            closeness_centrality,
            'Closeness Centrality',
            title="Top 10 Nodes by Closeness Centrality",
            savepath=None
        )
        st.pyplot(fig_close)
        st.dataframe(df_close_top10, width="stretch")
    except Exception as e:
        st.error(f"Error while plotting Top 10: {e}")

    st.subheader(f"Highest-Closeness node and its {k_neighbors} strongest direct connections")
    try:
        df_neighbors_close, subG_close = top_k_neighbors_of_top_closeness(
            G,
            k=10_000,  # get all and then truncate
            closeness_centrality=closeness_centrality,
            output_file="top50_neighbors_closeness_full.csv"
        )
        df_neighbors_close = df_neighbors_close.head(k_neighbors).copy()

        st.caption("Neighbors table (sorted by neighbor Closeness):")
        st.dataframe(df_neighbors_close, width="stretch")

        fig_star_close = plot_star_by_closeness(
            df_neighbors_close,
            value_col="Closeness",
            title=f"Node with Highest Closeness and Its {k_neighbors} Strongest Direct Connections",
            radius=radius, start_angle_deg=90, cmap_name=cmap_name, arrows=arrows,
            neighbor_size_min=neighbor_size_min, neighbor_size_max=neighbor_size_max, center_size=center_size
        )
        if fig_star_close is not None:
            st.pyplot(fig_star_close)
        else:
            st.info("No data to draw the star.")
    except Exception as e:
        st.error(f"Error while generating the star: {e}")

    st.markdown("---")
    st.write("Optional: load a CSV with precomputed results.")
    df = csv_loader_ui(default_name="closeness_centrality_results.csv")
    show_basic_bar_if_columns(df, x_col="Node ID", y_col="Closeness Centrality", title="Top by Closeness Centrality (CSV)")
