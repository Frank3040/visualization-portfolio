# 🕸️ Facebook Network Analysis — Streamlit App

An interactive **Streamlit application** designed to analyze a Facebook social network using an **edge list** (`facebook_combined.txt`).  
It includes dedicated sections for **Degree**, **Betweenness**, and **Closeness** centralities, each featuring Top-10 charts and a **star-layout visualization** showing the most central node and its top *k* neighbors.

---

## ✨ Features

- **Automatic graph loading** from `facebook_combined.txt` (no manual upload required).  
- Four main tabs: **Home**, **Degree**, **Betweenness**, and **Closeness**.  
- **Top-10 ranking visualizations** (horizontal bar charts).  
- **Star-shaped visual layout** displaying:
  - Node size proportional to its centrality value.
  - Color mapped to metric intensity via customizable colormap.
  - Adjustable parameters (radius, node sizes, arrows, colors).  
- **Efficient caching system** for graph and metrics (using Streamlit’s session state).  
- Smart error handling and informative messages for missing files or empty data.  

---

## 🗂️ Expected Project Structure

```
your_project/
├─ app.py
└─ facebook_combined.txt       # Required edge list file
```

**Edge list format:**  
A plain text file with space-separated pairs of node IDs per line, e.g.:

```
0 1
0 2
1 3
...
```

> The graph is loaded as an **undirected network** (`nx.Graph()`), and all node IDs are treated as **integers**.

---

## 🔧 Requirements

- **Python** 3.9 or higher  
- Required libraries:
  - `streamlit`
  - `networkx`
  - `matplotlib`
  - `pandas`
  - `numpy`

### Installation (recommended)

```bash
python3 -m venv .venv
source .venv/bin/activate     # Windows: .\.venv\Scripts\Activate.ps1
pip install -U streamlit networkx matplotlib pandas numpy
```

---

## ▶️ How to Run

1. Place both `app.py` and `facebook_combined.txt` in the **same directory**.  
2. Start the Streamlit app:

```bash
streamlit run app.py
```

The app will open automatically in your browser at **http://localhost:8501**.  
If the port is busy, use:
```bash
streamlit run app.py --server.port 8502
```

> **Running on a remote server (e.g., via SSH):**  
> ```bash
> streamlit run app.py --server.address 0.0.0.0 --server.port 8501
> ```  
> Then access it at `http://<your_server_ip>:8501`.

---

## ⚙️ Configuration

- **Edge file name:**  
  The app automatically looks for `facebook_combined.txt`:
  ```python
  EDGE_FILE = "facebook_combined.txt"
  ```
  You can rename the file or update this line if needed.

- **Betweenness approximation (`k`):**  
  By default, the app uses a **sampling size of k=256** for faster computation on large networks.  
  You can disable or adjust this approximation directly from the **Betweenness** panel.

---

## 🧭 Section Overview

### 🏠 Home
- Displays general information and a **Network Topology Summary** table, including:
  - Nodes, Edges, Average Degree, Density (~0.010820)
  - Connected Components (1)
  - Diameter (8)
  - Average Shortest Path Length (3.69)  
- If the edge file is missing, default placeholder metrics are shown.

---

### 📈 Degree Centrality

**Concept:**  
Degree centrality measures how many **direct connections** a node has.  
In social networks, it identifies **highly connected users** — individuals who interact with many others.  
Formally:
\[
C_D(v) = \frac{\text{deg}(v)}{n-1}
\]
where \( \text{deg}(v) \) is the number of neighbors of node *v* and *n* is the total number of nodes.

**In this section:**
1. A **Top-10 ranking** of nodes by degree centrality (horizontal bar chart).  
2. A **data table** showing node IDs and their degree values.  
3. A **star-layout visualization** of the node with the highest degree:
   - You can select *k*, the number of neighbors to display.
   - Node size and color intensity represent degree centrality.
   - Adjustable parameters: radius, color palette, node sizes, and arrows.

---

### 🪢 Betweenness Centrality

**Concept:**  
Betweenness centrality quantifies how often a node lies on the **shortest paths** between other nodes.  
Nodes with high betweenness act as **bridges or intermediaries** between communities, crucial for information flow.  
Formally:
\[
C_B(v) = \sum_{s \neq v \neq t} \frac{\sigma_{st}(v)}{\sigma_{st}}
\]
where \( \sigma_{st} \) is the number of shortest paths from *s* to *t*, and \( \sigma_{st}(v) \) counts those passing through *v*.

**In this section:**
1. Option to compute **exact** or **approximate betweenness** (adjustable sampling `k`).  
2. **Top-10 nodes** by betweenness centrality.  
3. A **star visualization** of the most central node and its *k* strongest neighbors:
   - Nodes closer to the center bridge more paths.
   - Colors and sizes highlight their influence within the network.

---

### 📍 Closeness Centrality

**Concept:**  
Closeness centrality measures how **near** a node is to all others in the network.  
A node with high closeness can **reach others quickly**, making it efficient for spreading information.  
Formally:
\[
C_C(v) = \frac{1}{\sum_{u} d(v, u)}
\]
where \( d(v, u) \) is the shortest path distance between *v* and *u*.

**In this section:**
1. **Top-10 nodes** with the highest closeness centrality.  
2. **Data table** and **visual star layout** of the most central node:
   - The closer the neighbors, the stronger the connection (color and size scaling).
   - Customizable visualization controls identical to other panels.

---

## 🧠 Technical Details

- **Graph loading:**
  ```python
  nx.read_edgelist(path, create_using=nx.Graph(), nodetype=int)
  ```
- **Caching:**
  - Graph: `@st.cache_resource`
  - CSV reading: `@st.cache_data`
  - Centralities stored in `st.session_state` using dynamic keys (`n`, `m`).

- **Star Visualization:**
  - Positions nodes radially using trigonometric layout.
  - **Colors** generated via `matplotlib.cm` and normalized by metric value.
  - **Node sizes** scaled linearly between min–max values of the metric.

---

## 🧪 Additional Notes (Optional)

The helper function `load_csv_from_path()` allows loading and displaying external CSV data for future extensions — e.g., integrating precomputed metrics or exporting results.

---

## 🚀 Performance Tips

- **Betweenness centrality** is computationally expensive:  
  - Use **approximation** (`k=64–256`) for large graphs.  
  - For very large datasets, precompute metrics offline with NetworkX and import them.
- Keep your environment clean by using a **virtual environment**.
- Use recent library versions for faster performance and improved visualization stability.

---

## 🧰 Troubleshooting

| Issue | Cause | Solution |
|-------|--------|-----------|
| **File not found: facebook_combined.txt** | The file isn’t in the same folder as `app.py`. | Place it next to `app.py` or update the file path. |
| **App doesn’t open in browser** | Streamlit didn’t auto-launch. | Visit `http://localhost:8501` manually or change port. |
| **Empty or missing charts** | Edge list is empty or incorrectly formatted. | Verify each line has two node IDs separated by a space. |
| **Betweenness takes too long** | Graph is large and using full computation. | Enable approximation and lower `k` to 64–128. |

---

## 📄 License

You can adapt any open-source license such as **MIT**, **Apache 2.0**, or **GPLv3**.  
Include authorship and citation if redistributing or modifying.

---

## 🙌 Credits

Developed using:
- [Streamlit](https://streamlit.io/)
- [NetworkX](https://networkx.org/)
- [Matplotlib](https://matplotlib.org/)
- [Pandas](https://pandas.pydata.org/)
- [NumPy](https://numpy.org/)

---

## ⚡ Quick Commands

```bash
# 1) Create a virtual environment and install dependencies
python3 -m venv .venv
source .venv/bin/activate            # Windows: .\.venv\Scripts\Activate.ps1
pip install -U streamlit networkx matplotlib pandas numpy

# 2) Run the Streamlit app
streamlit run app.py
```