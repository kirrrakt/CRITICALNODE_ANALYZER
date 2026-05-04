import streamlit as st
import networkx as nx
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import json
import random
from collections import defaultdict

# ─── Конфігурація сторінки ────────────────────────────────────────────────────
st.set_page_config(
    page_title="CriticalNode — Аналіз вузлів мережі",
    page_icon="🔴",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS стилі ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Rajdhani', sans-serif;
    background-color: #050a0f;
    color: #c8d8e8;
}

/* Головний заголовок */
.main-header {
    background: linear-gradient(135deg, #0d1b2a 0%, #1a2f45 50%, #0d1b2a 100%);
    border: 1px solid #1e4d6b;
    border-left: 4px solid #ff3c3c;
    padding: 20px 28px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.main-header::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, #ff3c3c, transparent);
}
.main-header h1 {
    font-family: 'Share Tech Mono', monospace;
    color: #ff3c3c;
    font-size: 1.8rem;
    margin: 0;
    letter-spacing: 2px;
    text-shadow: 0 0 20px rgba(255,60,60,0.4);
}
.main-header p {
    color: #7a9bb5;
    margin: 6px 0 0 0;
    font-size: 0.95rem;
    letter-spacing: 1px;
}

/* Метрик-карточки */
.metric-card {
    background: linear-gradient(135deg, #0d1b2a, #112233);
    border: 1px solid #1e3a52;
    border-top: 2px solid;
    padding: 16px 20px;
    margin-bottom: 12px;
    position: relative;
}
.metric-card.critical { border-top-color: #ff3c3c; }
.metric-card.warning  { border-top-color: #ffa500; }
.metric-card.info     { border-top-color: #00bfff; }
.metric-card.success  { border-top-color: #00ff88; }

.metric-value {
    font-family: 'Share Tech Mono', monospace;
    font-size: 2rem;
    font-weight: bold;
    line-height: 1;
}
.metric-label {
    font-size: 0.8rem;
    color: #5a7a95;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-top: 4px;
}

/* Таблиця */
.stDataFrame { border: 1px solid #1e3a52 !important; }

/* Секційний заголовок */
.section-title {
    font-family: 'Share Tech Mono', monospace;
    color: #00bfff;
    font-size: 0.85rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    border-bottom: 1px solid #1e3a52;
    padding-bottom: 8px;
    margin: 20px 0 14px 0;
}

/* Попередження про критичний вузол */
.critical-alert {
    background: rgba(255, 60, 60, 0.08);
    border: 1px solid rgba(255, 60, 60, 0.3);
    border-left: 3px solid #ff3c3c;
    padding: 12px 16px;
    margin: 8px 0;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.85rem;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #080f18 !important;
    border-right: 1px solid #1e3a52;
}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stNumberInput label {
    color: #7a9bb5 !important;
    font-size: 0.85rem;
    letter-spacing: 1px;
    text-transform: uppercase;
}

/* Кнопки */
.stButton > button {
    background: linear-gradient(135deg, #1a2f45, #0d1b2a) !important;
    border: 1px solid #1e4d6b !important;
    color: #00bfff !important;
    font-family: 'Share Tech Mono', monospace !important;
    letter-spacing: 1px !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    border-color: #00bfff !important;
    box-shadow: 0 0 12px rgba(0,191,255,0.2) !important;
}

/* Скрити Streamlit меню */
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─── Функції обчислення метрик ───────────────────────────────────────────────

def compute_all_metrics(G: nx.Graph) -> pd.DataFrame:
    """Обчислює всі графові метрики для кожного вузла."""
    n = G.number_of_nodes()
    if n == 0:
        return pd.DataFrame()

    degree_cent     = nx.degree_centrality(G)
    betweenness     = nx.betweenness_centrality(G, normalized=True)
    closeness       = nx.closeness_centrality(G)

    try:
        eigenvector = nx.eigenvector_centrality(G, max_iter=500)
    except nx.PowerIterationFailedConvergence:
        eigenvector = {v: 0.0 for v in G.nodes()}

    # Локальний кластерний коефіцієнт
    clustering      = nx.clustering(G)

    rows = []
    for node in G.nodes():
        rows.append({
            "Вузол":              str(node),
            "Степінь":            G.degree(node),
            "Degree Centrality":  round(degree_cent[node], 4),
            "Betweenness":        round(betweenness[node], 4),
            "Closeness":          round(closeness[node], 4),
            "Eigenvector":        round(eigenvector[node], 4),
            "Clustering Coeff":   round(clustering[node], 4),
        })

    df = pd.DataFrame(rows)

    # Зважена сукупна оцінка критичності
    df["Критичність"] = (
        0.35 * df["Betweenness"] +
        0.25 * df["Degree Centrality"] +
        0.25 * df["Closeness"] +
        0.15 * df["Eigenvector"]
    ).round(4)

    df = df.sort_values("Критичність", ascending=False).reset_index(drop=True)
    return df


def find_critical_nodes(G: nx.Graph, df: pd.DataFrame, top_n: int = 3) -> list:
    """Повертає список критичних вузлів."""
    # Також перевіряємо точки зчленування (articulation points)
    art_points = set(nx.articulation_points(G)) if not G.is_directed() else set()

    critical = []
    top_nodes = df.head(top_n)["Вузол"].tolist()

    for node in df["Вузол"].tolist():
        if node in top_nodes or node in art_points:
            critical.append(node)

    return list(set(critical))


def simulate_removal(G: nx.Graph, node: str) -> dict:
    """Симулює видалення вузла та аналізує наслідки."""
    try:
        node_int = int(node)
    except ValueError:
        node_int = node

    if node_int not in G.nodes():
        return {}

    # До видалення
    components_before = nx.number_connected_components(G)
    if nx.is_connected(G):
        avg_path_before = nx.average_shortest_path_length(G)
        diameter_before = nx.diameter(G)
    else:
        largest = max(nx.connected_components(G), key=len)
        sub = G.subgraph(largest)
        avg_path_before = nx.average_shortest_path_length(sub)
        diameter_before = nx.diameter(sub)

    # Після видалення
    G2 = G.copy()
    G2.remove_node(node_int)
    components_after = nx.number_connected_components(G2)

    if G2.number_of_nodes() > 0 and nx.is_connected(G2):
        avg_path_after = nx.average_shortest_path_length(G2)
        diameter_after = nx.diameter(G2)
    elif G2.number_of_nodes() > 1:
        largest2 = max(nx.connected_components(G2), key=len)
        sub2 = G2.subgraph(largest2)
        avg_path_after  = nx.average_shortest_path_length(sub2)
        diameter_after  = nx.diameter(sub2)
    else:
        avg_path_after = 0
        diameter_after = 0

    delta_comp = components_after - components_before
    delta_path = avg_path_after - avg_path_before

    return {
        "components_before": components_before,
        "components_after":  components_after,
        "delta_components":  delta_comp,
        "avg_path_before":   round(avg_path_before, 3),
        "avg_path_after":    round(avg_path_after, 3),
        "delta_path":        round(delta_path, 3),
        "diameter_before":   diameter_before,
        "diameter_after":    diameter_after,
        "nodes_lost":        G.degree(node_int),
    }


# ─── Побудова графу ───────────────────────────────────────────────────────────

def build_random_graph(n_nodes, n_edges, seed):
    random.seed(seed)
    G = nx.gnm_random_graph(n_nodes, n_edges, seed=seed)
    # Якщо граф незв'язний — додаємо ребра для зв'язності
    components = list(nx.connected_components(G))
    for i in range(1, len(components)):
        u = random.choice(list(components[0]))
        v = random.choice(list(components[i]))
        G.add_edge(u, v)
    return G


def build_graph_from_edges(edge_text: str):
    G = nx.Graph()
    for line in edge_text.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.replace(",", " ").split()
        if len(parts) >= 2:
            try:
                u, v = int(parts[0]), int(parts[1])
                G.add_edge(u, v)
            except ValueError:
                u, v = parts[0], parts[1]
                G.add_edge(u, v)
    return G


# ─── Візуалізація графу ───────────────────────────────────────────────────────

def draw_graph(G: nx.Graph, df: pd.DataFrame, critical_nodes: list,
               selected_node: str = None, layout_type: str = "spring") -> go.Figure:

    # Розміщення вузлів
    pos_funcs = {
        "spring":    lambda: nx.spring_layout(G, seed=42, k=2.5 / np.sqrt(G.number_of_nodes())),
        "kamada":    lambda: nx.kamada_kawai_layout(G),
        "circular":  lambda: nx.circular_layout(G),
        "spectral":  lambda: nx.spectral_layout(G),
    }
    pos = pos_funcs.get(layout_type, pos_funcs["spring"])()

    # Метрика критичності для кольору
    crit_map = dict(zip(df["Вузол"], df["Критичність"]))

    # Ребра
    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(width=1.2, color="rgba(30,77,107,0.6)"),
        hoverinfo="none",
        name="Ребра",
    )

    # Вузли
    node_x, node_y, node_text, node_hover = [], [], [], []
    node_color, node_size, node_border_color = [], [], []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

        crit = crit_map.get(str(node), 0)
        deg  = G.degree(node)
        node_text.append(str(node))
        node_hover.append(
            f"<b>Вузол {node}</b><br>"
            f"Степінь: {deg}<br>"
            f"Критичність: {crit:.4f}<br>"
            f"Betweenness: {df[df['Вузол']==str(node)]['Betweenness'].values[0] if str(node) in df['Вузол'].values else 0:.4f}"
        )

        is_critical = str(node) in critical_nodes
        is_selected = str(node) == selected_node

        if is_selected:
            node_color.append("#ffffff")
            node_size.append(28)
            node_border_color.append("#ffffff")
        elif is_critical:
            node_color.append(crit)
            node_size.append(22)
            node_border_color.append("#ff3c3c")
        else:
            node_color.append(crit)
            node_size.append(14)
            node_border_color.append("#1e4d6b")

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        text=node_text,
        textposition="top center",
        textfont=dict(family="Share Tech Mono", size=10, color="#c8d8e8"),
        hovertext=node_hover,
        hoverinfo="text",
        marker=dict(
            size=node_size,
            color=node_color,
            colorscale=[
                [0.0,  "#1a2f45"],
                [0.3,  "#0066aa"],
                [0.6,  "#ff8800"],
                [1.0,  "#ff3c3c"],
            ],
            cmin=0, cmax=df["Критичність"].max() if not df.empty else 1,
            colorbar=dict(
                title=dict(text="Критичність", font=dict(color="#7a9bb5", size=11)),
                tickfont=dict(color="#7a9bb5", size=10),
                bgcolor="rgba(0,0,0,0)",
                bordercolor="#1e3a52",
                thickness=12,
            ),
            line=dict(color=node_border_color, width=2),
            showscale=True,
        ),
        name="Вузли",
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        paper_bgcolor="#050a0f",
        plot_bgcolor="#080f18",
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False,
        hovermode="closest",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   showline=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   showline=False),
        font=dict(family="Rajdhani", color="#c8d8e8"),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  ГОЛОВНИЙ ІНТЕРФЕЙС
# ═══════════════════════════════════════════════════════════════════════════════

# Заголовок
st.markdown("""
<div class="main-header">
  <h1>⬡ CRITICALNODE ANALYZER</h1>
  <p>СИСТЕМА ВИЯВЛЕННЯ КРИТИЧНИХ ВУЗЛІВ КОМП'ЮТЕРНОЇ МЕРЕЖІ НА ОСНОВІ ГРАФОВИХ МЕТРИК</p>
</div>
""", unsafe_allow_html=True)

# ─── Бічна панель ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="section-title">⚙ ПАРАМЕТРИ МЕРЕЖІ</p>', unsafe_allow_html=True)

    input_mode = st.selectbox(
        "Джерело графу",
        ["🎲 Випадковий граф", "✏️ Власні ребра", "📋 Приклад мережі"],
    )

    if input_mode == "🎲 Випадковий граф":
        n_nodes = st.slider("Кількість вузлів", 6, 40, 15)
        n_edges = st.slider("Кількість ребер",
                            n_nodes - 1,
                            min(n_nodes * (n_nodes - 1) // 2, n_nodes * 4),
                            min(n_nodes * 2, n_nodes * (n_nodes - 1) // 2))
        seed = st.number_input("Seed (відтворюваність)", 0, 9999, 42)
        G = build_random_graph(n_nodes, n_edges, int(seed))

    elif input_mode == "✏️ Власні ребра":
        st.caption("Введіть ребра (по одному на рядок): вузол1 вузол2")
        edge_text = st.text_area(
            "Список ребер",
            value="0 1\n0 2\n1 2\n1 3\n2 4\n3 4\n3 5\n4 6\n5 6\n6 7",
            height=200,
        )
        G = build_graph_from_edges(edge_text)

    else:  # Приклад мережі
        preset = st.selectbox("Оберіть топологію", [
            "Корпоративна мережа (зіркова)",
            "Mesh-мережа",
            "Ієрархічна мережа",
        ])
        if preset == "Корпоративна мережа (зіркова)":
            G = nx.star_graph(10)
            nx.add_path(G, [1, 11, 12, 2])
            nx.add_path(G, [3, 13, 14, 4])
        elif preset == "Mesh-мережа":
            G = nx.grid_2d_graph(4, 4)
            G = nx.convert_node_labels_to_integers(G)
        else:
            G = nx.balanced_tree(3, 3)

    st.markdown('<p class="section-title">🔬 АНАЛІЗ</p>', unsafe_allow_html=True)

    top_n = st.slider("Топ критичних вузлів", 1, 10, 3)

    layout_type = st.selectbox(
        "Розміщення вузлів",
        ["spring", "kamada", "circular", "spectral"],
        format_func=lambda x: {
            "spring":   "Spring (фізична симуляція)",
            "kamada":   "Kamada-Kawai (естетичний)",
            "circular": "Circular (кільцевий)",
            "spectral": "Spectral (спектральний)",
        }[x],
    )

    st.markdown('<p class="section-title">🎯 СИМУЛЯЦІЯ АТАКИ</p>', unsafe_allow_html=True)
    selected_node = st.selectbox(
        "Видалити вузол",
        ["— не обрано —"] + [str(n) for n in sorted(G.nodes())],
    )
    if selected_node == "— не обрано —":
        selected_node = None

# ─── Обчислення ───────────────────────────────────────────────────────────────
if G.number_of_nodes() == 0:
    st.error("Граф порожній. Перевірте введені ребра.")
    st.stop()

df_metrics    = compute_all_metrics(G)
critical_list = find_critical_nodes(G, df_metrics, top_n)
art_points    = list(nx.articulation_points(G))

# ─── Верхній рядок метрик ─────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card info">
      <div class="metric-value" style="color:#00bfff">{G.number_of_nodes()}</div>
      <div class="metric-label">Вузлів у мережі</div>
    </div>""", unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card info">
      <div class="metric-value" style="color:#00bfff">{G.number_of_edges()}</div>
      <div class="metric-label">Ребер (з'єднань)</div>
    </div>""", unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card critical">
      <div class="metric-value" style="color:#ff3c3c">{len(critical_list)}</div>
      <div class="metric-label">Критичних вузлів</div>
    </div>""", unsafe_allow_html=True)

with col4:
    conn_status = "ЗВ'ЯЗНИЙ" if nx.is_connected(G) else "НЕ ЗВ'ЯЗНИЙ"
    conn_color  = "#00ff88" if nx.is_connected(G) else "#ff3c3c"
    card_class  = "success" if nx.is_connected(G) else "critical"
    st.markdown(f"""
    <div class="metric-card {card_class}">
      <div class="metric-value" style="color:{conn_color}; font-size:1.3rem">{conn_status}</div>
      <div class="metric-label">Стан мережі</div>
    </div>""", unsafe_allow_html=True)

# ─── Основна область ──────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "🌐  Граф мережі",
    "📊  Метрики вузлів",
    "💣  Симуляція атаки",
])

# ══ TAB 1: Граф ══════════════════════════════════════════════════════════════
with tab1:
    left, right = st.columns([3, 1])

    with left:
        fig = draw_graph(G, df_metrics, critical_list, selected_node, layout_type)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with right:
        st.markdown('<p class="section-title">🔴 КРИТИЧНІ ВУЗЛИ</p>', unsafe_allow_html=True)

        top_df = df_metrics.head(top_n)
        for _, row in top_df.iterrows():
            is_art = row["Вузол"] in [str(a) for a in art_points]
            tag = " ⚠ ТОЧКА ЗЧЛЕНУВАННЯ" if is_art else ""
            st.markdown(f"""
            <div class="critical-alert">
              <b>Вузол {row['Вузол']}</b>{tag}<br>
              Критичність: <b>{row['Критичність']:.4f}</b><br>
              Betweenness: {row['Betweenness']:.4f} | Ступінь: {int(row['Степінь'])}
            </div>""", unsafe_allow_html=True)

        if art_points:
            st.markdown('<p class="section-title">⚠ ТОЧКИ ЗЧЛЕНУВАННЯ</p>', unsafe_allow_html=True)
            st.markdown(f"""
            <div style="font-family:'Share Tech Mono',monospace; font-size:0.8rem; color:#ffa500;">
            Видалення цих вузлів розриває мережу:<br><br>
            {', '.join(str(a) for a in sorted(art_points))}
            </div>""", unsafe_allow_html=True)

        st.markdown('<p class="section-title">📌 ЛЕГЕНДА</p>', unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size:0.8rem; line-height:2; font-family:'Share Tech Mono',monospace;">
        🔴 Висока критичність<br>
        🟠 Середня критичність<br>
        🔵 Низька критичність<br>
        ⬜ Обраний вузол
        </div>""", unsafe_allow_html=True)

# ══ TAB 2: Метрики ═══════════════════════════════════════════════════════════
with tab2:
    st.markdown('<p class="section-title">📐 ГРАФОВІ МЕТРИКИ ЦЕНТРАЛЬНОСТІ</p>',
                unsafe_allow_html=True)

    # Пояснення метрик
    with st.expander("ℹ️ Опис метрик"):
        st.markdown("""
**Degree Centrality** — частка сусідів вузла від максимально можливого.
Чим більше з'єднань — тим вищий вплив на локальному рівні.

**Betweenness Centrality** — частка найкоротших шляхів між усіма парами
вузлів, що проходять через даний вузол. Висока betweenness = «міст» у мережі.

**Closeness Centrality** — зворотна середня відстань від вузла до всіх інших.
Показує, наскільки швидко інформація досягає решти мережі.

**Eigenvector Centrality** — важливість вузла з урахуванням важливості його
сусідів. Аналог PageRank.

**Clustering Coefficient** — ступінь кластеризації оточення вузла (0–1).

**Критичність** — зважена оцінка: 0.35 × Betweenness + 0.25 × Degree
+ 0.25 × Closeness + 0.15 × Eigenvector.
        """)

    # Таблиця з підсвічуванням
    def highlight_critical(row):
        if row["Вузол"] in critical_list:
            return ["background-color: rgba(255,60,60,0.12); color: #ff9999"] * len(row)
        return [""] * len(row)

    styled = df_metrics.style.apply(highlight_critical, axis=1).format({
        "Degree Centrality": "{:.4f}",
        "Betweenness":       "{:.4f}",
        "Closeness":         "{:.4f}",
        "Eigenvector":       "{:.4f}",
        "Clustering Coeff":  "{:.4f}",
        "Критичність":       "{:.4f}",
    })
    st.dataframe(styled, use_container_width=True, height=400)

    # Порівняльна гістограма
    st.markdown('<p class="section-title">📈 ПОРІВНЯННЯ МЕТРИК (ТОП-15)</p>',
                unsafe_allow_html=True)

    top15 = df_metrics.head(15)
    metric_choice = st.selectbox(
        "Метрика для візуалізації",
        ["Критичність", "Betweenness", "Degree Centrality", "Closeness", "Eigenvector"],
    )

    colors = [
        "#ff3c3c" if v in critical_list else "#1e6fa8"
        for v in top15["Вузол"]
    ]

    bar_fig = go.Figure(go.Bar(
        x=top15["Вузол"],
        y=top15[metric_choice],
        marker_color=colors,
        marker_line_color="#1e3a52",
        marker_line_width=1,
        text=top15[metric_choice].round(4),
        textposition="outside",
        textfont=dict(family="Share Tech Mono", size=9, color="#c8d8e8"),
    ))
    bar_fig.update_layout(
        paper_bgcolor="#050a0f",
        plot_bgcolor="#080f18",
        xaxis=dict(
            title="Вузол",
            color="#7a9bb5",
            gridcolor="#112233",
            tickfont=dict(family="Share Tech Mono", size=10),
        ),
        yaxis=dict(
            title=metric_choice,
            color="#7a9bb5",
            gridcolor="#112233",
        ),
        font=dict(color="#c8d8e8"),
        margin=dict(t=20, b=40),
        height=320,
    )
    st.plotly_chart(bar_fig, use_container_width=True, config={"displayModeBar": False})

# Радарний графік для топ-3
st.markdown(
    '<p class="section-title">🕸 ПРОФІЛЬ КРИТИЧНИХ ВУЗЛІВ (ТОП-3)</p>',
    unsafe_allow_html=True
)

radar_metrics = [
    "Degree Centrality",
    "Betweenness",
    "Closeness",
    "Eigenvector",
    "Clustering Coeff"
]

radar_fig = go.Figure()

for _, row in df_metrics.head(3).iterrows():
    vals = [row[m] for m in radar_metrics]
    vals += [vals[0]]

    radar_fig.add_trace(go.Scatterpolar(
        r=vals,
        theta=radar_metrics + [radar_metrics[0]],
        fill='toself',
    fillcolor='rgba(255, 60, 60, 0.1)',
    line=dict(color='rgb(255, 60, 60)'),
    opacity=0.8,
))

    radar_fig.update_layout(
        polar=dict(
            bgcolor="#080f18",
            radialaxis=dict(visible=True, color="#3a5a78", gridcolor="#1e3a52"),
            angularaxis=dict(color="#7a9bb5", gridcolor="#1e3a52"),
        ),
        paper_bgcolor="#050a0f",
        font=dict(color="#c8d8e8"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#c8d8e8")),
        height=380,
        margin=dict(t=20),
    )
    st.plotly_chart(radar_fig, use_container_width=True, config={"displayModeBar": False})

# ══ TAB 3: Симуляція атаки ═══════════════════════════════════════════════════
with tab3:
    st.markdown('<p class="section-title">💣 СИМУЛЯЦІЯ ВИДАЛЕННЯ ВУЗЛА</p>',
                unsafe_allow_html=True)

    if selected_node is None:
        st.info("Оберіть вузол у бічній панелі для симуляції видалення.")

        # Показуємо автоматичний аналіз найкритичнішого
        st.markdown("**Автоматичний аналіз найкритичнішого вузла:**")
        selected_node = df_metrics.iloc[0]["Вузол"]
        auto = True
    else:
        auto = False

    result = simulate_removal(G, selected_node)

    if result:
        c1, c2, c3 = st.columns(3)
        with c1:
            delta_c = result["delta_components"]
            clr = "#ff3c3c" if delta_c > 0 else "#00ff88"
            st.markdown(f"""
            <div class="metric-card {'critical' if delta_c > 0 else 'success'}">
              <div class="metric-value" style="color:{clr}">
                {result['components_before']} → {result['components_after']}
              </div>
              <div class="metric-label">Компонент зв'язності (до → після)</div>
            </div>""", unsafe_allow_html=True)

        with c2:
            dp = result["delta_path"]
            clr2 = "#ff3c3c" if dp > 0.5 else ("#ffa500" if dp > 0 else "#00ff88")
            st.markdown(f"""
            <div class="metric-card {'critical' if dp > 0.5 else 'warning'}">
              <div class="metric-value" style="color:{clr2}">
                {result['avg_path_before']} → {result['avg_path_after']}
              </div>
              <div class="metric-label">Середній шлях (до → після)</div>
            </div>""", unsafe_allow_html=True)

        with c3:
            st.markdown(f"""
            <div class="metric-card warning">
              <div class="metric-value" style="color:#ffa500">
                {result['nodes_lost']}
              </div>
              <div class="metric-label">Вузлів втрачає прямий зв'язок</div>
            </div>""", unsafe_allow_html=True)

        # Аналіз наслідків
        st.markdown('<p class="section-title">📋 АНАЛІЗ НАСЛІДКІВ</p>',
                    unsafe_allow_html=True)

        severity = "КРИТИЧНИЙ" if result["delta_components"] > 0 else (
            "ВИСОКИЙ" if result["delta_path"] > 1.0 else (
                "СЕРЕДНІЙ" if result["delta_path"] > 0.3 else "НИЗЬКИЙ"
            )
        )
        sev_color = {"КРИТИЧНИЙ": "#ff3c3c", "ВИСОКИЙ": "#ff6600",
                     "СЕРЕДНІЙ": "#ffa500", "НИЗЬКИЙ": "#00ff88"}[severity]

        st.markdown(f"""
        <div style="border:1px solid {sev_color}; border-left:4px solid {sev_color};
                    background:rgba(0,0,0,0.3); padding:16px; margin:12px 0;
                    font-family:'Share Tech Mono',monospace;">
          <div style="color:{sev_color}; font-size:1.1rem; margin-bottom:10px;">
            ▶ РІВЕНЬ ЗАГРОЗИ: {severity}
          </div>
          {"<b>⚠ МЕРЕЖА РОЗПАДАЄТЬСЯ НА ЧАСТИНИ!</b> Видалення вузла " + selected_node +
           " призводить до появи " + str(result['delta_components']) +
           " нової(их) ізольованої(их) підмережі." if result["delta_components"] > 0 else
           "Мережа залишається зв'язною, але деградує."}
          <br><br>
          • Середній найкоротший шлях {'збільшується' if result['delta_path'] > 0 else 'зменшується'} 
            на <b>{abs(result['delta_path']):.3f}</b><br>
          • Діаметр мережі: {result['diameter_before']} → {result['diameter_after']}<br>
          • Прямих сусідів втрачає зв'язок: {result['nodes_lost']}
        </div>""", unsafe_allow_html=True)

        # Граф після видалення
        st.markdown('<p class="section-title">🔍 ГРАФ ПІСЛЯ ВИДАЛЕННЯ ВУЗЛА</p>',
                    unsafe_allow_html=True)

        try:
            node_int = int(selected_node)
        except ValueError:
            node_int = selected_node

        G_after = G.copy()
        G_after.remove_node(node_int)

        if G_after.number_of_nodes() > 0:
            df_after = compute_all_metrics(G_after)
            crit_after = find_critical_nodes(G_after, df_after, top_n)
            fig_after = draw_graph(G_after, df_after, crit_after, None, layout_type)
            st.plotly_chart(fig_after, use_container_width=True,
                            config={"displayModeBar": False})

    # Масовий аналіз (топ-10)
    st.markdown('<p class="section-title">🏆 ПОРІВНЯЛЬНИЙ АНАЛІЗ ВПЛИВУ ВСІХ ВУЗЛІВ</p>',
                unsafe_allow_html=True)

    if st.button("⚡ Запустити повний аналіз (всі вузли)"):
        results_all = []
        nodes_list  = list(G.nodes())
        prog = st.progress(0)

        for i, nd in enumerate(nodes_list):
            res = simulate_removal(G, str(nd))
            if res:
                results_all.append({
                    "Вузол":         str(nd),
                    "Δ Компоненти":  res["delta_components"],
                    "Δ Шлях":        res["delta_path"],
                    "Ступінь":       G.degree(nd),
                    "Ризик":         (res["delta_components"] * 10 +
                                      res["delta_path"] * 2 +
                                      G.degree(nd) * 0.5),
                })
            prog.progress((i + 1) / len(nodes_list))

        prog.empty()
        df_all = pd.DataFrame(results_all).sort_values("Ризик", ascending=False)

        st.dataframe(df_all.style.background_gradient(
            subset=["Ризик"], cmap="RdYlGn_r"
        ).format({"Δ Шлях": "{:.3f}", "Ризик": "{:.2f}"}),
            use_container_width=True)

        # Scatter: степінь vs вплив на шлях
        scatter_fig = go.Figure(go.Scatter(
            x=df_all["Ступінь"],
            y=df_all["Δ Шлях"],
            mode="markers+text",
            text=df_all["Вузол"],
            textposition="top center",
            textfont=dict(family="Share Tech Mono", size=9),
            marker=dict(
                size=df_all["Ризик"].clip(lower=5) * 2,
                color=df_all["Ризик"],
                colorscale=[[0, "#1a2f45"], [0.5, "#ffa500"], [1, "#ff3c3c"]],
                showscale=True,
                colorbar=dict(title="Ризик", tickfont=dict(color="#7a9bb5")),
                line=dict(color="#1e3a52", width=1),
            ),
            hovertemplate="<b>Вузол %{text}</b><br>Ступінь: %{x}<br>Δ Шлях: %{y:.3f}<extra></extra>",
        ))
        scatter_fig.update_layout(
            paper_bgcolor="#050a0f", plot_bgcolor="#080f18",
            xaxis=dict(title="Ступінь вузла", color="#7a9bb5", gridcolor="#112233"),
            yaxis=dict(title="Зміна середнього шляху (Δ)", color="#7a9bb5", gridcolor="#112233"),
            font=dict(color="#c8d8e8"),
            margin=dict(t=20),
            height=350,
        )
        st.plotly_chart(scatter_fig, use_container_width=True,
                        config={"displayModeBar": False})

# ─── Підвал ──────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding:20px; color:#2a4a65;
            font-family:'Share Tech Mono',monospace; font-size:0.75rem;
            border-top:1px solid #1e3a52; margin-top:32px;">
  CRITICALNODE ANALYZER v1.0 · Курсова робота · ХНУВС ННІ №4 · 2026
</div>""", unsafe_allow_html=True)
