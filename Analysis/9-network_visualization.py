from pathlib import Path
from collections import defaultdict

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


BASE_DIR = Path("/Users/maysounbrghl/Desktop/Thesis/jfk_data_analysis/data/structured_output2")
NETWORK_DIR = BASE_DIR / "network_output"

EDGES_FILE = NETWORK_DIR / "edges_weighted.csv"
COMMUNITIES_FILE = NETWORK_DIR / "communities.csv"
BROKERS_FILE = NETWORK_DIR / "hidden_brokers.csv"
NODES_FILE = NETWORK_DIR / "nodes_weighted_degree.csv"

OUTPUT_DIR = BASE_DIR / "visualization_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TOP_CENTRAL_NODES = 40
TOP_BROKERS = 10
BROKER_NEIGHBORS = 8
TOP_COMMUNITIES = 8
CHUNK_SIZE = 500_000


print("Loading nodes...")

nodes = pd.read_csv(NODES_FILE)
nodes.columns = [c.strip() for c in nodes.columns]

node_col = "node" if "node" in nodes.columns else nodes.columns[0]
degree_col = "weighted_degree" if "weighted_degree" in nodes.columns else nodes.columns[1]

nodes[node_col] = nodes[node_col].astype(str)
nodes[degree_col] = pd.to_numeric(nodes[degree_col], errors="coerce").fillna(0)

top_nodes = set(
    nodes.sort_values(degree_col, ascending=False)
    .head(TOP_CENTRAL_NODES)[node_col]
)


print("Loading communities...")

communities = pd.read_csv(COMMUNITIES_FILE)
communities.columns = [c.strip() for c in communities.columns]

communities["node"] = communities["node"].astype(str)
node_to_comm = dict(zip(communities["node"], communities["community"]))

top_community_ids = (
    communities["community"]
    .value_counts()
    .head(TOP_COMMUNITIES)
    .index
    .tolist()
)

community_nodes = set(
    communities[communities["community"].isin(top_community_ids)]["node"]
)

selected_community_nodes = top_nodes.intersection(community_nodes)


print("Loading brokers...")

brokers = pd.read_csv(BROKERS_FILE)
brokers.columns = [c.strip() for c in brokers.columns]

brokers["entity"] = brokers["entity"].astype(str)
brokers["betweenness"] = pd.to_numeric(
    brokers["betweenness"],
    errors="coerce"
).fillna(0)

top_brokers = set(
    brokers.sort_values("betweenness", ascending=False)
    .head(TOP_BROKERS)["entity"]
)


print("Collecting relevant edges...")

central_edges = []
community_edges = []
broker_neighbor_weights = defaultdict(list)

for chunk in pd.read_csv(EDGES_FILE, chunksize=CHUNK_SIZE):
    chunk.columns = [c.strip() for c in chunk.columns]

    if not {"source", "target", "weight"}.issubset(chunk.columns):
        raise ValueError("edges_weighted.csv must contain source, target, and weight columns.")

    chunk["source"] = chunk["source"].astype(str)
    chunk["target"] = chunk["target"].astype(str)
    chunk["weight"] = pd.to_numeric(chunk["weight"], errors="coerce").fillna(1)

    for row in chunk.itertuples(index=False):
        source = row.source
        target = row.target
        weight = row.weight

        # Central actors graph
        if source in top_nodes and target in top_nodes:
            central_edges.append((source, target, weight))

        # Community graph
        if source in selected_community_nodes and target in selected_community_nodes:
            community_edges.append((source, target, weight))

        # Broker graph: collect strongest neighbors for each broker
        if source in top_brokers:
            broker_neighbor_weights[source].append((target, weight))
        if target in top_brokers:
            broker_neighbor_weights[target].append((source, weight))


def draw_network(
    G,
    title,
    output_path,
    node_size_attr=None,
    highlight_nodes=None,
    community_colors=False
):
    plt.figure(figsize=(16, 12))

    if G.number_of_nodes() == 0:
        print(f"Skipping empty graph: {title}")
        return

    pos = nx.spring_layout(G, seed=42, k=0.7)

    weights = [G[u][v].get("weight", 1) for u, v in G.edges()]
    max_weight = max(weights) if weights else 1
    edge_widths = [0.5 + 3 * (w / max_weight) for w in weights]

    if node_size_attr:
        values = [G.nodes[n].get(node_size_attr, 1) for n in G.nodes()]
        max_value = max(values) if values else 1
        sizes = [300 + 2500 * (v / max_value) for v in values]
    else:
        sizes = 700

    if community_colors:
        node_colors = [
            G.nodes[n].get("community", 0)
            for n in G.nodes()
        ]
    else:
        node_colors = [
            1 if highlight_nodes and n in highlight_nodes else 0
            for n in G.nodes()
        ]

    nx.draw_networkx_edges(
        G,
        pos,
        width=edge_widths,
        alpha=0.35
    )

    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=sizes,
        node_color=node_colors,
        cmap=plt.cm.tab20,
        alpha=0.9
    )

    labels = {}

    for node in G.nodes():
        if highlight_nodes and node in highlight_nodes:
            labels[node] = node
        elif G.degree(node) >= 2:
            labels[node] = node

    nx.draw_networkx_labels(
        G,
        pos,
        labels=labels,
        font_size=8
    )

    plt.title(title, fontsize=18)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print("Saved:", output_path)


print("Creating central actors network...")

G_central = nx.Graph()

for _, row in nodes.iterrows():
    node = str(row[node_col])

    if node in top_nodes:
        G_central.add_node(
            node,
            weighted_degree=float(row[degree_col]),
            community=node_to_comm.get(node, -1)
        )

for source, target, weight in central_edges:
    G_central.add_edge(source, target, weight=weight)

draw_network(
    G_central,
    "Central Actors in the JFK Entity Network",
    OUTPUT_DIR / "central_actors_network.png",
    node_size_attr="weighted_degree",
    community_colors=True
)


print("Creating community network...")

G_comm = nx.Graph()

for node in selected_community_nodes:
    degree_value = float(
        nodes.loc[nodes[node_col] == node, degree_col].iloc[0]
    )

    G_comm.add_node(
        node,
        weighted_degree=degree_value,
        community=node_to_comm.get(node, -1)
    )

for source, target, weight in community_edges:
    G_comm.add_edge(source, target, weight=weight)

draw_network(
    G_comm,
    "Main Communities and Central Actors",
    OUTPUT_DIR / "community_network.png",
    node_size_attr="weighted_degree",
    community_colors=True
)


print("Creating broker network...")

G_broker = nx.Graph()

broker_scores = dict(
    zip(brokers["entity"], brokers["betweenness"])
)

for broker in top_brokers:
    G_broker.add_node(
        broker,
        betweenness=broker_scores.get(broker, 0),
        community=node_to_comm.get(broker, -1)
    )

    neighbors = sorted(
        broker_neighbor_weights[broker],
        key=lambda x: -x[1]
    )[:BROKER_NEIGHBORS]

    for neighbor, weight in neighbors:
        G_broker.add_node(
            neighbor,
            betweenness=broker_scores.get(neighbor, 0),
            community=node_to_comm.get(neighbor, -1)
        )

        G_broker.add_edge(
            broker,
            neighbor,
            weight=weight
        )

draw_network(
    G_broker,
    "Top Brokers and Their Strongest Network Neighbors",
    OUTPUT_DIR / "broker_network.png",
    node_size_attr="betweenness",
    highlight_nodes=top_brokers,
    community_colors=False
)


print("\nDONE.")
print("Created:")
print(" -", OUTPUT_DIR / "central_actors_network.png")
print(" -", OUTPUT_DIR / "community_network.png")
print(" -", OUTPUT_DIR / "broker_network.png")