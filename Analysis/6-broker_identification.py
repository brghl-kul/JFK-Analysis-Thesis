from pathlib import Path
import pandas as pd
import networkx as nx

BASE_DIR = Path("/Users/maysounbrghl/Desktop/Thesis/jfk_data_analysis/data/structured_output2")
NETWORK_DIR = BASE_DIR / "network_output"

EDGES_FILE = NETWORK_DIR / "edges_weighted.csv"
COMM_FILE = NETWORK_DIR / "communities.csv"

OUT_BET = NETWORK_DIR / "betweenness_approx.csv"
OUT_BROKERS = NETWORK_DIR / "hidden_brokers.csv"


TOP_N_NODES = 40000
MIN_WEIGHT = 2
RELATION_FILTER = "mentions"
K_SAMPLES = 2000
SEED = 42


print("Loading edges...")

if not EDGES_FILE.exists():
    raise FileNotFoundError(f"Missing file: {EDGES_FILE}")

edges = pd.read_csv(EDGES_FILE)


print("Filtering edges...")

edges = edges[edges["relation"].str.lower() == RELATION_FILTER]

edges["weight"] = pd.to_numeric(edges["weight"], errors="coerce").fillna(1.0)

edges = edges[edges["weight"] >= MIN_WEIGHT]


print("Building graph...")

G = nx.Graph()

for row in edges.itertuples(index=False):
    u = str(row.source)
    v = str(row.target)
    w = float(row.weight)

    if u == v:
        continue

    if G.has_edge(u, v):
        G[u][v]["weight"] += w
    else:
        G.add_edge(u, v, weight=w)

print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())

# CONVERT WEIGHTS TO DISTANCE

for u, v, d in G.edges(data=True):
    w = d.get("weight", 1.0)
    d["length"] = 1.0 / w

# COMPUTE BETWEENNESS 

print("Computing approximate betweenness...")

bet = nx.betweenness_centrality(
    G,
    k=K_SAMPLES,
    normalized=True,
    weight="length",
    seed=SEED
)

bet_df = pd.DataFrame({
    "entity": list(bet.keys()),
    "betweenness": list(bet.values())
}).sort_values("betweenness", ascending=False)

bet_df.to_csv(OUT_BET, index=False)

print("Saved:", OUT_BET)


print("Loading communities...")

comm = pd.read_csv(COMM_FILE)
node_to_comm = dict(zip(comm["node"].astype(str), comm["community"]))

# IDENTIFY BROKERS

print("Identifying brokers...")

bet_df["community"] = bet_df["entity"].map(
    lambda x: node_to_comm.get(str(x), None)
)

bet_df.to_csv(OUT_BROKERS, index=False)

print("Saved:", OUT_BROKERS)


print("\nTop 20 brokers:")
print(bet_df.head(20))