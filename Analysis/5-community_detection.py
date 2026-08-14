from pathlib import Path
import pandas as pd
import networkx as nx
import community as community_louvain

BASE_DIR = Path("/Users/maysounbrghl/Desktop/Thesis/jfk_data_analysis/data/structured_output2")
NETWORK_DIR = BASE_DIR / "network_output"

INPUT_FILE = NETWORK_DIR / "edges_weighted.csv"

OUTPUT_FILE = NETWORK_DIR / "communities.csv"
SUMMARY_FILE = NETWORK_DIR / "community_summary.csv"

print("Loading weighted network edges...")
print("Input:", INPUT_FILE)

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Could not find file: {INPUT_FILE}")

df = pd.read_csv(INPUT_FILE)

# BUILD GRAPH

print("Building graph...")

G = nx.from_pandas_edgelist(
    df,
    source="source",
    target="target",
    edge_attr="weight",
    create_using=nx.Graph()
)

print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())


# COMMUNITY DETECTION

print("Running Louvain...")

partition = community_louvain.best_partition(G, weight="weight", random_state=42)

communities = pd.DataFrame(
    [{"node": node, "community": comm} for node, comm in partition.items()]
)

communities.to_csv(OUTPUT_FILE, index=False)

summary = (
    communities.groupby("community")
    .size()
    .reset_index(name="node_count")
    .sort_values("node_count", ascending=False)
)

summary.to_csv(SUMMARY_FILE, index=False)

print("\nDONE.")
print("Saved:", OUTPUT_FILE)
print("Saved:", SUMMARY_FILE)