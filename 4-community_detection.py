import pandas as pd
import networkx as nx
import community as community_louvain

INPUT = "data/structured_output2/network_outputs_clean/edges_weighted_clean.csv"

OUTPUT = "data/structured_output2/network_outputs_clean/communities.csv"
SUMMARY_OUTPUT = "data/structured_output2/network_outputs_clean/community_summary.csv"

print("Loading network...")
df = pd.read_csv(INPUT)

print("Building graph...")
G = nx.from_pandas_edgelist(
    df,
    source="source",
    target="target",
    edge_attr="weight"
)

print("Running Louvain community detection...")
partition = community_louvain.best_partition(G, weight="weight")

rows = []
for node, comm in partition.items():
    rows.append({"node": node, "community": comm})

out = pd.DataFrame(rows)
out.to_csv(OUTPUT, index=False)

summary = out.groupby("community").size().reset_index(name="node_count")
summary.to_csv(SUMMARY_OUTPUT, index=False)

print("Done.")
print("Communities saved to:", OUTPUT)
print("Summary saved to:", SUMMARY_OUTPUT)