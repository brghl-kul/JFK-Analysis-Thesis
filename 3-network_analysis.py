import csv
from collections import defaultdict, Counter
import pandas as pd
import matplotlib.pyplot as plt
import os

# PATHS
INPUT_FILE = "jfk_data_analysis/data/structured_output2/relations.csv"
OUTPUT_FILE = "jfk_data_analysis/data/structured_output2/edges_weighted_clean.csv"
PLOT_FILE = "jfk_data_analysis/data/structured_output2/top20_nodes_weighted_degree_clean.png"
STATS_FILE = "jfk_data_analysis/data/structured_output2/network_stats_clean.txt"

# Words to REMOVE (metadata, not real entities)
BAD_TERMS = {
    "subject", "copy", "memorandum", "memo", "name", "agency",
    "headquarters", "officer", "file", "date", "from", "to",
    "re", "wh", "dc", "ci", "mar", "dec"
}

def clean_entity(e):
    if not e:
        return None

    e = e.strip()

    if len(e) < 3:
        return None

    if e.lower() in BAD_TERMS:
        return None

    if e.isupper() and len(e) > 20:
        return None

    return e


print("Reading relations and building weighted network...")

edge_weights = defaultdict(int)
node_degree = Counter()

count = 0

with open(INPUT_FILE, "r", encoding="utf-8", errors="ignore") as f:
    reader = csv.DictReader(f)

    for row in reader:

        s = clean_entity(row["source_text"])
        t = clean_entity(row["target_text"])
        rel = row["relation"]

        if not s or not t:
            continue

        if s == t:
            continue

        # normalize order for cooccurs
        if rel == "cooccurs" and s > t:
            s, t = t, s

        edge_weights[(s, t, rel)] += 1

        node_degree[s] += 1
        node_degree[t] += 1

        count += 1

        if count % 5000000 == 0:
            print(f"Processed {count:,} rows...")

print("Writing weighted edges...")

with open(OUTPUT_FILE, "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["source", "target", "relation", "weight"])

    for (s, t, rel), w in sorted(edge_weights.items(), key=lambda x: -x[1]):
        writer.writerow([s, t, rel, w])

print("Creating Top 20 plot...")

top20 = node_degree.most_common(20)

names = [x[0] for x in top20]
values = [x[1] for x in top20]

plt.figure(figsize=(10,6))
plt.barh(names[::-1], values[::-1])
plt.title("Top 20 Nodes (Cleaned Network)")
plt.xlabel("Weighted degree")
plt.tight_layout()
plt.savefig(PLOT_FILE)
plt.close()

print("Writing network stats...")

with open(STATS_FILE, "w") as f:
    f.write(f"Total nodes: {len(node_degree)}\n")
    f.write(f"Total edges: {len(edge_weights)}\n")

print("\nDONE.")
print("Created:")
print(OUTPUT_FILE)
print(PLOT_FILE)
print(STATS_FILE)