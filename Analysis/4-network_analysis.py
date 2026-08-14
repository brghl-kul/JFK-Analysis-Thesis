import csv
from pathlib import Path
from collections import defaultdict, Counter

import pandas as pd
import matplotlib.pyplot as plt


BASE_DIR = Path("/Users/maysounbrghl/Desktop/Thesis/jfk_data_analysis/data/structured_output2")

INPUT_FILE = BASE_DIR / "relations_canonical_matched.csv"

OUTPUT_DIR = BASE_DIR / "network_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EDGES_FILE = OUTPUT_DIR / "edges_weighted.csv"
NODES_FILE = OUTPUT_DIR / "nodes_weighted_degree.csv"
PLOT_FILE = OUTPUT_DIR / "top20_nodes_weighted_degree.png"
STATS_FILE = OUTPUT_DIR / "network_stats.txt"


BAD_TERMS = {
    "subject", "copy", "memorandum", "memo", "name", "agency",
    "headquarters", "officer", "file", "date", "from", "to",
    "re", "wh", "dc", "ci", "mar", "dec"
}


def clean_entity(entity):
    if not entity:
        return None

    entity = str(entity).strip()

    if len(entity) < 3:
        return None

    if entity.lower() in BAD_TERMS:
        return None

    if entity.isupper() and len(entity) > 25:
        return None

    return entity


# BUILD WEIGHTED EDGES

print("Reading relations file...")
print("Input:", INPUT_FILE)

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Could not find relations file: {INPUT_FILE}")

edge_weights = defaultdict(int)
node_weighted_degree = Counter()
node_relation_counts = Counter()

rows_read = 0
rows_used = 0

with INPUT_FILE.open("r", encoding="utf-8", errors="ignore") as f:
    reader = csv.DictReader(f)

    required_cols = {"source_text", "target_text", "source_type", "target_type", "relation"}
    missing = required_cols - set(reader.fieldnames)

    if missing:
        raise ValueError(f"Missing required columns in relations.csv: {missing}")

    for row in reader:
        rows_read += 1

        source = clean_entity(row["source_text"])
        target = clean_entity(row["target_text"])

        if not source or not target:
            continue

        if source == target:
            continue

        source_type = row["source_type"].strip()
        target_type = row["target_type"].strip()
        relation = row["relation"].strip()

        # Undirected normalization for co-occurrence relations
        if relation == "cooccurs":
            if source > target:
                source, target = target, source
                source_type, target_type = target_type, source_type

        edge_key = (
            source,
            target,
            source_type,
            target_type,
            relation
        )

        edge_weights[edge_key] += 1

        node_weighted_degree[source] += 1
        node_weighted_degree[target] += 1

        node_relation_counts[source] += 1
        node_relation_counts[target] += 1

        rows_used += 1

        if rows_read % 1_000_000 == 0:
            print(f"Processed {rows_read:,} rows...")



print("Writing weighted edge file...")

with EDGES_FILE.open("w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)

    writer.writerow([
        "source",
        "target",
        "source_type",
        "target_type",
        "relation",
        "weight"
    ])

    for (source, target, source_type, target_type, relation), weight in sorted(
        edge_weights.items(),
        key=lambda x: -x[1]
    ):
        writer.writerow([
            source,
            target,
            source_type,
            target_type,
            relation,
            weight
        ])


print("Writing node weighted degree file...")

with NODES_FILE.open("w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)

    writer.writerow([
        "node",
        "weighted_degree"
    ])

    for node, degree in node_weighted_degree.most_common():
        writer.writerow([
            node,
            degree
        ])

print("Creating Top 20 weighted-degree plot...")

top20 = node_weighted_degree.most_common(20)

if top20:
    names = [item[0] for item in top20]
    values = [item[1] for item in top20]

    plt.figure(figsize=(12, 7))
    plt.barh(names[::-1], values[::-1])
    plt.title("Top 20 Entities by Weighted Degree")
    plt.xlabel("Weighted degree")
    plt.tight_layout()
    plt.savefig(PLOT_FILE, dpi=300)
    plt.close()


print("Writing network stats...")

with STATS_FILE.open("w", encoding="utf-8") as f:
    f.write("Network Construction Summary\n")
    f.write("============================\n\n")

    f.write(f"Input file: {INPUT_FILE}\n")
    f.write(f"Rows read from relations.csv: {rows_read:,}\n")
    f.write(f"Rows used after cleaning: {rows_used:,}\n\n")

    f.write(f"Total unique nodes: {len(node_weighted_degree):,}\n")
    f.write(f"Total unique weighted edges: {len(edge_weights):,}\n\n")

    f.write("Top 20 nodes by weighted degree:\n")
    for node, degree in top20:
        f.write(f"{node}: {degree:,}\n")

print("\nDONE.")
print("Created:")
print("Edges:", EDGES_FILE)
print("Nodes:", NODES_FILE)
print("Plot:", PLOT_FILE)
print("Stats:", STATS_FILE)