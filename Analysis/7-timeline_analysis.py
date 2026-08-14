import re
import csv
from pathlib import Path
from collections import Counter, defaultdict

import pandas as pd
import matplotlib.pyplot as plt



BASE_DIR = Path("/Users/maysounbrghl/Desktop/Thesis/jfk_data_analysis/data/structured_output2")

DOCS_FILE = BASE_DIR / "docs.csv"
RELATIONS_FILE = BASE_DIR / "relations_canonical_matched.csv"

OUTPUT_DIR = BASE_DIR / "timeline_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_EDGES_BY_YEAR = OUTPUT_DIR / "timeline_edges_by_year.csv"
OUT_ENTITY_ACTIVITY = OUTPUT_DIR / "timeline_entity_activity_by_year.csv"
OUT_TOP_ENTITIES = OUTPUT_DIR / "timeline_top_entities_by_year.csv"
OUT_STATS_BY_YEAR = OUTPUT_DIR / "timeline_stats_by_year.csv"

PLOT_RELATIONS = OUTPUT_DIR / "timeline_relations_by_year.png"
PLOT_ENTITIES = OUTPUT_DIR / "timeline_unique_entities_by_year.png"
PLOT_TOP_YEAR = OUTPUT_DIR / "timeline_top_entities_peak_year.png"


CHUNK_SIZE = 500_000
MIN_YEAR = 1950
MAX_YEAR = 2025
TOP_N_PER_YEAR = 20



def clean_text(value):
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def extract_year_from_text(value):
    """
    Extract the earliest plausible year from the dates field.
    """

    if pd.isna(value):
        return None

    text = str(value)

    years = re.findall(r"\b(19[5-9]\d|20[0-2]\d)\b", text)
    years = [int(y) for y in years]

    years = [
        y for y in years
        if MIN_YEAR <= y <= MAX_YEAR
    ]

    if not years:
        return None

    return min(years)


print("Loading docs file...")

if not DOCS_FILE.exists():
    raise FileNotFoundError(f"Missing docs.csv: {DOCS_FILE}")

docs = pd.read_csv(DOCS_FILE)
docs.columns = [c.strip() for c in docs.columns]

required_doc_cols = {"file", "dates"}
missing = required_doc_cols - set(docs.columns)

if missing:
    raise ValueError(f"docs.csv is missing columns: {missing}")

docs["file"] = docs["file"].apply(clean_text)
docs["year"] = docs["dates"].apply(extract_year_from_text)

file_to_year = (
    docs.dropna(subset=["year"])
    .assign(year=lambda x: x["year"].astype(int))
    .set_index("file")["year"]
    .to_dict()
)

print(f"Documents with usable year: {len(file_to_year):,}")


print("Processing canonical relations...")

if not RELATIONS_FILE.exists():
    raise FileNotFoundError(f"Missing relations file: {RELATIONS_FILE}")

edge_year_counts = Counter()
entity_year_counts = Counter()
year_relation_counts = Counter()
year_unique_edges = defaultdict(set)
year_unique_entities = defaultdict(set)

rows_used = 0
chunks_done = 0

for chunk in pd.read_csv(RELATIONS_FILE, chunksize=CHUNK_SIZE):
    chunks_done += 1
    chunk.columns = [c.strip() for c in chunk.columns]

    required_cols = {"file", "source_text", "target_text", "relation"}
    missing = required_cols - set(chunk.columns)

    if missing:
        raise ValueError(f"relations file is missing columns: {missing}")

    chunk["file"] = chunk["file"].apply(clean_text)
    chunk["source_text"] = chunk["source_text"].apply(clean_text)
    chunk["target_text"] = chunk["target_text"].apply(clean_text)
    chunk["relation"] = chunk["relation"].apply(clean_text)

    chunk["year"] = chunk["file"].map(file_to_year)
    chunk = chunk.dropna(subset=["year"])
    chunk["year"] = chunk["year"].astype(int)

    for row in chunk.itertuples(index=False):
        source = row.source_text
        target = row.target_text
        relation = row.relation
        year = row.year

        if not source or not target:
            continue

        if source == target:
            continue

        if relation == "cooccurs" and source > target:
            source, target = target, source

        edge_key = (year, source, target, relation)

        edge_year_counts[edge_key] += 1
        entity_year_counts[(year, source)] += 1
        entity_year_counts[(year, target)] += 1
        year_relation_counts[year] += 1
        year_unique_edges[year].add((source, target, relation))
        year_unique_entities[year].add(source)
        year_unique_entities[year].add(target)

        rows_used += 1

    print(f"Chunk {chunks_done:,} done | Rows used: {rows_used:,}")



print("Writing timeline_edges_by_year.csv...")

with OUT_EDGES_BY_YEAR.open("w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)

    writer.writerow([
        "year",
        "source",
        "target",
        "relation",
        "weight"
    ])

    for (year, source, target, relation), weight in sorted(
        edge_year_counts.items(),
        key=lambda x: (x[0][0], -x[1])
    ):
        writer.writerow([
            year,
            source,
            target,
            relation,
            weight
        ])



print("Writing timeline_entity_activity_by_year.csv...")

with OUT_ENTITY_ACTIVITY.open("w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)

    writer.writerow([
        "year",
        "entity",
        "activity_count"
    ])

    for (year, entity), count in sorted(
        entity_year_counts.items(),
        key=lambda x: (x[0][0], -x[1])
    ):
        writer.writerow([
            year,
            entity,
            count
        ])


print("Writing timeline_top_entities_by_year.csv...")

top_rows = []

years = sorted({year for year, _ in entity_year_counts.keys()})

for year in years:
    year_entities = [
        (entity, count)
        for (y, entity), count in entity_year_counts.items()
        if y == year
    ]

    top_entities = sorted(
        year_entities,
        key=lambda x: -x[1]
    )[:TOP_N_PER_YEAR]

    for rank, (entity, count) in enumerate(top_entities, start=1):
        top_rows.append({
            "year": year,
            "rank": rank,
            "entity": entity,
            "activity_count": count
        })

top_df = pd.DataFrame(top_rows)
top_df.to_csv(OUT_TOP_ENTITIES, index=False)


print("Writing timeline_stats_by_year.csv...")

stats_rows = []

for year in sorted(year_relation_counts.keys()):
    stats_rows.append({
        "year": year,
        "relation_count": year_relation_counts[year],
        "unique_edges": len(year_unique_edges[year]),
        "unique_entities": len(year_unique_entities[year])
    })

stats_df = pd.DataFrame(stats_rows)
stats_df.to_csv(OUT_STATS_BY_YEAR, index=False)


print("Creating timeline plots...")

if not stats_df.empty:
    # Plot 1: relation count by year
    plt.figure(figsize=(12, 6))
    plt.plot(stats_df["year"], stats_df["relation_count"], marker="o")
    plt.title("Number of Entity Relations by Year")
    plt.xlabel("Year")
    plt.ylabel("Relation count")
    plt.tight_layout()
    plt.savefig(PLOT_RELATIONS, dpi=300)
    plt.close()

    # Plot 2: unique entities by year
    plt.figure(figsize=(12, 6))
    plt.plot(stats_df["year"], stats_df["unique_entities"], marker="o")
    plt.title("Number of Unique Entities by Year")
    plt.xlabel("Year")
    plt.ylabel("Unique entities")
    plt.tight_layout()
    plt.savefig(PLOT_ENTITIES, dpi=300)
    plt.close()

    # Plot 3: top entities in peak year
    if not top_df.empty:
        peak_year = int(
            stats_df.sort_values("relation_count", ascending=False).iloc[0]["year"]
        )

        peak_top = top_df[top_df["year"] == peak_year].head(20)

        plt.figure(figsize=(12, 7))
        plt.barh(
            peak_top["entity"][::-1],
            peak_top["activity_count"][::-1]
        )
        plt.title(f"Top 20 Entities by Activity in {peak_year}")
        plt.xlabel("Activity count")
        plt.tight_layout()
        plt.savefig(PLOT_TOP_YEAR, dpi=300)
        plt.close()


print("\nDONE.")
print(f"Rows used: {rows_used:,}")
print("Created:")
print(" -", OUT_EDGES_BY_YEAR)
print(" -", OUT_ENTITY_ACTIVITY)
print(" -", OUT_TOP_ENTITIES)
print(" -", OUT_STATS_BY_YEAR)
print(" -", PLOT_RELATIONS)
print(" -", PLOT_ENTITIES)
print(" -", PLOT_TOP_YEAR)