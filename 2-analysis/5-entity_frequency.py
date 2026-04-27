import pandas as pd

EDGES_FILE = "edges_weighted_clean.csv"
OUT_FILE_MENTIONS = "entity_frequency_mentions.csv"
OUT_FILE_ALL = "entity_frequency_all_relations.csv"

CHUNKSIZE = 1_000_000  # adjust if needed (500k if memory is tight)

def compute_entity_frequency(edges_file: str, out_file: str, relation_filter: str | None):
    freq = {}  # entity -> frequency (weighted)

    total_rows = 0
    kept_rows = 0

    for chunk in pd.read_csv(edges_file, chunksize=CHUNKSIZE):
        total_rows += len(chunk)

        # Optional: filter to only one relation type (e.g., "mentions")
        if relation_filter is not None:
            chunk = chunk[chunk["relation"].astype(str).str.lower() == relation_filter.lower()]

        kept_rows += len(chunk)
        if len(chunk) == 0:
            continue

        # Ensure weight is numeric; default to 1 if missing/bad
        w = pd.to_numeric(chunk["weight"], errors="coerce").fillna(1.0)

        # Add weights to both endpoints (source and target)
        src = chunk["source"].astype(str).values
        tgt = chunk["target"].astype(str).values
        ww = w.values

        for s, t, weight in zip(src, tgt, ww):
            freq[s] = freq.get(s, 0.0) + float(weight)
            freq[t] = freq.get(t, 0.0) + float(weight)

        print(f"Processed rows: {total_rows:,} | Kept: {kept_rows:,}", end="\r")

    print("\nBuilding output table...")

    out = pd.DataFrame({
        "entity": list(freq.keys()),
        "frequency": list(freq.values())
    }).sort_values("frequency", ascending=False)

    out.to_csv(out_file, index=False)
    print(f"Saved: {out_file}")
    print("\nTop 20 entities:")
    print(out.head(20).to_string(index=False))

# A) Mentions frequency (closest to “who appears most often?”)
compute_entity_frequency(EDGES_FILE, OUT_FILE_MENTIONS, relation_filter="mentions")

# B) Overall visibility across ALL relations (connectivity visibility)
compute_entity_frequency(EDGES_FILE, OUT_FILE_ALL, relation_filter=None)