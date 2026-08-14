from pathlib import Path
import pandas as pd
import re


BASE_DIR = Path("/Users/maysounbrghl/Desktop/Thesis/jfk_data_analysis/data/structured_output2")

ENTITIES_FILE = BASE_DIR / "merged_entities_final.csv"
RELATIONS_FILE = BASE_DIR / "relations.csv"

OUTPUT_MATCHED = BASE_DIR / "relations_canonical_matched.csv"
OUTPUT_REMOVED = BASE_DIR / "relations_removed_unmatched.csv"

CHUNK_SIZE = 500_000


def clean_text(x):
    if pd.isna(x):
        return ""
    return re.sub(r"\s+", " ", str(x)).strip()


def norm_key(x):
    return clean_text(x).lower()


print("Loading approved entities...")
entities = pd.read_csv(ENTITIES_FILE)
entities.columns = [c.strip() for c in entities.columns]

if "Id" not in entities.columns:
    raise ValueError("Column 'Id' not found in merged_entities_final.csv")

if "canonical_name" not in entities.columns:
    raise ValueError("Column 'canonical_name' not found in merged_entities_final.csv")

entities["Id"] = entities["Id"].apply(clean_text)
entities["canonical_name"] = entities["canonical_name"].fillna("").apply(clean_text)

entities.loc[entities["canonical_name"] == "", "canonical_name"] = entities.loc[
    entities["canonical_name"] == "", "Id"
]

entities["Id_key"] = entities["Id"].apply(norm_key)
entities = entities[entities["Id_key"] != ""].copy()

id_to_canonical = (
    entities.drop_duplicates(subset=["Id_key"], keep="first")
    .set_index("Id_key")["canonical_name"]
    .to_dict()
)

approved_ids = set(id_to_canonical.keys())

print(f"Approved entities loaded: {len(approved_ids):,}")


# remove old output files if they exist
for path in [OUTPUT_MATCHED, OUTPUT_REMOVED]:
    if path.exists():
        path.unlink()


total_rows = 0
total_kept = 0
total_removed = 0
chunk_number = 0

print("Processing relations in chunks...")

for chunk in pd.read_csv(RELATIONS_FILE, chunksize=CHUNK_SIZE):
    chunk_number += 1

    chunk.columns = [c.strip() for c in chunk.columns]

    required = ["source_text", "target_text"]
    for col in required:
        if col not in chunk.columns:
            raise ValueError(f"Column '{col}' not found in relations.csv")

    chunk["source_text"] = chunk["source_text"].apply(clean_text)
    chunk["target_text"] = chunk["target_text"].apply(clean_text)

    chunk["source_text_original"] = chunk["source_text"]
    chunk["target_text_original"] = chunk["target_text"]

    chunk["source_key"] = chunk["source_text"].apply(norm_key)
    chunk["target_key"] = chunk["target_text"].apply(norm_key)

    chunk["source_matched"] = chunk["source_key"].isin(approved_ids)
    chunk["target_matched"] = chunk["target_key"].isin(approved_ids)

    keep_mask = chunk["source_matched"] & chunk["target_matched"]

    matched = chunk[keep_mask].copy()
    removed = chunk[~keep_mask].copy()

    matched["source_text"] = matched["source_key"].map(id_to_canonical)
    matched["target_text"] = matched["target_key"].map(id_to_canonical)

    def removal_reason(row):
        if not row["source_matched"] and not row["target_matched"]:
            return "source_text_and_target_text_not_found_in_approved_entities"
        if not row["source_matched"]:
            return "source_text_not_found_in_approved_entities"
        if not row["target_matched"]:
            return "target_text_not_found_in_approved_entities"
        return ""

    if len(removed) > 0:
        removed["removal_reason"] = removed.apply(removal_reason, axis=1)
    else:
        removed["removal_reason"] = []

    helper_cols = ["source_key", "target_key", "source_matched", "target_matched"]

    matched = matched.drop(columns=[c for c in helper_cols if c in matched.columns])
    removed = removed.drop(columns=[c for c in helper_cols if c in removed.columns])

    matched.to_csv(
        OUTPUT_MATCHED,
        index=False,
        mode="a",
        header=not OUTPUT_MATCHED.exists()
    )

    removed.to_csv(
        OUTPUT_REMOVED,
        index=False,
        mode="a",
        header=not OUTPUT_REMOVED.exists()
    )

    total_rows += len(chunk)
    total_kept += len(matched)
    total_removed += len(removed)

    print(
        f"Chunk {chunk_number:,} done | "
        f"Rows processed: {total_rows:,} | "
        f"Kept: {total_kept:,} | "
        f"Removed: {total_removed:,}"
    )


print("\nDone.")
print(f"Total rows processed: {total_rows:,}")
print(f"Total rows kept: {total_kept:,}")
print(f"Total rows removed: {total_removed:,}")

print(f"\nSaved matched canonical relations to:\n{OUTPUT_MATCHED}")
print(f"\nSaved removed unmatched relations to:\n{OUTPUT_REMOVED}")