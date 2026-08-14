from pathlib import Path
import pandas as pd
import re


BASE_DIR = Path("/Users/maysounbrghl/Desktop/Thesis/jfk_data_analysis")

H_FILE = BASE_DIR / "data/Merge/entities_H.xlsx"
M_FILE = BASE_DIR / "data/Merge/entities_M_cleaned.xlsx"

OUTPUT_FILE = BASE_DIR / "data/Merge/merged_entities.xlsx"


def clean_text(x):
    if pd.isna(x):
        return ""
    return re.sub(r"\s+", " ", str(x)).strip()


def normalize_id_for_match(x):
    """
    Normalize Id for normal matching:
    - trim spaces
    - collapse spaces
    - lowercase
    """
    return clean_text(x).lower()


def has_any_letter(text: str) -> bool:
    return bool(re.search(r"[A-Za-z]", clean_text(text)))


def is_non_text_id(text: str) -> bool:
    """
    True if Id has no letters and is only numbers/symbols/both.
    Examples:
      123
      !!!
      12/7
      --
    """
    text = clean_text(text)
    if text == "":
        return True
    return not has_any_letter(text)


def make_merge_key(source_file, entity_id):
    """
    Merge rule:
    - Normal case: source_files + normalized Id
    - If Id has no letters (only numbers/symbols/both):
      use source_files only, so it can still align and later keep H Id
    """
    source_key = clean_text(source_file).lower()
    entity_id_clean = clean_text(entity_id)

    if is_non_text_id(entity_id_clean):
        return f"{source_key}|||NON_TEXT_ID"

    return f"{source_key}|||{normalize_id_for_match(entity_id_clean)}"


def choose_final_id(id_h, id_m):
    """
    Final Id rule:
    - prefer entities_H Id when available
    - otherwise use entities_M Id
    """
    id_h = clean_text(id_h)
    id_m = clean_text(id_m)

    if id_h:
        return id_h
    return id_m


def choose_final_value(v_h, v_m):
    """
    Prefer H when available, otherwise M.
    """
    v_h = clean_text(v_h)
    v_m = clean_text(v_m)

    if v_h:
        return v_h
    return v_m


print("Loading files...")

entities_H = pd.read_excel(H_FILE)
entities_M = pd.read_excel(M_FILE)

entities_H.columns = [c.strip() for c in entities_H.columns]
entities_M.columns = [c.strip() for c in entities_M.columns]

print("Columns in entities_H:", list(entities_H.columns))
print("Columns in entities_M:", list(entities_M.columns))


required_cols = ["source_files", "Id", "Description"]

for col in required_cols:
    if col not in entities_H.columns:
        raise ValueError(f"Column '{col}' not found in entities_H.xlsx")
    if col not in entities_M.columns:
        raise ValueError(f"Column '{col}' not found in entities_M_cleaned.xlsx")


# Clean columns

for df in [entities_H, entities_M]:
    df["source_files"] = df["source_files"].apply(clean_text)
    df["Id"] = df["Id"].apply(clean_text)
    df["Description"] = df["Description"].fillna("").apply(clean_text)

    if "Type" not in df.columns:
        df["Type"] = ""
    df["Type"] = df["Type"].fillna("").apply(clean_text)

    if "frequency" not in df.columns:
        df["frequency"] = 0
    df["frequency"] = pd.to_numeric(df["frequency"], errors="coerce").fillna(0).astype(int)



# Build merge keys

entities_H["merge_key"] = entities_H.apply(
    lambda r: make_merge_key(r["source_files"], r["Id"]),
    axis=1
)

entities_M["merge_key"] = entities_M.apply(
    lambda r: make_merge_key(r["source_files"], r["Id"]),
    axis=1
)

print("\nDuplicate merge keys in H:", entities_H["merge_key"].duplicated().sum())
print("Duplicate merge keys in M:", entities_M["merge_key"].duplicated().sum())


# Rename columns before merge

entities_H = entities_H.rename(columns={
    "source_files": "source_files_H",
    "Id": "Id_H",
    "Type": "Type_H",
    "frequency": "frequency_H",
    "Description": "Description_H",
})

entities_M = entities_M.rename(columns={
    "source_files": "source_files_M",
    "Id": "Id_M",
    "Type": "Type_M",
    "frequency": "frequency_M",
    "Description": "Description_M",
})


# Merge

merged = pd.merge(
    entities_H,
    entities_M,
    on="merge_key",
    how="outer",
    indicator=True
)

print("\nMerge summary:")
print(merged["_merge"].value_counts())


# Status

merged["Status"] = merged["_merge"].map({
    "both": "Both",
    "left_only": "Hanqing",
    "right_only": "Maysoun",
})


# Final shared columns
# Always prefer H values when present
merged["source_files"] = merged.apply(
    lambda r: choose_final_value(r.get("source_files_H", ""), r.get("source_files_M", "")),
    axis=1
)

merged["Id"] = merged.apply(
    lambda r: choose_final_id(r.get("Id_H", ""), r.get("Id_M", "")),
    axis=1
)

merged["Type"] = merged.apply(
    lambda r: choose_final_value(r.get("Type_H", ""), r.get("Type_M", "")),
    axis=1
)

# Recreate file_Id if you want it in output
merged["file_Id"] = merged.apply(
    lambda r: f"{clean_text(r['source_files'])}{clean_text(r['Id'])}",
    axis=1
)

# Frequencies

merged["frequency_H"] = pd.to_numeric(merged["frequency_H"], errors="coerce").fillna(0).astype(int)
merged["frequency_M"] = pd.to_numeric(merged["frequency_M"], errors="coerce").fillna(0).astype(int)

merged["Frequency"] = merged[["frequency_H", "frequency_M"]].max(axis=1)

# Description rule
# Both + Hanqing -> use H description
# Maysoun -> use M description
merged["Description"] = merged.apply(
    lambda r: clean_text(r.get("Description_H", ""))
    if r["Status"] in ["Both", "Hanqing"]
    else clean_text(r.get("Description_M", "")),
    axis=1
)


# Final output columns

final_cols = [
    "file_Id",
    "source_files",
    "Id",
    "Type",
    "frequency_H",
    "Description",
    "frequency_M",
    "Frequency",
    "Status",
]

merged_final = merged[final_cols].copy()

# Optional sort
merged_final = merged_final.sort_values(["source_files", "Id"]).reset_index(drop=True)

# Save

merged_final.to_excel(OUTPUT_FILE, index=False)

print(f"\nSaved merged file to:\n{OUTPUT_FILE}")
print("\nFinal status counts:")
print(merged_final["Status"].value_counts())