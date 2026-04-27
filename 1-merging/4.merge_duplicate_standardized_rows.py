from pathlib import Path
import re
import pandas as pd


BASE_DIR = Path("/Users/maysounbrghl/Desktop/Thesis/jfk_data_analysis")
INPUT_FILE = BASE_DIR / "data/Merge/merged_entities_standardized_cleaned.xlsx"
OUTPUT_FILE = BASE_DIR / "data/Merge/merged_entities_standardized_deduplicated.xlsx"

# Helpers

def clean_text(x):
    if pd.isna(x):
        return ""
    return re.sub(r"\s+", " ", str(x)).strip()


def normalize_key(x):
    return clean_text(x).lower()


def prefer_h_value(group: pd.DataFrame, column: str) -> str:
    # Prefer Hanqing / Both
    for _, row in group.iterrows():
        status = clean_text(row.get("Status", ""))
        value = clean_text(row.get(column, ""))
        if status in ["Hanqing", "Both"] and value:
            return value

    # Fallback: any non-empty
    for _, row in group.iterrows():
        value = clean_text(row.get(column, ""))
        if value:
            return value

    return ""


def choose_status(group: pd.DataFrame) -> str:
    statuses = {clean_text(x) for x in group["Status"].tolist()}

    if "Both" in statuses:
        return "Both"
    if "Hanqing" in statuses:
        return "Hanqing"
    return "Maysoun"


def merge_one_group(group: pd.DataFrame, original_columns: list[str]) -> dict:
    # Case 1: only one row -> keep unchanged
    if len(group) == 1:
        row = group.iloc[0].copy()

        freq_h = int(pd.to_numeric(pd.Series([row.get("frequency_H", 0)]), errors="coerce").fillna(0).iloc[0])
        freq_m = int(pd.to_numeric(pd.Series([row.get("frequency_M", 0)]), errors="coerce").fillna(0).iloc[0])

        row["frequency_H"] = freq_h
        row["frequency_M"] = freq_m
        row["Frequency"] = max(freq_h, freq_m)

        return {col: row[col] if col in row.index else "" for col in original_columns}

    # Case 2: multiple rows -> merge
    result = {}

    result["frequency_H"] = int(pd.to_numeric(group["frequency_H"], errors="coerce").fillna(0).sum())
    result["frequency_M"] = int(pd.to_numeric(group["frequency_M"], errors="coerce").fillna(0).sum())
    result["Frequency"] = max(result["frequency_H"], result["frequency_M"])

    for col in original_columns:
        if col in ["frequency_H", "frequency_M", "Frequency"]:
            continue
        result[col] = prefer_h_value(group, col)

    if "Status" in original_columns:
        result["Status"] = choose_status(group)

    return result

# Load

df = pd.read_excel(INPUT_FILE)
df.columns = [c.strip() for c in df.columns]

required_cols = ["source_files", "canonical_name", "frequency_H", "frequency_M"]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in {INPUT_FILE}")

# Clean object columns
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].apply(clean_text)

# Ensure optional columns exist
optional_cols = [
    "file_Id",
    "Id",
    "Type",
    "canonical_source",
    "canonical_file_Id",
    "Description",
    "description_source",
    "Status",
]
for col in optional_cols:
    if col not in df.columns:
        df[col] = ""

# Numeric columns
df["frequency_H"] = pd.to_numeric(df["frequency_H"], errors="coerce").fillna(0).astype(int)
df["frequency_M"] = pd.to_numeric(df["frequency_M"], errors="coerce").fillna(0).astype(int)
df["Frequency"] = df[["frequency_H", "frequency_M"]].max(axis=1)

original_columns = df.columns.tolist()

# Build grouping keys
df["source_files_key"] = df["source_files"].apply(normalize_key)
df["canonical_name_key"] = df["canonical_name"].apply(normalize_key)

# Fallback: if canonical_name empty, use Id
if "Id" in df.columns:
    df.loc[df["canonical_name_key"] == "", "canonical_name_key"] = df.loc[
        df["canonical_name_key"] == "", "Id"
    ].apply(normalize_key)

print(f"Original rows: {len(df)}")

group_cols = ["source_files_key", "canonical_name_key"]

# Merge groups safely

merged_rows = []
for _, group in df.groupby(group_cols, dropna=False, sort=False):
    merged_rows.append(merge_one_group(group, original_columns))

merged_df = pd.DataFrame(merged_rows)

print(f"Merged rows: {len(merged_df)}")
print(f"Removed duplicates: {len(df) - len(merged_df)}")

# Optional sort
sort_cols = [c for c in ["source_files", "canonical_name", "Id"] if c in merged_df.columns]
if sort_cols:
    merged_df = merged_df.sort_values(sort_cols).reset_index(drop=True)

# Save
merged_df.to_excel(OUTPUT_FILE, index=False)

print(f"\nSaved deduplicated file to:\n{OUTPUT_FILE}")