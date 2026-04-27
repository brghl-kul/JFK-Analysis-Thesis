from pathlib import Path
import re
import pandas as pd


BASE_DIR = Path("/Users/maysounbrghl/Desktop/Thesis/jfk_data_analysis")
INPUT_FILE = BASE_DIR / "data/Merge/entities_M.xlsx"
CLEANED_FILE = BASE_DIR / "data/Merge/entities_M_cleaned.xlsx"
REMOVED_FILE = BASE_DIR / "data/Merge/entities_M_removed_rows.xlsx"
MEANINGLESS_FILE = BASE_DIR / "data/Merge/entities_M_removed_meaningless_ids.xlsx"


MEANINGLESS_IDS = {
    "NAME",
    "STATE",
    "STATION",
    "OFFICE",
    "REPORT",
    "FILE",
    "DOCUMENT",
    "RECORD",
    "DATA",
    "PAGE",
    "COPY",
    "NUMBER",
    "ADDRESS",
    "DATE",
    "TIME",
    "PLACE",
    "AREA",
    "LOCATION",
    "GROUP",
    "UNIT",
    "SECTION",
    "SYSTEM",
    "PROGRAM",
    "PROJECT",
    "CONTROL",
    "CASE",
    "FORM",
    "ENTITY",
    "PERSON",
    "ORGANIZATION",
    "TITLE",
    "SUBJECT",
    "REFERENCE",
    "INFO",
    "INFORMATION",
    "ITEM",
    "ENTRY",
}


def clean_text(x):
    if pd.isna(x):
        return ""
    return re.sub(r"\s+", " ", str(x)).strip()


def remove_unwanted_symbols(text: str) -> str:
    """
    Remove all symbols except letters, digits, spaces, '/' and '.'
    """
    text = re.sub(r"[^A-Za-z0-9\s\/\.]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def remove_leading_numbers(text: str) -> str:
    """
    Remove numbers at the start, including separators after them.
    Examples:
      '123CIA' -> 'CIA'
      '45 Oswald' -> 'Oswald'
      '12/ CIA' -> 'CIA'
      '7. CIA' -> 'CIA'
    """
    text = re.sub(r"^\s*\d+[\s\/\.\-,:;]*", "", text)
    return text.strip()


def remove_leading_slash_dot(text: str) -> str:
    """
    Remove '/' or '.' only if they appear at the beginning.
    Keep them elsewhere.
    Examples:
      '/CIA' -> 'CIA'
      '.CIA' -> 'CIA'
      'A/B' -> 'A/B'
      'J.F.K.' -> 'J.F.K.'
    """
    text = re.sub(r"^[\/\.]+", "", text)
    return text.strip()


def normalize_for_check(text: str) -> str:
    return clean_text(text).upper()


def has_any_letter(text: str) -> bool:
    return bool(re.search(r"[A-Za-z]", text))


def is_only_numbers_and_symbols(text: str) -> bool:
    """
    True if text contains no letters after cleaning.
    """
    if text == "":
        return True
    return not has_any_letter(text)


def looks_meaningless(text: str) -> bool:
    return normalize_for_check(text) in MEANINGLESS_IDS


def classify_id(original_id: str) -> tuple[str, str]:
    """
    Returns:
      cleaned_id, action

    action:
      - keep
      - remove_numbers_or_symbols_only
      - remove_meaningless_id
    """
    text = clean_text(original_id)

    # 1) remove all unwanted symbols, keep only / and .
    text = remove_unwanted_symbols(text)

    # 2) remove leading numbers
    text = remove_leading_numbers(text)

    # 3) remove / or . only at the beginning
    text = remove_leading_slash_dot(text)

    # 4) normalize spaces
    text = clean_text(text)

    # remove rows with no letters
    if is_only_numbers_and_symbols(text):
        return text, "remove_numbers_or_symbols_only"

    # remove meaningless rows to separate file
    if looks_meaningless(text):
        return text, "remove_meaningless_id"

    return text, "keep"


df = pd.read_excel(INPUT_FILE)
df.columns = [c.strip() for c in df.columns]

if "Id" not in df.columns:
    raise ValueError("Column 'Id' not found in entities_M.xlsx")

# preserve original
df["Id_original"] = df["Id"].apply(clean_text)

results = df["Id"].apply(classify_id)
df["Id_cleaned"] = results.apply(lambda x: x[0])
df["action"] = results.apply(lambda x: x[1])

# replace Id with cleaned Id
df["Id"] = df["Id_cleaned"]

cleaned_df = df[df["action"] == "keep"].copy()
removed_df = df[df["action"] == "remove_numbers_or_symbols_only"].copy()
meaningless_df = df[df["action"] == "remove_meaningless_id"].copy()

removed_df["removal_reason"] = "Id became only numbers/symbols or had no letters after cleaning."
meaningless_df["removal_reason"] = "Id is generic/meaningless."

# drop helper column
cleaned_df = cleaned_df.drop(columns=["Id_cleaned"])
removed_df = removed_df.drop(columns=["Id_cleaned"])
meaningless_df = meaningless_df.drop(columns=["Id_cleaned"])

# sort
clean_sort_cols = [c for c in ["source_files", "Id"] if c in cleaned_df.columns]
if clean_sort_cols:
    cleaned_df = cleaned_df.sort_values(clean_sort_cols).reset_index(drop=True)

removed_sort_cols = [c for c in ["source_files", "Id_original"] if c in removed_df.columns]
if removed_sort_cols:
    removed_df = removed_df.sort_values(removed_sort_cols).reset_index(drop=True)

meaningless_sort_cols = [c for c in ["source_files", "Id_original"] if c in meaningless_df.columns]
if meaningless_sort_cols:
    meaningless_df = meaningless_df.sort_values(meaningless_sort_cols).reset_index(drop=True)

cleaned_df.to_excel(CLEANED_FILE, index=False)
removed_df.to_excel(REMOVED_FILE, index=False)
meaningless_df.to_excel(MEANINGLESS_FILE, index=False)

print(f"Original rows: {len(df)}")
print(f"Kept rows: {len(cleaned_df)}")
print(f"Removed numbers/symbols-only rows: {len(removed_df)}")
print(f"Removed meaningless rows: {len(meaningless_df)}")

print(f"\nSaved cleaned file to:\n{CLEANED_FILE}")
print(f"\nSaved removed rows to:\n{REMOVED_FILE}")
print(f"\nSaved meaningless rows to:\n{MEANINGLESS_FILE}")

print("\nAction summary:")
print(df["action"].value_counts(dropna=False))