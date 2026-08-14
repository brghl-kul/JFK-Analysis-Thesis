import pandas as pd
from pathlib import Path
from rapidfuzz import process, fuzz
import re


BASE_FILE = Path("/Users/maysounbrghl/Desktop/Thesis/jfk_data_analysis/nodes_LLM.xlsx")
COMPARE_FILE = Path("/Users/maysounbrghl/Desktop/Thesis/jfk_data_analysis/nodes_Spacy.xlsx")

OUTPUT_DIR = Path("/Users/maysounbrghl/Desktop/Thesis/jfk_data_analysis/comparison_output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "comparison_report.xlsx"

FUZZY_THRESHOLD = 88


def normalize_id(text):
    """
    Normalize Id values:
    - lowercase
    - remove punctuation
    - remove extra spaces
    - remove 'the', 'a', 'an' only at beginning
    """

    text = str(text).strip().lower()

    text = re.sub(r"[^\w\s]", " ", text)

    text = " ".join(text.split())

    for word in ["the ", "a ", "an "]:
        if text.startswith(word):
            text = text[len(word):]

    return text


def load_file(path):
    df = pd.read_excel(path)

    cols_lower = [str(c).strip().lower() for c in df.columns]

    if "id" in cols_lower:
        id_col = df.columns[cols_lower.index("id")]
    else:
        id_col = df.columns[0]

    if "frequency" in cols_lower:
        freq_col = df.columns[cols_lower.index("frequency")]
    else:
        freq_col = df.columns[2]

    df = df[[id_col, freq_col]].copy()
    df.columns = ["Id", "frequency"]

    df["Id"] = df["Id"].astype(str).str.strip()
    df["frequency"] = pd.to_numeric(df["frequency"], errors="coerce").fillna(0).astype(int)

    df = df[df["Id"] != ""].copy()
    return df


base_df = load_file(BASE_FILE)
compare_df = load_file(COMPARE_FILE)

base_df = base_df.groupby("Id", as_index=False)["frequency"].sum()
compare_df = compare_df.groupby("Id", as_index=False)["frequency"].sum()

base_df["Id_norm"] = base_df["Id"].apply(normalize_id)
compare_df["Id_norm"] = compare_df["Id"].apply(normalize_id)

base_grouped = base_df.groupby("Id_norm", as_index=False).agg(
    {"Id": "first", "frequency": "sum"}
)

compare_grouped = compare_df.groupby("Id_norm", as_index=False).agg(
    {"Id": "first", "frequency": "sum"}
)

base_lookup = dict(zip(base_grouped["Id_norm"], base_grouped["Id"]))
compare_lookup = dict(zip(compare_grouped["Id_norm"], compare_grouped["Id"]))

base_freq_lookup = dict(zip(base_grouped["Id_norm"], base_grouped["frequency"]))
compare_freq_lookup = dict(zip(compare_grouped["Id_norm"], compare_grouped["frequency"]))


base_ids = set(base_grouped["Id_norm"])
compare_ids = set(compare_grouped["Id_norm"])

exact_common_ids = base_ids & compare_ids
only_in_base = base_ids - compare_ids
only_in_compare = compare_ids - base_ids


exact_jaccard = len(exact_common_ids) / len(base_ids | compare_ids) if (base_ids | compare_ids) else 0
exact_recall = len(exact_common_ids) / len(base_ids) if base_ids else 0
exact_precision = len(exact_common_ids) / len(compare_ids) if compare_ids else 0


fuzzy_matches = []
used_compare_ids = set()

compare_choices = list(only_in_compare)

for base_id in only_in_base:
   
    available_choices = [x for x in compare_choices if x not in used_compare_ids]

    if not available_choices:
        continue

    match = process.extractOne(
        base_id,
        available_choices,
        scorer=fuzz.token_sort_ratio
    )

    if match:
        matched_compare_id, score, _ = match

        if score >= FUZZY_THRESHOLD:
            fuzzy_matches.append({
                "base_normalized": base_id,
                "compare_normalized": matched_compare_id,
                "base_original": base_lookup[base_id],
                "compare_original": compare_lookup[matched_compare_id],
                "base_frequency": base_freq_lookup[base_id],
                "compare_frequency": compare_freq_lookup[matched_compare_id],
                "score": score
            })
            used_compare_ids.add(matched_compare_id)

fuzzy_df = pd.DataFrame(fuzzy_matches)


fuzzy_base_ids = set()
fuzzy_compare_ids = set()

for row in fuzzy_matches:
    fuzzy_base_ids.add(row["base_normalized"])
    fuzzy_compare_ids.add(row["compare_normalized"])

enhanced_common_count = len(exact_common_ids) + len(fuzzy_matches)

enhanced_only_in_base = only_in_base - fuzzy_base_ids
enhanced_only_in_compare = only_in_compare - fuzzy_compare_ids

enhanced_jaccard = enhanced_common_count / (
    enhanced_common_count + len(enhanced_only_in_base) + len(enhanced_only_in_compare)
) if (enhanced_common_count + len(enhanced_only_in_base) + len(enhanced_only_in_compare)) else 0

enhanced_recall = enhanced_common_count / len(base_ids) if base_ids else 0
enhanced_precision = enhanced_common_count / len(compare_ids) if compare_ids else 0

exact_rows = []
for entity_id in exact_common_ids:
    exact_rows.append({
        "match_type": "exact",
        "base_normalized": entity_id,
        "compare_normalized": entity_id,
        "base_original": base_lookup[entity_id],
        "compare_original": compare_lookup[entity_id],
        "base_frequency": base_freq_lookup[entity_id],
        "compare_frequency": compare_freq_lookup[entity_id],
        "score": 100
    })

combined_matches_df = pd.DataFrame(exact_rows + fuzzy_matches)

if not combined_matches_df.empty:
    combined_matches_df["difference"] = (
        combined_matches_df["compare_frequency"] - combined_matches_df["base_frequency"]
    )
    combined_matches_df["abs_difference"] = combined_matches_df["difference"].abs()

    numerator = combined_matches_df[["base_frequency", "compare_frequency"]].min(axis=1).sum()
    denominator = combined_matches_df[["base_frequency", "compare_frequency"]].max(axis=1).sum()
    enhanced_weighted_similarity = numerator / denominator if denominator else 0
else:
    enhanced_weighted_similarity = 0


only_in_base_df = base_grouped[base_grouped["Id_norm"].isin(enhanced_only_in_base)].copy()
only_in_compare_df = compare_grouped[compare_grouped["Id_norm"].isin(enhanced_only_in_compare)].copy()

only_in_base_df = only_in_base_df.sort_values(["frequency", "Id"], ascending=[False, True])
only_in_compare_df = only_in_compare_df.sort_values(["frequency", "Id"], ascending=[False, True])


summary = pd.DataFrame({
    "Metric": [
        "Base Unique",
        "Compare Unique",
        "Exact Common",
        "Fuzzy Matches Added",
        "Enhanced Common",
        "Only in Base After Fuzzy",
        "Only in Compare After Fuzzy",
        "Exact Jaccard %",
        "Exact Recall %",
        "Exact Precision %",
        "Enhanced Jaccard %",
        "Enhanced Recall %",
        "Enhanced Precision %",
        "Enhanced Weighted Similarity %",
        "Fuzzy Threshold"
    ],
    "Value": [
        len(base_ids),
        len(compare_ids),
        len(exact_common_ids),
        len(fuzzy_matches),
        enhanced_common_count,
        len(enhanced_only_in_base),
        len(enhanced_only_in_compare),
        round(exact_jaccard * 100, 2),
        round(exact_recall * 100, 2),
        round(exact_precision * 100, 2),
        round(enhanced_jaccard * 100, 2),
        round(enhanced_recall * 100, 2),
        round(enhanced_precision * 100, 2),
        round(enhanced_weighted_similarity * 100, 2),
        FUZZY_THRESHOLD
    ]
})


with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
    summary.to_excel(writer, sheet_name="Summary", index=False)
    combined_matches_df.to_excel(writer, sheet_name="Matched Exact+Fuzzy", index=False)
    fuzzy_df.to_excel(writer, sheet_name="Fuzzy Matches Only", index=False)
    only_in_base_df.to_excel(writer, sheet_name="Only in Base", index=False)
    only_in_compare_df.to_excel(writer, sheet_name="Only in Compare", index=False)


print("\nComparison complete")
print(f"Report saved to: {OUTPUT_FILE}")

print("\nSummary:")
print(summary)