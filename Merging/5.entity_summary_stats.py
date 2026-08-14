from pathlib import Path
import pandas as pd
import re


BASE_DIR = Path("/Users/maysounbrghl/Desktop/Thesis/jfk_data_analysis")
INPUT_FILE = BASE_DIR / "data/Merge/merged_entities_final.xlsx"
OUTPUT_FILE = BASE_DIR / "data/Merge/entity_summary_stats.xlsx"

# Helpers

def clean_text(x):
    if pd.isna(x):
        return ""
    return re.sub(r"\s+", " ", str(x)).strip()


def normalize(x):
    return clean_text(x).lower()


def normalize_type(t):
    t = normalize(t)
    if "org" in t:
        return "Organization"
    if "person" in t:
        return "Person"
    return "Other"


# Load

df = pd.read_excel(INPUT_FILE)
df.columns = [c.strip() for c in df.columns]

df["Type"] = df["Type"].apply(clean_text)
df["Status"] = df["Status"].apply(clean_text)

df["Type_clean"] = df["Type"].apply(normalize_type)

# Split by type

org_df = df[df["Type_clean"] == "Organization"]
person_df = df[df["Type_clean"] == "Person"]


# Counts with BOTH included

def count_with_both(df_subset):
    maysoun = df_subset[df_subset["Status"].isin(["Maysoun", "Both"])]
    hanqing = df_subset[df_subset["Status"].isin(["Hanqing", "Both"])]

    return len(maysoun), len(hanqing)


m_person, h_person = count_with_both(person_df)
m_org, h_org = count_with_both(org_df)

# Totals

m_total = m_person + m_org
h_total = h_person + h_org

# Percentages

def pct(part, whole):
    return round((part / whole) * 100, 2) if whole else 0

# Final table (your requested one)

final_table = pd.DataFrame({
    "Category": [
        "Maysoun_Person",
        "Maysoun_Organization",
        "Hanqing_Person",
        "Hanqing_Organization",
    ],
    "Count": [
        m_person,
        m_org,
        h_person,
        h_org,
    ],
    "Percentage": [
        pct(m_person, m_total),
        pct(m_org, m_total),
        pct(h_person, h_total),
        pct(h_org, h_total),
    ]
})


# Also keep original summary (optional)

summary_table = pd.DataFrame({
    "Metric": ["Total Persons", "Total Organizations"],
    "Maysoun": [m_person, m_org],
    "Hanqing": [h_person, h_org],
})


# Save to Excel

with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
    final_table.to_excel(writer, sheet_name="Final_Table", index=False)
    summary_table.to_excel(writer, sheet_name="Counts", index=False)
    df.to_excel(writer, sheet_name="Full_Data", index=False)

print(f"\nSaved results to:\n{OUTPUT_FILE}")