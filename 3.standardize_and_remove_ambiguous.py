from pathlib import Path
import re
import pandas as pd


BASE_DIR = Path("/Users/maysounbrghl/Desktop/Thesis/jfk_data_analysis")
INPUT_FILE = BASE_DIR / "data/Merge/merged_entities.xlsx"
ENTITIES_H_FILE = BASE_DIR / "data/Merge/entities_H.xlsx"

OUTPUT_FILE = BASE_DIR / "data/Merge/merged_entities_standardized_cleaned.xlsx"
REMOVED_FILE = BASE_DIR / "data/Merge/removed_ambiguous_entities.xlsx"
REVIEW_FILE = BASE_DIR / "data/Merge/alias_review_candidates.xlsx"

# Safe canonical mappings

CANONICAL_MAP = {
    # John F. Kennedy
    "JFK": "John F. Kennedy",
    "JOHN F KENNEDY": "John F. Kennedy",
    "JOHN F. KENNEDY": "John F. Kennedy",
    "JOHN FITZGERALD KENNEDY": "John F. Kennedy",
    "JOHN KENNEDY": "John F. Kennedy",
    "PRESIDENT KENNEDY": "John F. Kennedy",
    "PRESIDENT JOHN KENNEDY": "John F. Kennedy",
    "PRESIDENT JOHN F. KENNEDY": "John F. Kennedy",

    # Robert F. Kennedy
    "ROBERT KENNEDY": "Robert F. Kennedy",
    "ROBERT F KENNEDY": "Robert F. Kennedy",
    "ROBERT F. KENNEDY": "Robert F. Kennedy",
    "BOBBY KENNEDY": "Robert F. Kennedy",
    "RFK": "Robert F. Kennedy",
    "ATTORNEY GENERAL KENNEDY": "Robert F. Kennedy",

    # Martin Luther King Jr.
    "MARTIN LUTHER KING": "Martin Luther King Jr.",
    "MARTIN LUTHER KING JR": "Martin Luther King Jr.",
    "MARTIN LUTHER KING JR.": "Martin Luther King Jr.",
    "DR MARTIN LUTHER KING": "Martin Luther King Jr.",
    "DR. MARTIN LUTHER KING": "Martin Luther King Jr.",
    "MLK": "Martin Luther King Jr.",

    # Lee Harvey Oswald
    "LEE HARVEY OSWALD": "Lee Harvey Oswald",
    "LEE OSWALD": "Lee Harvey Oswald",
    "OSWALD": "Lee Harvey Oswald",
    "LHO (LEE HARVEY OSWALD)": "Lee Harvey Oswald",
    "LEE HARVEY OSWALD (LHO)": "Lee Harvey Oswald",
    "LEE HENRY OSWALD": "Lee Harvey Oswald",

    # Other Individuals
    "ALFREDO MIRABAL": "Alfredo Mirabal Diaz",
    "CLAY SHAW": "Clay L. Shaw",
    "DAVID FERRIE": "David W. Ferrie",
    "ELENA GARRO": "Elena Garro de Paz",
    "EUSEBIO AZCUE": "Eusebio Azcue Lopez",
    "GEORGE DEMOHRENSCHILDT": "George de Mohrenschildt",
    "MARINA NIKOLAEVNA PUSAKOVA": "Marina Oswald",
    "MARINA PRUSAKOVA": "Marina Oswald",
    "MRS. OSWALD": "Marina Oswald",
    "OSCAR CONTRERAS": "Oscar Contreras Lartigue",
    "RUBEN DURAN": "Ruben Duran Navarro",
    "SILVIA DURAN": "Silvia Tirado de Duran",
    "VALERIY KOSTIKOV": "Valeriy Vladimirovich Kostikov",
    "VALERI VLADIMIROVICH KOSTOKOV": "Valeriy Vladimirovich Kostikov",
    "JACK RUBY": "Jack Ruby",
    "RUBY": "Jack Ruby",

    # Agencies / Organizations
    "CIA": "Central Intelligence Agency",
    "CENTRAL INTELLIGENCE AGENCY": "Central Intelligence Agency",
    "THE CIA": "Central Intelligence Agency",
    "FBI": "Federal Bureau of Investigation",
    "FEDERAL BUREAU OF INVESTIGATION": "Federal Bureau of Investigation",
    "NSA": "National Security Agency",
    "NATIONAL SECURITY AGENCY": "National Security Agency",
    "JCS": "Joint Chiefs of Staff",
    "JOINT CHIEFS OF STAFF": "Joint Chiefs of Staff",
    "ARB": "Assassination Records Review Board",
    "ARRB": "Assassination Records Review Board",
    "ASSASINATIONS RECORDS REVIEW BOARD": "Assassination Records Review Board",
    "JFK ASSASSINATION REVIEW BOARD": "Assassination Records Review Board",
    "JFK ASSASSINATION RECORDS REVIEW BOARD": "Assassination Records Review Board",
    "HOUSE SELECT COMMITTEE ON ASSASSINATIONS (HSCA)": "HSCA",
    "HOUSE SELECT COMMITTEE ON ASSASSINATIONS": "HSCA",
    "KENNEDY SELECT COMMITTEE ON ASSASSINATIONS": "HSCA",
    "DOMESTIC CONTACTS DIVISION": "Domestic Contact Division",
    "OFFICE OF SECURITY (OS)": "Office of Security",
    "U. S. MARINE CORPS": "United States Marines",
    "UNITED STATES MARINE CORPS": "United States Marines",
    "PARTIDO COMUNISTA": "Communist Party",
    "COMMUNIST PARTY (CP)": "Communist Party",
    "PARTIDO COMUNISTA DE LOS ESTADOS UNIDOS": "American Communist Party",
    "COMMUNIST PARTY OF USA": "American Communist Party",

    # Entities / Locations
    "USSR": "Soviet Union",
    "SOVIET UNION": "Soviet Union",
    "UNION OF SOVIET SOCIALIST REPUBLICS": "Soviet Union",
    "WHITE HOUSE": "White House",
    "STATE DEPARTMENT": "U.S. Department of State",
    "DEPARTMENT OF STATE": "U.S. Department of State",
    "DEPARTMENT OF DEFENSE": "U.S. Department of Defense",
    "PENTAGON": "U.S. Department of Defense",
    "U.S. EMBASSY IN MOSCOW": "American Embassy in Moscow",
    "UNITED STATES EMBASSY IN MOSCOW": "American Embassy in Moscow",
    "CONSULADO": "Consulate",
    "CONSULADO CUBANO": "Cuban Consulate",
    "CONSULADO RUSO": "Soviet Consulate",
    "RUSSIAN CONSULATE": "Soviet Consulate",
    "RUSSIAN EMBASSY": "Soviet Embassy",
    "EMBAJADA CUBANA": "Cuban Embassy",
    "EMBAJADA DE CUBA": "Cuban Embassy",
}

# Ambiguous names to remove only when description is weak

AMBIGUOUS_IDS = {
    "KENNEDY",
    "MR KENNEDY",
    "MRS KENNEDY",
    "KING",
    "MR KING",
    "DR KING",
    "JOHNSON",
    "MR JOHNSON",
    "SMITH",
    "MR SMITH",
    "BROWN",
    "MR BROWN",
    "JONES",
    "MR JONES",
    "WILLIAMS",
    "MR WILLIAMS",
    "MARTIN",
    "ROBERT",
    "JOHN",
    "PRESIDENT",
    "ATTORNEY GENERAL",
    "DIRECTOR",
    "AGENT",
    "OFFICER",
    "NAME",
    "STATE",
    "STATION",
}

# Helpers

def norm_text(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def clean_text(x):
    return re.sub(r"\s+", " ", norm_text(x)).strip()


def normalize_for_match(text: str) -> str:
    text = clean_text(text).upper()
    text = text.replace("’", "'")
    text = re.sub(r"[^\w\s\.\-'\(\)]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_id_for_h_match(text: str) -> str:
    text = clean_text(text).upper()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_weak_description(desc: str) -> bool:
    d = clean_text(desc).lower()
    weak_values = {
        "",
        "organization mentioned in the jfk records",
        "person mentioned in the jfk records",
        "location mentioned in the jfk records",
        "entity mentioned in the jfk records",
        "mentioned in the jfk records",
        "organization mentioned in the jfk record",
        "person mentioned in the jfk record",
        "location mentioned in the jfk record",
        "entity mentioned in the jfk record",
        "unknown",
        "n/a",
        "na",
        "none",
    }
    return d in weak_values


def clean_display_name(text: str) -> str:
    text = clean_text(text)
    if not text:
        return ""

    acronyms = {
        "CIA", "FBI", "JFK", "RFK", "MLK", "NSA",
        "JCS", "USSR", "KGB", "HSCA", "ARRB", "ARB"
    }

    parts = text.split()
    out = []
    for p in parts:
        up = p.upper().strip(".")
        if up in acronyms:
            out.append(up)
        elif len(p) == 1:
            out.append(p.upper())
        else:
            out.append(p.capitalize())
    return " ".join(out)


def is_ambiguous_id(entity_id: str) -> bool:
    normalized = normalize_for_match(entity_id)

    if normalized in AMBIGUOUS_IDS:
        return True

    parts = normalized.split()
    if len(parts) == 1:
        token = parts[0]
        if token not in CANONICAL_MAP and len(token) > 2:
            safe_singletons = {
                "OSWALD", "RUBY", "JFK", "CIA", "FBI", "NSA",
                "JCS", "USSR", "KGB", "CUBA", "HSCA", "ARRB", "ARB"
            }
            if token not in safe_singletons:
                return True

    return False


def canonicalize_entity(entity_id: str) -> tuple[str, str]:
    original = clean_text(entity_id)
    normalized = normalize_for_match(original)

    if not original:
        return "", "empty"

    if normalized in CANONICAL_MAP:
        return CANONICAL_MAP[normalized], "dictionary_exact"

    simplified = re.sub(
        r"\b(MR|MRS|MS|MISS|DR|DR\.|PRESIDENT|PRES|GENERAL|GEN|SENATOR|SEN|GOVERNOR|GOV|ATTORNEY GENERAL|AGENT|OFFICER|THE)\b",
        "",
        normalized
    )
    simplified = re.sub(r"\s+", " ", simplified).strip()

    if simplified in CANONICAL_MAP:
        return CANONICAL_MAP[simplified], "dictionary_simplified"

    return clean_display_name(original), "cleaned_original"


def looks_like_person_name(text: str) -> bool:
    text = clean_text(text)
    parts = re.findall(r"[A-Za-z][A-Za-z\.\-']*", text)
    return len(parts) >= 2


def surname_key(text: str) -> str:
    text = normalize_for_match(text)
    parts = text.split()
    if not parts:
        return ""
    return parts[-1]


def build_alias_review(df: pd.DataFrame) -> pd.DataFrame:
    tmp = df.copy()
    tmp["is_person_like"] = tmp["Id"].apply(looks_like_person_name)
    tmp["surname_group"] = tmp["Id"].apply(surname_key)

    review = (
        tmp[tmp["is_person_like"]]
        .groupby("surname_group", dropna=False)
        .agg(
            n_rows=("Id", "size"),
            n_unique_ids=("Id", lambda x: len(set(clean_text(v) for v in x if pd.notna(v)))),
            ids=("Id", lambda x: " | ".join(sorted(set(clean_text(v) for v in x if pd.notna(v)))[:40])),
            canonical_names=("canonical_name", lambda x: " | ".join(sorted(set(clean_text(v) for v in x if pd.notna(v)))[:20])),
            statuses=("Status", lambda x: " | ".join(sorted(set(clean_text(v) for v in x if pd.notna(v))))),
        )
        .reset_index()
    )

    review = review[
        (review["surname_group"] != "") &
        (review["n_unique_ids"] > 1)
    ].copy()

    review = review.sort_values(
        by=["n_unique_ids", "n_rows", "surname_group"],
        ascending=[False, False, True]
    ).reset_index(drop=True)

    return review


def choose_best_description(desc_series):
    descriptions = [clean_text(x) for x in desc_series if clean_text(x)]
    if not descriptions:
        return ""

    non_weak = [d for d in descriptions if not is_weak_description(d)]
    if non_weak:
        return max(non_weak, key=len)

    return max(descriptions, key=len)

# Load

merged = pd.read_excel(INPUT_FILE)
entities_h = pd.read_excel(ENTITIES_H_FILE)

merged.columns = [c.strip() for c in merged.columns]
entities_h.columns = [c.strip() for c in entities_h.columns]

required_merged = ["source_files", "Id", "Description", "Status"]
required_h = ["Id", "Description"]

for col in required_merged:
    if col not in merged.columns:
        raise ValueError(f"Column '{col}' not found in {INPUT_FILE}")

for col in required_h:
    if col not in entities_h.columns:
        raise ValueError(f"Column '{col}' not found in {ENTITIES_H_FILE}")

for df in [merged, entities_h]:
    df["Id"] = df["Id"].apply(clean_text)
    df["Description"] = df["Description"].fillna("").apply(clean_text)

merged["source_files"] = merged["source_files"].apply(clean_text)
merged["Status"] = merged["Status"].apply(clean_text)

if "Type" not in merged.columns:
    merged["Type"] = ""
merged["Type"] = merged["Type"].apply(clean_text)

# Standardize names

canon_results = merged["Id"].apply(canonicalize_entity)
merged["canonical_name"] = canon_results.apply(lambda x: x[0])
merged["canonical_source"] = canon_results.apply(lambda x: x[1])

merged["canonical_file_Id"] = merged.apply(
    lambda r: f"{clean_text(r['source_files'])}{clean_text(r['canonical_name'])}",
    axis=1
)

# Build Hanqing description maps:
# 1) by exact Id
# 2) by canonical_name

entities_h["Id_norm"] = entities_h["Id"].apply(normalize_id_for_h_match)

h_desc_by_id = (
    entities_h.groupby("Id_norm")["Description"]
    .apply(choose_best_description)
    .to_dict()
)

h_canon = entities_h.copy()
h_canon_results = h_canon["Id"].apply(canonicalize_entity)
h_canon["canonical_name"] = h_canon_results.apply(lambda x: x[0])
h_canon["canonical_name_norm"] = h_canon["canonical_name"].apply(normalize_id_for_h_match)

h_desc_by_canonical = (
    h_canon.groupby("canonical_name_norm")["Description"]
    .apply(choose_best_description)
    .to_dict()
)

# Update descriptions ONLY for Status = Maysoun
# Priority:
#   1) Hanqing by exact Id
#   2) Hanqing by canonical_name
# Do not change Hanqing or Both

updated_from_h_exact = 0
updated_from_h_canonical = 0
left_original = 0

for idx, row in merged.iterrows():
    status = clean_text(row.get("Status", ""))
    if status != "Maysoun":
        continue

    entity_id = clean_text(row.get("Id", ""))
    id_norm = normalize_id_for_h_match(entity_id)

    h_desc_exact = clean_text(h_desc_by_id.get(id_norm, ""))
    if h_desc_exact and not is_weak_description(h_desc_exact):
        merged.at[idx, "Description"] = h_desc_exact
        merged.at[idx, "description_source"] = "entities_H_by_exact_Id"
        updated_from_h_exact += 1
        continue

    canonical_name = clean_text(row.get("canonical_name", ""))
    canonical_norm = normalize_id_for_h_match(canonical_name)
    h_desc_canonical = clean_text(h_desc_by_canonical.get(canonical_norm, ""))

    if h_desc_canonical and not is_weak_description(h_desc_canonical):
        merged.at[idx, "Description"] = h_desc_canonical
        merged.at[idx, "description_source"] = "entities_H_by_canonical_name"
        updated_from_h_canonical += 1
        continue

    merged.at[idx, "description_source"] = "original"
    left_original += 1

# Remove only if:
#   1) Id is ambiguous
#   2) Description is weak/generic

removed_rows = []
keep_mask = []

for _, row in merged.iterrows():
    entity_id = clean_text(row.get("Id", ""))
    description = clean_text(row.get("Description", ""))

    ambiguous = is_ambiguous_id(entity_id)
    weak_desc = is_weak_description(description)

    if ambiguous and weak_desc:
        row_copy = row.copy()
        row_copy["removal_reason"] = "ambiguous_id_and_weak_description"
        removed_rows.append(row_copy)
        keep_mask.append(False)
    else:
        keep_mask.append(True)

cleaned_df = merged[keep_mask].copy()
removed_df = pd.DataFrame(removed_rows)

# Build review file from cleaned data

review_df = build_alias_review(cleaned_df)

# Save

cleaned_df.to_excel(OUTPUT_FILE, index=False)
removed_df.to_excel(REMOVED_FILE, index=False)
review_df.to_excel(REVIEW_FILE, index=False)

print(f"Saved cleaned standardized file to:\n{OUTPUT_FILE}")
print(f"Saved removed ambiguous rows to:\n{REMOVED_FILE}")
print(f"Saved alias review file to:\n{REVIEW_FILE}")

print(f"\nMaysoun descriptions updated from Hanqing by exact Id: {updated_from_h_exact}")
print(f"Maysoun descriptions updated from Hanqing by canonical name: {updated_from_h_canonical}")
print(f"Maysoun rows left with original description: {left_original}")
print(f"Rows removed as ambiguous + weak description: {len(removed_df)}")

print("\nCanonical source counts:")
print(cleaned_df["canonical_source"].value_counts())