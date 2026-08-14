import re
import csv
from pathlib import Path
from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer


BASE_DIR = Path("/Users/maysounbrghl/Desktop/Thesis/jfk_data_analysis/data/structured_output2")

DOCS_FILE = BASE_DIR / "docs.csv"
ENTITIES_FILE = BASE_DIR / "merged_entities_final.csv"

OUTPUT_DIR = BASE_DIR / "keyword_topic_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_KEYWORD_FREQ = OUTPUT_DIR / "keyword_frequency.csv"
OUT_TFIDF = OUTPUT_DIR / "tfidf_keywords.csv"
OUT_TOP_PERSONS = OUTPUT_DIR / "top_persons.csv"
OUT_TOP_ORGS = OUTPUT_DIR / "top_organizations.csv"
OUT_THEME_SUMMARY = OUTPUT_DIR / "theme_summary.csv"

PLOT_KEYWORDS = OUTPUT_DIR / "top30_keywords.png"
PLOT_PERSONS = OUTPUT_DIR / "top20_persons.png"
PLOT_ORGS = OUTPUT_DIR / "top20_organizations.png"
PLOT_THEMES = OUTPUT_DIR / "theme_distribution.png"


TOP_TFIDF_TERMS = 200
MIN_WORD_LENGTH = 3


CUSTOM_STOPWORDS = {
    "the", "and", "for", "that", "with", "from", "this", "were", "was",
    "are", "his", "her", "their", "have", "has", "had", "not", "but",
    "all", "any", "can", "may", "would", "could", "should", "shall",
    "will", "been", "being", "also", "into", "than", "then", "there",
    "where", "when", "which", "who", "whom", "what", "about", "after",
    "before", "during", "over", "under", "between", "within", "without",
    "document", "documents", "page", "pages", "file", "files", "record",
    "records", "subject", "copy", "date", "memo", "memorandum", "report",
    "said", "one", "two", "three", "first", "second", "new", "old",
    "mr", "mrs", "ms", "dr", "sir", "jfk"
}


THEMES = {
    "Intelligence and security agencies": [
        "central intelligence agency", "cia", "federal bureau of investigation",
        "fbi", "national security agency", "nsa", "kgb", "intelligence",
        "counterintelligence", "security", "surveillance", "station",
        "division", "clandestine", "operation", "operative", "asset"
    ],
    "Cuba and anti-Castro operations": [
        "cuba", "cuban", "fidel alejandro castro ruz", "fidel castro",
        "castro", "havana", "jmwave", "anti castro", "exile",
        "revolutionary", "communist party"
    ],
    "Soviet Union and Cold War": [
        "soviet", "ussr", "moscow", "russia", "kgb", "communist",
        "cold war", "eastern bloc", "marxist", "socialist"
    ],
    "Diplomacy and foreign relations": [
        "u.s. department of state", "department of state", "state department",
        "embassy", "consulate", "ambassador", "diplomatic", "foreign",
        "visa", "passport", "minister"
    ],
    "Military and defense": [
        "army", "navy", "air force", "military", "defense", "pentagon",
        "marine", "soldier", "officer", "weapon", "training"
    ],
    "JFK assassination and investigation": [
        "john f. kennedy", "kennedy", "assassination", "lee harvey oswald",
        "oswald", "jack ruby", "ruby", "warren commission", "commission",
        "dallas", "november", "investigation", "evidence"
    ],
    "Mexico and Latin America": [
        "mexico", "mexico city", "latin america", "guatemala", "nicaragua",
        "chile", "dominican", "caracas", "honduras", "panama"
    ],
}


def clean_text(value):
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def normalize_text(text):
    text = clean_text(text).lower()
    text = re.sub(r"[^a-zA-Z0-9\s\-.]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text):
    text = normalize_text(text)
    tokens = text.split()

    return [
        token for token in tokens
        if len(token) >= MIN_WORD_LENGTH
        and token not in CUSTOM_STOPWORDS
        and not token.isdigit()
    ]


def count_themes_in_text(text):
    lower = normalize_text(text)
    counts = Counter()

    for theme, keywords in THEMES.items():
        for keyword in keywords:
            pattern = r"\b" + re.escape(keyword.lower()) + r"\b"
            counts[theme] += len(re.findall(pattern, lower))

    return counts


def detect_entity_columns(df):
    cols = {c.lower().strip(): c for c in df.columns}

    name_col = None
    type_col = None
    count_col = None

    for candidate in ["canonical_name", "entity_text", "id", "entity", "name"]:
        if candidate in cols:
            name_col = cols[candidate]
            break

    for candidate in ["entity_label", "label", "type", "entity_type"]:
        if candidate in cols:
            type_col = cols[candidate]
            break

    for candidate in ["count_in_doc", "frequency", "count", "weight"]:
        if candidate in cols:
            count_col = cols[candidate]
            break

    if name_col is None:
        raise ValueError("Could not identify entity name column.")

    if type_col is None:
        raise ValueError("Could not identify entity type column.")

    return name_col, type_col, count_col


print("Loading docs.csv...")

if not DOCS_FILE.exists():
    raise FileNotFoundError(f"Missing docs.csv: {DOCS_FILE}")

docs = pd.read_csv(DOCS_FILE)
docs.columns = [c.strip() for c in docs.columns]

if "full_text" not in docs.columns:
    raise ValueError("docs.csv must contain a full_text column.")

texts = docs["full_text"].fillna("").astype(str).tolist()

print(f"Loaded documents: {len(texts):,}")


print("Calculating keyword frequencies from document text...")

keyword_counter = Counter()

for text in texts:
    keyword_counter.update(tokenize(text))

with OUT_KEYWORD_FREQ.open("w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["keyword", "frequency"])

    for keyword, freq in keyword_counter.most_common():
        writer.writerow([keyword, freq])


print("Calculating TF-IDF keywords...")

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words=list(CUSTOM_STOPWORDS),
    token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z\-]{2,}\b",
    max_features=5000,
    ngram_range=(1, 2)
)

tfidf_matrix = vectorizer.fit_transform(texts)
terms = vectorizer.get_feature_names_out()
mean_scores = tfidf_matrix.mean(axis=0).A1

tfidf_terms = sorted(
    zip(terms, mean_scores),
    key=lambda x: x[1],
    reverse=True
)[:TOP_TFIDF_TERMS]

with OUT_TFIDF.open("w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["term", "tfidf_score"])

    for term, score in tfidf_terms:
        writer.writerow([term, score])


print("Loading merged_entities_final.csv...")

if not ENTITIES_FILE.exists():
    raise FileNotFoundError(f"Missing merged_entities_final.csv: {ENTITIES_FILE}")

entities = pd.read_csv(ENTITIES_FILE)
entities.columns = [c.strip() for c in entities.columns]

name_col, type_col, count_col = detect_entity_columns(entities)

entities[name_col] = entities[name_col].apply(clean_text)
entities[type_col] = entities[type_col].apply(lambda x: clean_text(x).upper())

if count_col:
    entities[count_col] = pd.to_numeric(entities[count_col], errors="coerce").fillna(1)
else:
    entities["__count__"] = 1
    count_col = "__count__"

person_counts = Counter()
org_counts = Counter()

for _, row in entities.iterrows():
    name = row[name_col]
    label = row[type_col]
    count = row[count_col]

    if not name:
        continue

    if label == "PERSON":
        person_counts[name] += count
    elif label in {"ORG", "ORGANIZATION"}:
        org_counts[name] += count


with OUT_TOP_PERSONS.open("w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["person", "frequency"])

    for person, freq in person_counts.most_common():
        writer.writerow([person, freq])


with OUT_TOP_ORGS.open("w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["organization", "frequency"])

    for org, freq in org_counts.most_common():
        writer.writerow([org, freq])


print("Calculating theme summary from document text...")

theme_counter = Counter()
theme_doc_counts = Counter()

for text in texts:
    counts = count_themes_in_text(text)

    for theme, count in counts.items():
        theme_counter[theme] += count

        if count > 0:
            theme_doc_counts[theme] += 1


with OUT_THEME_SUMMARY.open("w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "theme",
        "keyword_occurrences",
        "document_count"
    ])

    for theme, count in theme_counter.most_common():
        writer.writerow([
            theme,
            count,
            theme_doc_counts[theme]
        ])



print("Creating plots...")

# Top 30 keywords
top_keywords = keyword_counter.most_common(30)

if top_keywords:
    labels = [x[0] for x in top_keywords]
    values = [x[1] for x in top_keywords]

    plt.figure(figsize=(12, 8))
    plt.barh(labels[::-1], values[::-1])
    plt.title("Top 30 Keywords in JFK Corpus")
    plt.xlabel("Frequency")
    plt.tight_layout()
    plt.savefig(PLOT_KEYWORDS, dpi=300)
    plt.close()


# Top 10 persons
top_persons = person_counts.most_common(10)

if top_persons:
    labels = [x[0] for x in top_persons]
    values = [x[1] for x in top_persons]

    plt.figure(figsize=(12, 8))
    plt.barh(labels[::-1], values[::-1])
    plt.title("Top 10 Persons in Validated Entity Dataset")
    plt.xlabel("Frequency")
    plt.tight_layout()
    plt.savefig(PLOT_PERSONS, dpi=300)
    plt.close()


# Top 10 organizations
top_orgs = org_counts.most_common(10)

if top_orgs:
    labels = [x[0] for x in top_orgs]
    values = [x[1] for x in top_orgs]

    plt.figure(figsize=(12, 8))
    plt.barh(labels[::-1], values[::-1])
    plt.title("Top 10 Organizations in Validated Entity Dataset")
    plt.xlabel("Frequency")
    plt.tight_layout()
    plt.savefig(PLOT_ORGS, dpi=300)
    plt.close()


# Theme distribution
theme_df = pd.DataFrame([
    {
        "theme": theme,
        "keyword_occurrences": theme_counter[theme],
        "document_count": theme_doc_counts[theme]
    }
    for theme in theme_counter
]).sort_values("keyword_occurrences", ascending=True)

if not theme_df.empty:
    plt.figure(figsize=(12, 7))
    plt.barh(theme_df["theme"], theme_df["keyword_occurrences"])
    plt.title("Theme Distribution in JFK Corpus")
    plt.xlabel("Keyword occurrences")
    plt.tight_layout()
    plt.savefig(PLOT_THEMES, dpi=300)
    plt.close()


print("\nDONE.")
print("Created:")
print(" -", OUT_KEYWORD_FREQ)
print(" -", OUT_TFIDF)
print(" -", OUT_TOP_PERSONS)
print(" -", OUT_TOP_ORGS)
print(" -", OUT_THEME_SUMMARY)
print(" -", PLOT_KEYWORDS)
print(" -", PLOT_PERSONS)
print(" -", PLOT_ORGS)
print(" -", PLOT_THEMES)