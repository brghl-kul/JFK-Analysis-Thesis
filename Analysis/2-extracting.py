import re
import csv
import json
import string
from pathlib import Path
from itertools import combinations
from typing import List, Tuple, Optional

import pandas as pd
import spacy
from tqdm import tqdm
from textblob import TextBlob
import textstat


TXT_DIR = Path("/Users/maysounbrghl/Desktop/Thesis/jfk_data_analysis/data/cleaned_text_files")
OUTPUT_DIR = Path("/Users/maysounbrghl/Desktop/Thesis/jfk_data_analysis/data/structured_output2")
DICT_FILE = Path("/Users/maysounbrghl/Desktop/Thesis/jfk_data_analysis/data/jfk_dictionary.csv")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DOCS_CSV = OUTPUT_DIR / "docs.csv"
ENTITIES_CSV = OUTPUT_DIR / "entities.csv"
RELATIONS_CSV = OUTPUT_DIR / "relations.csv"


try:
    nlp = spacy.load("en_core_web_trf", disable=["lemmatizer"])
except OSError:
    nlp = spacy.load("en_core_web_sm", disable=["lemmatizer"])

nlp.max_length = max(nlp.max_length, 5_000_000)


if not DICT_FILE.exists():
    raise FileNotFoundError(f"Missing dictionary file: {DICT_FILE}")

dict_df = pd.read_csv(DICT_FILE).fillna("")

required_cols = {"entity", "label", "canonical_name", "description"}
missing = required_cols - set(dict_df.columns)

if missing:
    raise ValueError(f"Dictionary file is missing columns: {missing}")

PERSON_ALIASES = {}
ORG_ALIASES = {}
DESCRIPTION_MAP = {}

patterns = []

for _, row in dict_df.iterrows():
    entity = str(row["entity"]).strip()
    label = str(row["label"]).strip().upper()
    canonical = str(row["canonical_name"]).strip()
    description = str(row["description"]).strip()

    if not entity or not canonical:
        continue

    if label not in {"PERSON", "ORG"}:
        continue

    patterns.append({
        "label": label,
        "pattern": entity
    })

    DESCRIPTION_MAP[canonical] = description

    if label == "PERSON":
        PERSON_ALIASES[entity] = canonical
    elif label == "ORG":
        ORG_ALIASES[entity] = canonical



if "entity_ruler" not in nlp.pipe_names:
    ruler = nlp.add_pipe("entity_ruler", before="ner")
else:
    ruler = nlp.get_pipe("entity_ruler")

ruler.add_patterns(patterns)



JFK_KEYWORDS = {
    "kennedy", "jfk", "president kennedy",
    "oswald", "lee harvey oswald",
    "cia", "fbi",
    "warren commission", "assassination", "conspiracy",
    "ruby", "castro",
    "soviet", "ussr"
}

JFK_EVENT_PHRASES = [
    "kennedy assassination",
    "warren commission",
    "november 22, 1963",
    "22 november 1963",
    "22 nov 1963",
    "magic bullet",
    "single-bullet theory"
]

BAD_ENTITIES = {
    "Mr", "Mrs", "Ms", "Dr", "Sir",
    "The", "This", "That",
    "Agency", "Bureau", "Commission", "Department",
    "Office", "Committee", "Service"
}



def load_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def clean_text_fragment(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = text.strip(" ,.;:-_()[]{}\"'")
    return text


def normalize_person(text: str) -> Optional[str]:
    text = clean_text_fragment(text)

    if not text or text in BAD_ENTITIES:
        return None

    if text in PERSON_ALIASES:
        return PERSON_ALIASES[text]

    parts = []

    for token in text.split():
        if re.fullmatch(r"[A-Z]\.", token):
            parts.append(token)
        elif token.isupper() and len(token) <= 4:
            parts.append(token)
        else:
            parts.append(token.capitalize())

    text = " ".join(parts)

    if len(text.split()) > 6:
        return None

    return PERSON_ALIASES.get(text, text)


def normalize_org(text: str) -> Optional[str]:
    text = clean_text_fragment(text)

    if not text or text in BAD_ENTITIES:
        return None

    if text in ORG_ALIASES:
        return ORG_ALIASES[text]

    if not text.isupper():
        parts = []

        for token in text.split():
            if token.upper() in {"CIA", "FBI", "HSCA", "KGB"}:
                parts.append(token.upper())
            else:
                parts.append(token.capitalize())

        text = " ".join(parts)

    return ORG_ALIASES.get(text, text)


def get_description(entity_text: str, entity_type: str) -> str:
    if entity_text in DESCRIPTION_MAP:
        return DESCRIPTION_MAP[entity_text]

    if entity_type == "PERSON":
        return "Person mentioned in the JFK records"

    if entity_type == "ORG":
        return "Organization mentioned in the JFK records"

    return ""


DATE_PATTERNS = [
    r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.? \d{1,2}, \d{4}\b",
    r"\b\d{1,2} (?:January|February|March|April|May|June|July|August|September|October|November|December) \d{4}\b",
    r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b",
    r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b",
]


def extract_dates(text: str, doc) -> str:
    dates = set()

    for pattern in DATE_PATTERNS:
        for match in re.findall(pattern, text):
            dates.add(match)

    for ent in doc.ents:
        if ent.label_ == "DATE":
            dates.add(ent.text)

    return ", ".join(sorted(dates))


def extract_numbers(text: str) -> str:
    nums = re.findall(r"\b\d{3,}\b", text)

    unique = []
    seen = set()

    for num in nums:
        if num not in seen:
            seen.add(num)
            unique.append(num)

    return ", ".join(unique)


def guess_doc_type(text: str) -> str:
    header = text[:2000].lower()

    if "memorandum" in header or "memo" in header:
        return "memo"
    if "dispatch" in header:
        return "dispatch"
    if "to: director" in header or "to director" in header:
        return "cable"
    if "confidential" in header or "secret" in header:
        return "classified"
    if "report" in header:
        return "report"
    if "letter" in header or header.startswith("dear"):
        return "letter"

    return "other"


FROM_PATTERNS = [
    r"(?im)^\s*from\s*[:\-]?\s*(.{3,120})\s*$",
    r"(?im)^\s*fm\s*[:\-]?\s*(.{3,120})\s*$",
    r"(?im)^\s*sent\s+by\s*[:\-]?\s*(.{3,120})\s*$",
    r"(?im)^\s*sender\s*[:\-]?\s*(.{3,120})\s*$",
]

TO_PATTERNS = [
    r"(?im)^\s*to\s*[:\-]?\s*(.{3,120})\s*$",
    r"(?im)^\s*attn\s*[:\-]?\s*(.{3,120})\s*$",
    r"(?im)^\s*attention\s*[:\-]?\s*(.{3,120})\s*$",
]

DEAR_PAT = r"(?im)^\s*dear\s+(.{2,120})\s*[,:\-]\s*$"
SIGN_PAT = r"(?im)^\s*(sincerely|yours truly|respectfully|cordially)\s*,?\s*$"


def _clean_header_value(value: str) -> str:
    if not value:
        return ""

    value = value.strip()

    value = re.split(
        r"\b(subject|subj|date|ref|re)\b\s*[:\-]",
        value,
        flags=re.I
    )[0].strip()

    value = re.sub(
        r"\b(confidential|secret|top secret|classified)\b",
        "",
        value,
        flags=re.I
    ).strip()

    value = re.sub(r"\s{2,}", " ", value).strip()

    if len(value) < 2 or len(value) > 120:
        return ""

    return value


def extract_sender(text: str) -> str:
    for pattern in FROM_PATTERNS:
        match = re.search(pattern, text)

        if match:
            candidate = _clean_header_value(match.group(1))

            if candidate:
                return candidate

    lines = text.splitlines()

    for i, line in enumerate(lines[:-2]):
        if re.search(SIGN_PAT, line):
            for j in range(i + 1, min(i + 6, len(lines))):
                name = _clean_header_value(lines[j].strip())

                if name:
                    return name

    return ""


def extract_recipient(text: str) -> str:
    for pattern in TO_PATTERNS:
        match = re.search(pattern, text)

        if match:
            candidate = _clean_header_value(match.group(1))

            if candidate:
                return candidate

    match = re.search(DEAR_PAT, text)

    if match:
        candidate = _clean_header_value(match.group(1))

        if candidate:
            return candidate

    return ""


def is_letter_like(text: str) -> bool:
    lower = text.lower()[:3000]

    return "dear " in lower and (
        "sincerely" in lower
        or "yours truly" in lower
        or "respectfully" in lower
    )

# ENTITY EXTRACTION — PERSON AND ORG 

def extract_entities(doc) -> Tuple[List[str], List[str]]:
    persons = []
    orgs = []

    for ent in doc.ents:
        raw = ent.text.strip()

        if not raw:
            continue

        if ent.label_ == "PERSON":
            norm = normalize_person(raw)

            if norm:
                persons.append(norm)

        elif ent.label_ == "ORG":
            norm = normalize_org(raw)

            if norm:
                orgs.append(norm)

    return persons, orgs


def sentiment(text: str):
    blob = TextBlob(text)
    return blob.sentiment.polarity, blob.sentiment.subjectivity


def keyword_counts(text: str):
    lower = text.lower()
    return {k: lower.count(k) for k in JFK_KEYWORDS if k in lower}


def jfk_events(text: str):
    lower = text.lower()
    return ", ".join(sorted({
        phrase for phrase in JFK_EVENT_PHRASES
        if phrase in lower
    }))


def text_statistics(text: str):
    words = text.split()
    word_count = len(words)

    sentences = re.split(r"[.!?]", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    avg_word_len = (
        sum(len(word) for word in words) / word_count
        if word_count else 0
    )

    avg_sentence_length = (
        sum(len(sentence.split()) for sentence in sentences) / len(sentences)
        if sentences else 0
    )

    type_token_ratio = (
        len(set(words)) / word_count
        if word_count else 0
    )

    try:
        flesch_reading_ease = textstat.flesch_reading_ease(text)
        flesch_kincaid_grade = textstat.flesch_kincaid_grade(text)
        total_syllables = textstat.syllable_count(text)
    except Exception:
        flesch_reading_ease = 0
        flesch_kincaid_grade = 0
        total_syllables = 0

    punctuation_counts = {
        p: text.count(p)
        for p in string.punctuation
    }

    return {
        "word_count": word_count,
        "avg_word_length": avg_word_len,
        "avg_sentence_length": avg_sentence_length,
        "type_token_ratio": type_token_ratio,
        "flesch_reading_ease": flesch_reading_ease,
        "flesch_kincaid_grade": flesch_kincaid_grade,
        "total_syllables": total_syllables,
        "punctuation_counts": json.dumps(punctuation_counts),
        "punctuation_count": sum(punctuation_counts.values()),
    }


def build_relations(
    file: str,
    persons: List[str],
    orgs: List[str],
    sender: str,
    recipient: str
):
    relations = []

    persons = list(dict.fromkeys(persons))
    orgs = list(dict.fromkeys(orgs))

    for person_a, person_b in combinations(persons, 2):
        relations.append({
            "file": file,
            "source_text": person_a,
            "source_type": "PERSON",
            "target_text": person_b,
            "target_type": "PERSON",
            "relation": "cooccurs",
        })

    for person in persons:
        for org in orgs:
            relations.append({
                "file": file,
                "source_text": person,
                "source_type": "PERSON",
                "target_text": org,
                "target_type": "ORG",
                "relation": "mentions",
            })

    for org_a, org_b in combinations(orgs, 2):
        relations.append({
            "file": file,
            "source_text": org_a,
            "source_type": "ORG",
            "target_text": org_b,
            "target_type": "ORG",
            "relation": "cooccurs",
        })

    if sender and recipient:
        relations.append({
            "file": file,
            "source_text": sender,
            "source_type": "SENDER",
            "target_text": recipient,
            "target_type": "RECIPIENT",
            "relation": "sender_to_recipient",
        })

    return relations


def process_file(path: Path):
    text = load_text(path)

    if not text:
        return None, [], []

    file = path.name
    doc = nlp(text)

    persons, orgs = extract_entities(doc)

    sender = extract_sender(text)
    recipient = extract_recipient(text)
    letter = is_letter_like(text)
    doc_type = guess_doc_type(text)

    dates = extract_dates(text, doc)
    numbers = extract_numbers(text)

    polarity, subjectivity = sentiment(text)
    events = jfk_events(text)
    keyword_frequency = keyword_counts(text)
    stats = text_statistics(text)

    doc_row = {
        "file": file,
        "doc_type": doc_type,
        "is_letter_like": letter,
        "sender": sender or "",
        "recipient": recipient or "",
        "dates": dates,
        "numbers": numbers,
        "jfk_events": events,
        "sentiment_polarity": polarity,
        "sentiment_subjectivity": subjectivity,
        "keyword_frequency": json.dumps(keyword_frequency),
        "total_keywords": sum(keyword_frequency.values()),
        "top_keyword": (
            max(keyword_frequency, key=keyword_frequency.get)
            if keyword_frequency else ""
        ),
        "full_text": text,
    }

    doc_row.update(stats)

    entities = []

    def add_entities(entity_list: List[str], label: str):
        counts = {}

        for entity in entity_list:
            counts[entity] = counts.get(entity, 0) + 1

        for entity, count in counts.items():
            entities.append({
                "file": file,
                "entity_text": entity,
                "entity_label": label,
                "count_in_doc": count,
                "description": get_description(entity, label),
            })

    add_entities(persons, "PERSON")
    add_entities(orgs, "ORG")

    relations = build_relations(
        file=file,
        persons=persons,
        orgs=orgs,
        sender=sender,
        recipient=recipient
    )

    return doc_row, entities, relations


def main():
    if not TXT_DIR.exists():
        print(f"❌ Missing directory: {TXT_DIR}")
        return

    files = sorted(TXT_DIR.glob("*.txt"))

    DOC_FIELDS = [
        "file",
        "doc_type",
        "is_letter_like",
        "sender",
        "recipient",
        "dates",
        "numbers",
        "jfk_events",
        "sentiment_polarity",
        "sentiment_subjectivity",
        "keyword_frequency",
        "total_keywords",
        "top_keyword",
        "full_text",
        "word_count",
        "avg_word_length",
        "avg_sentence_length",
        "type_token_ratio",
        "flesch_reading_ease",
        "flesch_kincaid_grade",
        "total_syllables",
        "punctuation_counts",
        "punctuation_count",
    ]

    ENT_FIELDS = [
        "file",
        "entity_text",
        "entity_label",
        "count_in_doc",
        "description",
    ]

    REL_FIELDS = [
        "file",
        "source_text",
        "source_type",
        "target_text",
        "target_type",
        "relation",
    ]

    with DOCS_CSV.open("w", newline="", encoding="utf-8") as f_docs, \
         ENTITIES_CSV.open("w", newline="", encoding="utf-8") as f_entities, \
         RELATIONS_CSV.open("w", newline="", encoding="utf-8") as f_relations:

        docs_writer = csv.DictWriter(f_docs, fieldnames=DOC_FIELDS)
        entities_writer = csv.DictWriter(f_entities, fieldnames=ENT_FIELDS)
        relations_writer = csv.DictWriter(f_relations, fieldnames=REL_FIELDS)

        docs_writer.writeheader()
        entities_writer.writeheader()
        relations_writer.writeheader()

        for path in tqdm(files, desc="Extracting PERSON and ORG entities"):
            doc_row, entities, relations = process_file(path)

            if not doc_row:
                continue

            docs_writer.writerow(doc_row)

            for entity in entities:
                entities_writer.writerow(entity)

            for relation in relations:
                relations_writer.writerow(relation)

    print("\n🎉 Structured extraction complete!")
    print("📄 Docs CSV      →", DOCS_CSV)
    print("🔤 Entities CSV  →", ENTITIES_CSV)
    print("🕸 Relations CSV →", RELATIONS_CSV)


if __name__ == "__main__":
    main()