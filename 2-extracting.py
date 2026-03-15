#!/usr/bin/env python3
"""
2-extracting-info.py

Structured information extractor for cleaned JFK/HSCA/CIA/FBI/NARA documents.
Goals:
  - Extract named entities (PERSON, ORG, LOC).
  - Extract dates and significant numbers.
  - Classify document types (memo, cable, report, letter).
  - Identify sender and recipient where applicable.
  - Compute sentiment and keyword frequencies.
  - Output structured CSV files for documents, entities, and relationships.
"""

import os
import re
import csv
import json
import string
from pathlib import Path
from itertools import combinations
from typing import List, Dict, Tuple

import spacy
from tqdm import tqdm
from textblob import TextBlob
import textstat


# PATHS
TXT_DIR = Path("data/cleaned_text_files")
OUTPUT_DIR = Path("data/structured_output2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DOCS_CSV      = OUTPUT_DIR / "docs.csv"
ENTITIES_CSV  = OUTPUT_DIR / "entities.csv"
RELATIONS_CSV = OUTPUT_DIR / "relations.csv"


nlp = spacy.load("en_core_web_sm", disable=["lemmatizer"])


# JFK RELEVANT KEYWORDS
JFK_KEYWORDS = {
    "kennedy", "jfk", "president kennedy",
    "oswald", "lee harvey oswald",
    "cia", "fbi", "dallas", "dealey plaza",
    "warren commission", "assassination", "conspiracy",
    "ruby", "castro", "cuba", "mexico city",
    "soviet", "ussr"
}

JFK_EVENT_PHRASES = [
    "kennedy assassination",
    "dealey plaza",
    "warren commission",
    "november 22, 1963",
    "22 november 1963",
    "22 nov 1963",
    "magic bullet",
    "single-bullet theory"
]


# LOADING CLEANED TEXT
def load_text(p: Path) -> str:
    try:
        with p.open("r", encoding="utf-8") as f:
            return f.read()
    except:
        return ""


# DATE & NUMBER EXTRACTION
DATE_PATTERNS = [
    r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.? \d{1,2}, \d{4}\b",
    r"\b\d{1,2} (?:January|February|March|April|May|June|July|August|September|October|November|December) \d{4}\b",
    r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b",
    r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b",
]

def extract_dates(text: str, doc) -> str:
    dates = set()
    for pat in DATE_PATTERNS:
        for m in re.findall(pat, text):
            dates.add(m)
    for ent in doc.ents:
        if ent.label_ == "DATE":
            dates.add(ent.text)
    return ", ".join(sorted(dates))


def extract_numbers(text: str) -> str:
    nums = re.findall(r"\b\d{3,}\b", text)
    unique, seen = [], set()
    for n in nums:
        if n not in seen:
            seen.add(n)
            unique.append(n)
    return ", ".join(unique)


# DOCUMENT TYPE CLASSIFIER
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


# -----------------------------
# SENDER & RECIPIENT (IMPROVED)
# -----------------------------
# Works better for JFK/CIA/FBI formats: FROM/TO without colons, FM/TO, ATTN, etc.
FROM_PATTERNS = [
    r"(?im)^\s*from\s*[:\-]?\s*(.{3,120})\s*$",
    r"(?im)^\s*fm\s*[:\-]?\s*(.{3,120})\s*$",          # “FM CIA MEXICO CITY”
    r"(?im)^\s*sent\s+by\s*[:\-]?\s*(.{3,120})\s*$",
    r"(?im)^\s*sender\s*[:\-]?\s*(.{3,120})\s*$",
]

TO_PATTERNS = [
    r"(?im)^\s*to\s*[:\-]?\s*(.{3,120})\s*$",
    r"(?im)^\s*attn\s*[:\-]?\s*(.{3,120})\s*$",        # “ATTN: …”
    r"(?im)^\s*attention\s*[:\-]?\s*(.{3,120})\s*$",
]

# Letter-style patterns (best-effort)
DEAR_PAT = r"(?im)^\s*dear\s+(.{2,120})\s*[,:\-]\s*$"
SIGN_PAT = r"(?im)^\s*(sincerely|yours truly|respectfully|cordially)\s*,?\s*$"

def _clean_header_value(v: str) -> str:
    """Normalize extracted sender/recipient header value (keep org/office labels)."""
    if not v:
        return ""
    v = v.strip()

    # Stop at common header breaks (Subject/Date/etc.)
    v = re.split(r"\b(subject|subj|date|ref|re)\b\s*[:\-]", v, flags=re.I)[0].strip()

    # Remove trailing classification stamps
    v = re.sub(r"\b(confidential|secret|top secret|classified)\b", "", v, flags=re.I).strip()

    # Remove extra whitespace
    v = re.sub(r"\s{2,}", " ", v).strip()

    # Too short / too long => reject
    if len(v) < 2 or len(v) > 120:
        return ""

    return v


def extract_sender(text: str) -> str:
    # Try header styles first (FROM / FM)
    for pat in FROM_PATTERNS:
        m = re.search(pat, text)
        if m:
            cand = _clean_header_value(m.group(1))
            if cand:
                return cand

    # Letter signature fallback: find a sign-off line then take next non-empty line as name
    lines = text.splitlines()
    for i, line in enumerate(lines[:-2]):
        if re.search(SIGN_PAT, line):
            # look ahead for a name line
            for j in range(i+1, min(i+6, len(lines))):
                name = lines[j].strip()
                if name:
                    name = _clean_header_value(name)
                    if name:
                        return name
    return ""


def extract_recipient(text: str) -> str:
    # Try header styles first (TO / ATTN)
    for pat in TO_PATTERNS:
        m = re.search(pat, text)
        if m:
            cand = _clean_header_value(m.group(1))
            if cand:
                return cand

    # Letter fallback: Dear X,
    m = re.search(DEAR_PAT, text)
    if m:
        cand = _clean_header_value(m.group(1))
        if cand:
            return cand

    return ""


def is_letter_like(text: str) -> bool:
    lower = text.lower()[:3000]
    return ("dear " in lower) and ("sincerely" in lower or "yours truly" in lower or "respectfully" in lower)


# ENTITY EXTRACTION
def extract_entities(doc) -> Tuple[List[str], List[str], List[str]]:
    persons, orgs, locs = [], [], []
    for ent in doc.ents:
        t = ent.text.strip()
        if not t:
            continue
        if ent.label_ == "PERSON" and len(t.split()) <= 5:
            persons.append(t)
        elif ent.label_ == "ORG":
            orgs.append(t)
        elif ent.label_ in ("GPE", "LOC"):
            locs.append(t)
    return persons, orgs, locs


# SENTIMENT, EVENTS, KEYWORDS, STATS
def sentiment(text: str):
    blob = TextBlob(text)
    return blob.sentiment.polarity, blob.sentiment.subjectivity


def keyword_counts(text: str):
    lower = text.lower()
    return {k: lower.count(k) for k in JFK_KEYWORDS if k in lower}


def jfk_events(text: str):
    lower = text.lower()
    return ", ".join(sorted({p for p in JFK_EVENT_PHRASES if p in lower}))


def text_statistics(text: str):
    words = text.split()
    word_count = len(words)

    sentences = re.split(r"[.!?]", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    avg_word_len = (sum(len(w) for w in words) / word_count) if word_count else 0
    avg_sentence_length = (sum(len(s.split()) for s in sentences) / len(sentences)) if sentences else 0
    ttr = (len(set(words)) / word_count) if word_count else 0

    try:
        fre = textstat.flesch_reading_ease(text)
        grade = textstat.flesch_kincaid_grade(text)
        syll = textstat.syllable_count(text)
    except:
        fre = grade = syll = 0

    punct = {p: text.count(p) for p in string.punctuation}

    return {
        "word_count": word_count,
        "avg_word_length": avg_word_len,
        "avg_sentence_length": avg_sentence_length,
        "type_token_ratio": ttr,
        "flesch_reading_ease": fre,
        "flesch_kincaid_grade": grade,
        "total_syllables": syll,
        "punctuation_counts": json.dumps(punct),
        "punctuation_count": sum(punct.values()),
    }


# RELATIONSHIPS
def build_relations(file, persons, orgs, locs, sender, recipient):
    rels = []

    persons = list(dict.fromkeys(persons))
    orgs    = list(dict.fromkeys(orgs))
    locs    = list(dict.fromkeys(locs))

    # Person ↔ Person
    for a, b in combinations(persons, 2):
        rels.append({
            "file": file,
            "source_text": a,
            "source_type": "PERSON",
            "target_text": b,
            "target_type": "PERSON",
            "relation": "cooccurs",
        })

    # Person ↔ Org
    for p in persons:
        for o in orgs:
            rels.append({
                "file": file,
                "source_text": p,
                "source_type": "PERSON",
                "target_text": o,
                "target_type": "ORG",
                "relation": "mentions",
            })

    # Org ↔ Org
    for a, b in combinations(orgs, 2):
        rels.append({
            "file": file,
            "source_text": a,
            "source_type": "ORG",
            "target_text": b,
            "target_type": "ORG",
            "relation": "cooccurs",
        })

    # Person ↔ Location
    for p in persons:
        for loc in locs:
            rels.append({
                "file": file,
                "source_text": p,
                "source_type": "PERSON",
                "target_text": loc,
                "target_type": "LOCATION",
                "relation": "location_link",
            })

    # Sender → Recipient (NOW MUCH MORE LIKELY TO FIRE)
    if sender and recipient:
        rels.append({
            "file": file,
            "source_text": sender,
            "source_type": "SENDER",
            "target_text": recipient,
            "target_type": "RECIPIENT",
            "relation": "sender_to_recipient",
        })

    return rels


# MAIN DOCUMENT PROCESSOR
def process_file(path: Path):
    text = load_text(path)
    if not text:
        return None, [], []

    file = path.name
    doc = nlp(text)

    persons, orgs, locs = extract_entities(doc)

    sender    = extract_sender(text)
    recipient = extract_recipient(text)
    letter    = is_letter_like(text)
    doc_type  = guess_doc_type(text)

    dates     = extract_dates(text, doc)
    numbers   = extract_numbers(text)

    pol, subj = sentiment(text)
    events    = jfk_events(text)
    kfreq     = keyword_counts(text)
    stats     = text_statistics(text)

    doc_row = {
        "file": file,
        "doc_type": doc_type,
        "is_letter_like": letter,
        "sender": sender or "",
        "recipient": recipient or "",
        "dates": dates,
        "numbers": numbers,
        "jfk_events": events,
        "sentiment_polarity": pol,
        "sentiment_subjectivity": subj,
        "keyword_frequency": json.dumps(kfreq),
        "total_keywords": sum(kfreq.values()),
        "top_keyword": max(kfreq, key=kfreq.get) if kfreq else "",
        "full_text": text,
    }
    doc_row.update(stats)

    entities = []
    def add(ents, label):
        counts = {}
        for e in ents:
            counts[e] = counts.get(e, 0) + 1
        for e, n in counts.items():
            entities.append({
                "file": file,
                "entity_text": e,
                "entity_label": label,
                "count_in_doc": n,
            })

    add(persons, "PERSON")
    add(orgs,    "ORG")
    add(locs,    "LOCATION")

    relations = build_relations(file, persons, orgs, locs, sender, recipient)
    return doc_row, entities, relations


# MAIN
def main():
    if not TXT_DIR.exists():
        print(f"❌ Missing directory: {TXT_DIR}")
        return

    files = sorted(TXT_DIR.glob("*.txt"))

    DOC_FIELDS = [
        "file", "doc_type", "is_letter_like",
        "sender", "recipient",
        "dates", "numbers", "jfk_events",
        "sentiment_polarity", "sentiment_subjectivity",
        "keyword_frequency", "total_keywords", "top_keyword",
        "full_text",
        "word_count", "avg_word_length", "avg_sentence_length",
        "type_token_ratio", "flesch_reading_ease",
        "flesch_kincaid_grade", "total_syllables",
        "punctuation_counts", "punctuation_count"
    ]

    ENT_FIELDS = ["file", "entity_text", "entity_label", "count_in_doc"]
    REL_FIELDS = ["file", "source_text", "source_type", "target_text", "target_type", "relation"]

    with DOCS_CSV.open("w", newline="", encoding="utf-8") as f_docs, \
         ENTITIES_CSV.open("w", newline="", encoding="utf-8") as f_ent, \
         RELATIONS_CSV.open("w", newline="", encoding="utf-8") as f_rel:

        docs_w = csv.DictWriter(f_docs, fieldnames=DOC_FIELDS)
        ents_w = csv.DictWriter(f_ent, fieldnames=ENT_FIELDS)
        rels_w = csv.DictWriter(f_rel, fieldnames=REL_FIELDS)

        docs_w.writeheader()
        ents_w.writeheader()
        rels_w.writeheader()

        for p in tqdm(files, desc="Structuring cleaned files"):
            doc_row, ents, rels = process_file(p)
            if not doc_row:
                continue

            docs_w.writerow(doc_row)
            for e in ents:
                ents_w.writerow(e)
            for r in rels:
                rels_w.writerow(r)

    print("\n🎉 Structured extraction complete!")
    print("📄 Docs CSV      →", DOCS_CSV)
    print("🔤 Entities CSV  →", ENTITIES_CSV)
    print("🕸 Relations CSV →", RELATIONS_CSV)


if __name__ == "__main__":
    main()
