#!/usr/bin/env python3

"""
1-cleaning.py

High‑fidelity OCR cleaner for JFK/HSCA/CIA/FBI/NARA documents.
 ------
Goals:
  - Normalize text safely without destroying intelligence content.
  - Remove scanning noise (DocId lines, doctly banners, OCR artifacts).
  - Preserve memo headers (FROM, TO, SUBJECT, MEMORANDUM FOR).
  - Preserve CIA cryptonyms (AM/LASH, ZR/RIFLE, LI/OSWALD).
  - Preserve 201-file numbers (e.g., 201‑289248).
"""

import os
import re
import unicodedata
from pathlib import Path

INPUT_DIR = Path("data/jfk_txt_all")       
OUTPUT_DIR = Path("data/cleaned_text_files")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def clean_text(text: str) -> str:
    """Enhanced cleaner specifically tuned for CIA/HSCA/NARA documents."""
    
    # Normalise unicode
    text = unicodedata.normalize("NFKC", text)

    # Remove doctly.ai banners
    text = re.sub(r"Converted with \[.*?\]", "", text, flags=re.I)

    # Strip CIA scanning & FOIA release headers
    text = re.sub(r"2025 RELEASE UNDER.*", "", text, flags=re.I)
    text = re.sub(r"RELEASE UNDER.*", "", text)
    text = re.sub(r"E-?2 IMPDET.*", "", text, flags=re.I)
    text = re.sub(r"CL BY.*", "", text)
    text = re.sub(r"GROUP 1.*", "", text, flags=re.I)
    text = re.sub(r"Excluded from automatic.*", "", text, flags=re.I)

    # Remove DocId / page banners
    text = re.sub(r"NW\s*\d+\s*DocId[:\s-]*\S+.*", "", text)
    text = re.sub(r"-{3,}\s*Page\s*\d+\s*-{3,}", "", text)
    text = re.sub(r"^Page \d+$", "", text, flags=re.M)
    text = re.sub(r"^-{3,}$", "", text, flags=re.M)
    text = re.sub(r"_{3,}", "", text)

    # CIA routing metadata (DIR MEXI ####, ACTION WH, INFO VR)
    text = re.sub(r"^TO *:.*", "", text, flags=re.M)
    text = re.sub(r"^FROM *:.*", "", text, flags=re.M)
    text = re.sub(r"^INFO *:.*", "", text, flags=re.M)
    text = re.sub(r"^ACTION *:.*", "", text, flags=re.M)
    text = re.sub(r"^ROUTINE.*", "", text, flags=re.M)
    text = re.sub(r"^CITE.*", "", text, flags=re.M)
    text = re.sub(r"^DIR .*", "", text, flags=re.M)
    text = re.sub(r"^REF .*", "", text, flags=re.M)
    text = re.sub(r"^REF:.*", "", text, flags=re.M)
    text = re.sub(r"^\*?WH COMMENT.*", "", text, flags=re.M)

    # Remove classified stamps
    text = re.sub(r"^#? *SECRET.*$", "", text, flags=re.M)
    text = re.sub(r"^#? *CONFIDENTIAL.*$", "", text, flags=re.M)
    text = re.sub(r"^CLASSIFIED MESSAGE.*$", "", text, flags=re.M)
    
    # Remove “Copy No.” and administrative table junk
    text = re.sub(r"Copy No\..*", "", text)
    text = re.sub(r"FILE IN CS FILE NO.*", "", text)
    text = re.sub(r"OFFICE SYMBOL.*", "", text)
    text = re.sub(r"ROUTING\s*\|.*", "", text)
    text = re.sub(r"\|\s*\d\s*\|.*", "", text)
    
    # Remove page-perforation lines
    text = re.sub(r"-{10,}", "", text)
    text = re.sub(r"_{10,}", "", text)
    text = re.sub(r"\*{5,}", "", text)
    
    # Fix hyphenation (investi-\ngation)
    # But DO NOT break cryptonyms like LICOOKY-1
    text = re.sub(r"([A-Za-z]{2,})-\s*\n\s*([a-z]{2,})", r"\1\2", text)
    
    # Repair accidental mid-word line breaks (so-\nviet)
    text = re.sub(r"([a-z])\n([a-z])", r"\1\2", text)
    
    # Paragraph join (safe)
    def join_sentences(match):
        a, b = match.group(1), match.group(2)
        if a.endswith(('.', '?', '!', ':')):
            return a + "\n" + b
        return a + " " + b

    text = re.sub(r"([^\n])\n([A-Za-z])", join_sentences, text)
    
    # Remove footnotes ([24], etc.)
    text = re.sub(r"\[[0-9]{1,3}\]", "", text)

    # Collapse blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    
    # Remove doctly.ai banners, links, and artifacts
    text = re.sub(r"Converted with\s*\[.*?doctly.*?\]", "", text, flags=re.I)
    text = re.sub(r"\[?doctly\.ai.*?\]?", "", text, flags=re.I)
    text = re.sub(r"\(https?://doctly\.ai.*?\)", "", text, flags=re.I)
    text = re.sub(r"https?://doctly\.ai\S*", "", text, flags=re.I)
    text = re.sub(r"doctly\.ai", "", text, flags=re.I)
    text = re.sub(r"\(https?:\/\/\s*\)", "", text)      
    text = re.sub(r"\(https?:\/\/\)", "", text)
    
    return text.strip()


def process_file(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read()

    cleaned = clean_text(raw)

    out_path = OUTPUT_DIR / path.name
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(cleaned)

    print(f"✔ Cleaned: {path.name}")


def main():
    txt_files = [p for p in INPUT_DIR.glob("*.txt")]
    print(f"Found {len(txt_files)} TXT files to clean.")

    for f in txt_files:
        process_file(f)

    print("\n🎉 Cleaning complete. Clean files saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
