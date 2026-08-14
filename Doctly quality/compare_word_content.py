from pathlib import Path
import csv
import re
from collections import Counter

BASE_DIR = Path("/Users/maysounbrghl/Desktop/Quality/man_transcribed")
OTHER_DIR = Path("/Users/maysounbrghl/Desktop/Quality/doctly")
OUTPUT_CSV = "word_content_comparison.csv"


def read_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def extract_words(text: str) -> list[str]:
    text = text.lower()

    # remove markdown links/images
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", text)

    # remove everything except letters/numbers
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # normalize spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text.split()


def compute_similarity(words1: list[str], words2: list[str]):
    counter1 = Counter(words1)
    counter2 = Counter(words2)

    common = counter1 & counter2   # intersection
    union = counter1 | counter2    # union

    common_count = sum(common.values())
    total_count = sum(union.values())

    similarity = (common_count / total_count) * 100 if total_count > 0 else 100

    # differences
    missing = list((counter1 - counter2).elements())
    extra = list((counter2 - counter1).elements())

    return similarity, missing, extra


def main():
    if not BASE_DIR.exists():
        print(f"Folder not found: {BASE_DIR}")
        return

    if not OTHER_DIR.exists():
        print(f"Folder not found: {OTHER_DIR}")
        return

    base_files = {f.name: f for f in BASE_DIR.glob("*.md")}
    other_files = {f.name: f for f in OTHER_DIR.glob("*.md")}

    common_names = sorted(set(base_files.keys()) & set(other_files.keys()))

    if not common_names:
        print("No matching .md files found.")
        return

    results = []
    total_score = 0.0

    print("\nComparing files (word content only, order ignored):\n")

    for name in common_names:
        words1 = extract_words(read_file(base_files[name]))
        words2 = extract_words(read_file(other_files[name]))

        score, missing, extra = compute_similarity(words1, words2)
        total_score += score

        print(f"{name}: {score:.2f}%")

        # show small differences (avoid flooding)
        if missing:
            print(f"  Missing words (in base only): {set(missing)}")

        if extra:
            print(f"  Extra words (in doctly): {set(extra)}")

        results.append((name, round(score, 2)))

    overall = total_score / len(common_names)

    print("\n--- Summary ---")
    print(f"Files compared: {len(common_names)}")
    print(f"Overall similarity: {overall:.2f}%")

    # save CSV
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "word_content_similarity_percent"])
        writer.writerows(results)
        writer.writerow([])
        writer.writerow(["overall_average", round(overall, 2)])

    print(f"\nResults saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()