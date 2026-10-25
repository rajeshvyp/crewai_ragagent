
"""
search_knowledge.py

Purpose:
1. Load the Tamil Jothidam text chunks.
2. Match Tamil and English questions against relevant passages.
3. Give priority to passages containing the actual answer.
4. Print a short, relevant excerpt instead of the entire chunk.

This is a simple keyword-based retrieval test.
It does not call the OpenAI API yet.
"""

import json
import re
import unicodedata
from pathlib import Path

# Find chunks.json in the same folder as this Python file.
PROJECT_FOLDER = Path(__file__).resolve().parent
CHUNKS_FILE = PROJECT_FOLDER / "chunks.json"


def normalize_text(text):
    """Normalize Unicode and whitespace for more reliable matching."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u200b", "").replace("\u200c", "")
    return re.sub(r"\s+", " ", text).casefold().strip()


def load_chunks():
    """Load the text chunks created by prepare_chunks.py."""
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(
            f"Cannot find {CHUNKS_FILE}. Run prepare_chunks.py first."
        )

    with CHUNKS_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)


def get_question_type(question):
    """Identify common questions in either Tamil or English."""
    q = normalize_text(question)

    # Questions asking for the definition of Jothidam.
    if (
        "ஜோதிடம்" in q
        and any(word in q for word in ["என்ன", "என்றால்", "விளக்கம்"])
    ) or any(
        phrase in q
        for phrase in [
            "what is jothidam",
            "what is astrology",
            "define astrology",
        ]
    ):
        return "definition"

    # Questions asking for the nine planets.
    if (
        "கிரகங்கள்" in q
        or "கிரகங்களை" in q
        or "planets" in q
    ):
        return "planets"

    return "general"


def extract_relevant_excerpt(text, question_type):
    """Return the relevant section rather than the entire chunk."""
    normalized = normalize_text(text)

    if question_type == "definition":
        # Locate the definition in the original text.
        match = re.search(
            r"ஜோதிடம்.{0,30}என்றால்.{0,20}ஜோதிடம்.{0,30}என்பது",
            normalized,
        )

        if match:
            start = match.start()
            # The definition and explanation are usually in this paragraph.
            return text[max(0, start - 20): start + 700].strip()

        # Fallback if OCR spacing prevents the pattern from matching.
        marker = normalized.find("நட்சத்திர மொழி")
        if marker >= 0:
            start = max(0, marker - 400)
            return text[start:start + 600].strip()

    elif question_type == "planets":
        # Find the section where the book introduces the nine planets.
        marker = normalized.find("சூரியன்")
        if marker >= 0:
            start = max(0, marker - 180)
            return text[start:start + 650].strip()

    # General questions: return a bounded excerpt.
    return text[:700].strip()


def search_knowledge(question, chunks):
    """Score chunks and return one best matching result."""
    q = normalize_text(question)
    question_type = get_question_type(question)

    # Special matching rules for the two questions we are testing.
    if question_type == "definition":
        keywords = [
            "ஜோதிடம்",
            "நட்சத்திரங்கள்",
            "வருங்காலத்தை",
            "நட்சத்திர மொழி",
        ]
    elif question_type == "planets":
        keywords = [
            "9 கிரகங்கள்",
            "சூரியன்",
            "சந்திரன்",
            "செவ்வாய்",
            "புதன்",
            "குரு",
            "சுக்கிரன்",
            "சனி",
            "ராகு",
            "கேது",
        ]
    else:
        # Remove common question words for a basic fallback search.
        stop_words = {
            "what", "which", "are", "the", "is", "and", "for",
            "என்ன", "எவை", "எது", "என்றால்",
        }
        keywords = [
            word for word in re.findall(r"[^\W_]+", q)
            if len(word) >= 3 and word not in stop_words
        ]

    best_chunk = None
    best_score = 0

    for chunk in chunks:
        text = normalize_text(chunk.get("text", ""))

        # Count matching keywords in this chunk.
        score = sum(
            1 for keyword in keywords
            if normalize_text(keyword) in text
        )

        # Prefer a chunk that matches more of the requested information.
        if score > best_score:
            best_score = score
            best_chunk = chunk

    if best_chunk is None:
        return None

    excerpt = extract_relevant_excerpt(
        best_chunk["text"], question_type
    )

    return {
        "chunk_id": best_chunk.get("chunk_id"),
        "page": best_chunk.get("page"),
        "score": best_score,
        "text": excerpt,
    }


def main():
    """Run an interactive search test."""
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} text chunks.")
    print("Ask a question in Tamil or English. Type 'exit' to stop.")

    while True:
        question = input("\nQuestion: ").strip()

        if question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        if not question:
            print("Please enter a question.")
            continue

        result = search_knowledge(question, chunks)

        if result is None:
            print("No matching passage found in the documents.")
            continue

        print("\n--- Relevant passage ---")
        print(f"Chunk: {result['chunk_id']}")
        print(f"Page: {result['page']}")
        print(f"Keyword score: {result['score']}")
        print(result["text"])


if __name__ == "__main__":
    main()
