
"""
prepare_chunks.py

Purpose:
1. Read OCR-extracted Tamil Jothidam text.
2. Split the text into overlapping chunks.
3. Preserve page markers where possible.
4. Save the chunks to chunks.json for vector indexing.

Input:
    extracted_data_ocr.txt

Output:
    chunks.json
"""

import json
import re
from pathlib import Path

# --------------------------------------------------
# 1. File paths and chunk settings
# --------------------------------------------------

PROJECT_FOLDER = Path(__file__).resolve().parent

INPUT_FILE = PROJECT_FOLDER / "extracted_data_ocr.txt"
OUTPUT_FILE = PROJECT_FOLDER / "chunks.json"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


# --------------------------------------------------
# 2. Read the OCR text
# --------------------------------------------------

def read_source_text():
    """Read the OCR text while preserving Tamil Unicode characters."""

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}\n"
            "Make sure extracted_data_ocr.txt is in the project folder."
        )

    with INPUT_FILE.open("r", encoding="utf-8-sig") as file:
        text = file.read()

    # Normalize line endings and remove null characters.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\x00", "")

    if not text.strip():
        raise ValueError("The OCR text file is empty.")

    return text


# --------------------------------------------------
# 3. Split the text into page sections
# --------------------------------------------------

def split_into_pages(text):
    """
    Split on markers such as:
        --- Page 1 ---
        --- Page 2 ---

    Any text before the first page marker is retained as page 0.
    """

    page_pattern = re.compile(
        r"(?m)^---\s*Page\s+(\d+)\s*---\s*$"
    )

    matches = list(page_pattern.finditer(text))
    pages = []

    if not matches:
        # Keep the entire text if no page markers are found.
        return [{"page": None, "text": text.strip()}]

    # Retain any introductory text before the first marker.
    prefix = text[:matches[0].start()].strip()
    if prefix:
        pages.append({"page": None, "text": prefix})

    for index, match in enumerate(matches):
        start = match.end()
        end = (
            matches[index + 1].start()
            if index + 1 < len(matches)
            else len(text)
        )

        page_number = int(match.group(1))
        page_text = text[start:end].strip()

        if page_text:
            pages.append({
                "page": page_number,
                "text": page_text,
            })

    return pages


# --------------------------------------------------
# 4. Create overlapping chunks
# --------------------------------------------------

def split_into_chunks(text, page_number):
    """
    Split a page's text into chunks.

    Adjacent chunks overlap so that useful context near a
    boundary is less likely to be lost.
    """

    chunks = []
    text = text.strip()

    if not text:
        return chunks

    start = 0

    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))

        # Prefer ending at a newline or space near the target.
        if end < len(text):
            minimum_end = start + int(CHUNK_SIZE * 0.75)

            newline_position = text.rfind("\n", minimum_end, end)
            space_position = text.rfind(" ", minimum_end, end)

            best_boundary = max(newline_position, space_position)

            if best_boundary > start:
                end = best_boundary

        chunk_text = text[start:end].strip()

        if chunk_text:
            chunks.append({
                "page": page_number,
                "text": chunk_text,
            })

        if end >= len(text):
            break

        # Move forward while retaining the overlap.
        next_start = max(0, end - CHUNK_OVERLAP)

        # Safety check to prevent an infinite loop.
        if next_start <= start:
            next_start = end

        start = next_start

    return chunks


# --------------------------------------------------
# 5. Build all chunks and assign IDs
# --------------------------------------------------

def build_chunks(text):
    """Create chunks while keeping their original page numbers."""

    pages = split_into_pages(text)
    all_chunks = []

    for page in pages:
        page_chunks = split_into_chunks(
            page["text"],
            page["page"],
        )

        all_chunks.extend(page_chunks)

    for index, chunk in enumerate(all_chunks):
        chunk["chunk_id"] = index

    return all_chunks


# --------------------------------------------------
# 6. Save the chunks as UTF-8 JSON
# --------------------------------------------------

def save_chunks(chunks):
    """Write chunks.json without escaping Tamil Unicode characters."""

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        json.dump(
            chunks,
            file,
            ensure_ascii=False,
            indent=2,
        )


# --------------------------------------------------
# 7. Main program
# --------------------------------------------------

def main():
    print("Reading OCR text...")
    text = read_source_text()

    print("Splitting text by page...")
    pages = split_into_pages(text)

    print(f"Found {len(pages)} page sections.")

    print("Creating overlapping chunks...")
    chunks = build_chunks(text)

    if not chunks:
        raise ValueError(
            "No chunks were created. Check the OCR text file."
        )

    save_chunks(chunks)

    print("\nChunk preparation completed successfully.")
    print(f"Input file: {INPUT_FILE}")
    print(f"Output file: {OUTPUT_FILE}")
    print(f"Total chunks: {len(chunks)}")
    print(f"Chunk size: {CHUNK_SIZE} characters")
    print(f"Chunk overlap: {CHUNK_OVERLAP} characters")

    # Verify whether the key Tamil definition is preserved.
    search_phrase = "ஜோதிடம் என்றால் என்ன"
    matching_chunks = [
        chunk for chunk in chunks
        if search_phrase in chunk["text"]
    ]

    print(
        f"\nChunks containing '{search_phrase}': "
        f"{len(matching_chunks)}"
    )

    for chunk in matching_chunks[:3]:
        print(
            f"\nChunk ID: {chunk['chunk_id']}, "
            f"Page: {chunk['page']}"
        )
        print(chunk["text"][:500])


if __name__ == "__main__":
    main()
