"""Project Gutenberg downloader and parser for ScatterQA benchmark."""

import logging
import re
from pathlib import Path

from src.utils.config import data_dir

logger = logging.getLogger(__name__)

# Curated novel list: well-known, diverse genres, multiple named characters
# (book_id, title, author)
SELECTED_NOVELS = [
    (1342, "Pride and Prejudice", "Jane Austen"),
    (11, "Alice's Adventures in Wonderland", "Lewis Carroll"),
    (1661, "The Adventures of Sherlock Holmes", "Arthur Conan Doyle"),
    (84, "Frankenstein", "Mary Shelley"),
    (1952, "The Yellow Wallpaper", "Charlotte Perkins Gilman"),
    (98, "A Tale of Two Cities", "Charles Dickens"),
    (2701, "Moby Dick", "Herman Melville"),
    (1260, "Jane Eyre", "Charlotte Brontë"),
    (16328, "Beowulf", "Anonymous"),
    (174, "The Picture of Dorian Gray", "Oscar Wilde"),
    (345, "Dracula", "Bram Stoker"),
    (2554, "Crime and Punishment", "Fyodor Dostoevsky"),
    (1497, "The Republic", "Plato"),
    (76, "Adventures of Huckleberry Finn", "Mark Twain"),
    (5200, "Metamorphosis", "Franz Kafka"),
    (46, "A Christmas Carol", "Charles Dickens"),
    (1080, "A Modest Proposal", "Jonathan Swift"),
    (219, "Heart of Darkness", "Joseph Conrad"),
    (2591, "Grimm's Fairy Tales", "Brothers Grimm"),
    (408, "The Souls of Black Folk", "W.E.B. Du Bois"),
    (4300, "Ulysses", "James Joyce"),
    (1400, "Great Expectations", "Charles Dickens"),
    (58585, "The Trial", "Franz Kafka"),
    (244, "A Study in Scarlet", "Arthur Conan Doyle"),
    (55, "The Wonderful Wizard of Oz", "L. Frank Baum"),
    (2600, "War and Peace", "Leo Tolstoy"),
    (730, "Oliver Twist", "Charles Dickens"),
    (35, "The Time Machine", "H.G. Wells"),
    (43, "The Strange Case of Dr. Jekyll and Mr. Hyde", "R.L. Stevenson"),
    (768, "Wuthering Heights", "Emily Brontë"),
    (1232, "The Prince", "Niccolò Machiavelli"),
    (161, "Sense and Sensibility", "Jane Austen"),
    (158, "Emma", "Jane Austen"),
    (3825, "Persuasion", "Jane Austen"),
    (33, "The Scarlet Letter", "Nathaniel Hawthorne"),
    (135, "Les Misérables", "Victor Hugo"),
    (514, "Little Women", "Louisa May Alcott"),
    (120, "Treasure Island", "Robert Louis Stevenson"),
    (996, "Don Quixote", "Miguel de Cervantes"),
    (1184, "The Count of Monte Cristo", "Alexandre Dumas"),
    (2852, "The Hound of the Baskervilles", "Arthur Conan Doyle"),
    (3207, "Leviathan", "Thomas Hobbes"),
    (766, "David Copperfield", "Charles Dickens"),
    (521, "The Life and Adventures of Robinson Crusoe", "Daniel Defoe"),
    (2542, "A Doll's House", "Henrik Ibsen"),
    (16, "Peter Pan", "J.M. Barrie"),
    (779, "The Adventures of Tom Sawyer", "Mark Twain"),
    (4363, "The Black Cat", "Edgar Allan Poe"),
    (74, "The Adventures of Tom Sawyer", "Mark Twain"),
    (27827, "The Kama Sutra", "Vatsyayana"),
]


def _strip_gutenberg_header_footer(text: str) -> str:
    """Remove Project Gutenberg boilerplate header and footer."""
    # Find start of actual text
    start_markers = [
        "*** START OF THIS PROJECT GUTENBERG",
        "*** START OF THE PROJECT GUTENBERG",
        "***START OF",
    ]
    start_idx = 0
    for marker in start_markers:
        idx = text.find(marker)
        if idx != -1:
            # Skip past the marker line
            nl = text.find("\n", idx)
            start_idx = nl + 1 if nl != -1 else idx + len(marker)
            break

    # Find end of actual text
    end_markers = [
        "*** END OF THIS PROJECT GUTENBERG",
        "*** END OF THE PROJECT GUTENBERG",
        "***END OF",
        "End of Project Gutenberg",
    ]
    end_idx = len(text)
    for marker in end_markers:
        idx = text.find(marker)
        if idx != -1:
            end_idx = idx
            break

    return text[start_idx:end_idx].strip()


def download_gutenberg_text(book_id: int) -> str:
    """Download a book from Project Gutenberg by ID."""
    import requests

    raw = data_dir("raw/gutenberg")
    cache_path = raw / f"{book_id}.txt"
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8", errors="replace")

    # Try multiple URL patterns
    urls = [
        f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt",
        f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt",
        f"https://www.gutenberg.org/files/{book_id}/{book_id}.txt",
    ]

    for url in urls:
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200:
                text = resp.text
                cleaned = _strip_gutenberg_header_footer(text)
                cache_path.write_text(cleaned, encoding="utf-8")
                logger.info("Downloaded book %d (%d chars)", book_id, len(cleaned))
                return cleaned
        except requests.RequestException:
            continue

    raise RuntimeError(f"Failed to download Gutenberg book {book_id}")


def split_into_chapters(text: str) -> list[dict[str, str]]:
    """Split a novel text into chapters."""
    # Common chapter patterns
    pattern = r'\n\s*(CHAPTER\s+[IVXLCDM\d]+\.?[^\n]*|Chapter\s+\d+[^\n]*)\s*\n'
    splits = re.split(pattern, text, flags=re.IGNORECASE)

    if len(splits) <= 1:
        # No chapter headers found — split by large whitespace gaps
        return [{"title": "Full Text", "text": text}]

    chapters = []
    # splits alternates between text-before-match and match
    for i in range(1, len(splits), 2):
        title = splits[i].strip()
        body = splits[i + 1].strip() if i + 1 < len(splits) else ""
        if body:
            chapters.append({"title": title, "text": body})

    return chapters if chapters else [{"title": "Full Text", "text": text}]
