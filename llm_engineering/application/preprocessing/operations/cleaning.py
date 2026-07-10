# What: Import Python's built-in regex module.
# Why: All text cleaning operations here use regular expression substitution (re.sub),
#      which lets us describe complex character patterns in a compact, readable way.
import re


def clean_text(text: str) -> str:
    """What: Normalise raw text into a clean, embedding-friendly string by removing
    noise characters and collapsing whitespace.
    Why: Raw scraped text (from HTML, Markdown, code repositories, etc.) contains
    special characters, Unicode symbols, HTML entities, and inconsistent spacing
    that add noise to embeddings without contributing semantic meaning. Cleaning
    before embedding improves vector quality and retrieval accuracy.
    """
    # What: Replace every character that is NOT a word character (\w), whitespace (\s),
    #       or common sentence punctuation (. , ! ?) with a single space.
    # Why: Characters like @, #, *, ^, |, {, }, backticks, emoji, etc. are common in
    #      scraped web/code content but carry no semantic signal for NLP embeddings.
    #      Replacing them with a space (rather than deleting) prevents adjacent words
    #      from being accidentally merged (e.g. "word#word" → "word word" not "wordword").
    text = re.sub(r"[^\w\s.,!?]", " ", text)

    # What: Collapse any sequence of one or more whitespace characters into a single space.
    # Why: The previous substitution can produce runs of multiple spaces wherever noisy
    #      characters appeared. Collapsing these makes downstream length calculations
    #      (e.g. chunk_size, min_length) accurate and eliminates tokeniser artefacts
    #      from multiple consecutive whitespace tokens.
    text = re.sub(r"\s+", " ", text)

    # What: Strip leading and trailing whitespace from the final string.
    # Why: After the substitutions above the string may still start or end with a space
    #      (if the original text began/ended with a noisy character). Stripping ensures
    #      the returned string has no accidental padding, keeping stored content clean.
    return text.strip()
