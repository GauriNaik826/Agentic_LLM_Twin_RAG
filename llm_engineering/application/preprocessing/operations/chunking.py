# What: Import Python's built-in regex module.
# Why: Used in chunk_article() to split text on sentence boundaries using a
#      look-behind regex, which is more accurate than splitting on "." alone
#      (avoids splitting on abbreviations like "Dr." or "e.g.").
import re

# What: Import two LangChain text splitters.
# Why: RecursiveCharacterTextSplitter splits on paragraph/newline boundaries first
#      (structurally aware); SentenceTransformersTokenTextSplitter then ensures
#      each resulting piece fits within the embedding model's token limit.
#      Using them in sequence gives both structural coherence and token safety.
from langchain.text_splitter import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter

# What: Import the singleton embedding model to read its constraints.
# Why: The embedding model has a fixed maximum token input length (e.g. 256 or 512 tokens).
#      Chunks exceeding this limit are silently truncated by the model, losing content.
#      Reading the limit here ensures every chunk is guaranteed to fit.
from llm_engineering.application.networks import EmbeddingModelSingleton

# What: Load the singleton embedding model at module level.
# Why: We only need model metadata (max_input_length, model_id) here — not inference.
#      Loading at module level means this metadata is read once at import time rather
#      than on every chunk_text() call, avoiding repeated singleton lookups.
embedding_model = EmbeddingModelSingleton()


def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list[str]:
    """What: Split text into chunks using a two-pass approach:
       1. Split on paragraph boundaries (structural split).
       2. Further split each paragraph chunk by token count (model-aware split).
    Why: A single character-count split can cut across paragraph boundaries, producing
    incoherent chunks. A single token split ignores document structure entirely.
    The two-pass approach first respects document structure, then ensures each piece
    fits within the embedding model's token window. Used for posts and repositories
    where content has clear paragraph/newline structure.
    """
    # What: First pass — split the text on double newlines (paragraph breaks).
    # Why: RecursiveCharacterTextSplitter with separator="\n\n" keeps paragraphs intact as
    #      the primary unit. chunk_size caps the character length of each paragraph chunk.
    #      chunk_overlap=0 here because we rely on the token splitter (second pass) to add
    #      the actual overlap between final chunks.
    character_splitter = RecursiveCharacterTextSplitter(separators=["\n\n"], chunk_size=chunk_size, chunk_overlap=0)
    text_split_by_characters = character_splitter.split_text(text)

    # What: Configure the token-aware splitter using the embedding model's own tokeniser.
    # Why: Different embedding models use different tokenisers (BERT-style, sentence-piece, etc.)
    #      and have different token-to-character ratios. Using the model's actual tokeniser
    #      (via model_name) and its exact max_input_length guarantees no chunk exceeds the
    #      model's context window, preventing silent content truncation during embedding.
    #      chunk_overlap carries over tokens between adjacent chunks to preserve sentence
    #      context at chunk boundaries.
    token_splitter = SentenceTransformersTokenTextSplitter(
        chunk_overlap=chunk_overlap,
        tokens_per_chunk=embedding_model.max_input_length,
        model_name=embedding_model.model_id,
    )

    # What: Apply the token splitter to every paragraph chunk from the first pass.
    # Why: A single paragraph may still exceed the token limit (e.g. a very long README
    #      section). Processing each paragraph independently with extend() flattens the
    #      results into one final list of token-safe chunks.
    chunks_by_tokens = []
    for section in text_split_by_characters:
        # What: extend() appends each token chunk individually rather than as a sub-list.
        # Why: Keeps chunks_by_tokens a flat list so the caller receives a simple
        #      list[str] regardless of how many sub-chunks each paragraph produced.
        chunks_by_tokens.extend(token_splitter.split_text(section))

    return chunks_by_tokens


def chunk_document(text: str, min_length: int, max_length: int) -> list[str]:
    """What: Alias that delegates directly to chunk_article().
    Why: Some callers use the generic name "chunk_document" rather than the
    content-specific "chunk_article". Having this alias keeps both call sites
    working without duplicating logic.
    """
    return chunk_article(text, min_length, max_length)


def chunk_article(text: str, min_length: int, max_length: int) -> list[str]:
    """What: Split long-form article text into semantically coherent chunks by accumulating
    sentences until a max_length character cap is hit, then emitting chunks that
    meet a min_length quality threshold.
    Why: Articles have natural sentence structure that should be respected —
    cutting mid-sentence produces incoherent chunks that embed poorly and hurt
    RAG retrieval quality. This function accumulates whole sentences, so every
    chunk starts and ends on a sentence boundary.
    min_length filters out very short trailing fragments that would be too sparse
    to produce a meaningful embedding.
    """
    # What: Split the article into individual sentences using a regex look-behind.
    # Why: A simple split on "." would incorrectly break on abbreviations like "Dr.",
    #      "e.g.", or "U.S.A.". The look-behind pattern:
    #        (?<!\w\.\w.)  — don't split after a word.letter.word.letter pattern (abbreviations)
    #        (?<![A-Z][a-z]\.)  — don't split after Title-case abbreviations (e.g. "Mr.")
    #        (?<=\.|\?|\!)  — DO split after a sentence-ending punctuation mark
    #      followed by whitespace \s. Together this gives clean sentence boundaries.
    sentences = re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s", text)

    # What: Initialise the output list and the running sentence accumulator.
    # Why: We build chunks by accumulating sentences until the max_length cap is hit,
    #      so we need a mutable buffer (current_chunk) and a results collector (extracts).
    extracts = []
    current_chunk = ""

    for sentence in sentences:
        # What: Strip leading/trailing whitespace from the sentence.
        # Why: The regex split may leave whitespace artifacts at the start or end
        #      of sentences; stripping keeps length calculations accurate.
        sentence = sentence.strip()

        # What: Skip empty sentences produced by consecutive punctuation or whitespace.
        # Why: Empty strings would add nothing to the chunk but would still trigger
        #      the length check, potentially causing premature chunk emission.
        if not sentence:
            continue

        # What: If adding this sentence keeps the chunk within the max_length cap, accumulate it.
        # Why: We always prefer larger chunks (up to max_length) because more context
        #      produces richer embeddings and better RAG retrieval. Only split when
        #      adding the next sentence would exceed the cap.
        if len(current_chunk) + len(sentence) <= max_length:
            # What: Append the sentence with a trailing space as natural word separator.
            # Why: Sentences need a space between them when joined; the trailing space
            #      is stripped by .strip() when the chunk is emitted.
            current_chunk += sentence + " "
        else:
            # What: max_length would be exceeded — emit current_chunk if it meets min_length.
            # Why: min_length ensures we don't store very short chunks (e.g. a single
            #      short sentence) that would produce low-quality sparse embeddings.
            if len(current_chunk) >= min_length:
                extracts.append(current_chunk.strip())

            # What: Start a new chunk with the sentence that didn't fit.
            # Why: The overflowing sentence becomes the first sentence of the next chunk
            #      rather than being discarded, so no content is lost.
            current_chunk = sentence + " "

    # What: After the loop, emit any remaining text that meets the min_length threshold.
    # Why: The loop only emits a chunk when the next sentence overflows max_length.
    #      The final accumulated sentences never trigger that condition, so we must
    #      emit them explicitly here to avoid silently dropping the article's closing content.
    if len(current_chunk) >= min_length:
        extracts.append(current_chunk.strip())

    return extracts
