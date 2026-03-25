"""
custom_article.py — Generic fallback article crawler
-----------------------------------------------------

This module provides `CustomArticleCrawler`, a lightweight crawler used for
any article URL that does not have a domain-specific implementation (e.g.,
Medium or LinkedIn). It acts as the safety-net in the dispatcher system:
when no registered pattern matches, the dispatcher falls back to this class.

Why a generic fallback?
- Most public article pages share a predictable structure (HTML → readable
  text), so a single general-purpose crawler covers the long tail of
  unsupported domains without requiring a bespoke class for each.
- It keeps the ingestion pipeline capable of processing new sources
  immediately, while specialized crawlers can be added incrementally for
  sites that need extra handling (auth, JS rendering, pagination, etc.).
"""

from urllib.parse import urlparse  # Parse a URL into its component parts (scheme, netloc, path…)

# AsyncHtmlLoader fetches one or more URLs asynchronously and returns raw HTML
# wrapped in LangChain Document objects. Using LangChain's loader avoids
# writing boilerplate aiohttp/requests code and provides consistent Document
# objects that plug directly into the transformer below.
from langchain_community.document_loaders import AsyncHtmlLoader

# Html2TextTransformer converts raw HTML Documents into plain-text Documents.
# Using a transformer (rather than manually parsing HTML with BeautifulSoup)
# keeps the code concise and leverages the html2text library's robust handling
# of nested tags, links, and whitespace normalization.
from langchain_community.document_transformers.html2text import Html2TextTransformer
from loguru import logger  # Structured logging used consistently across the codebase

from llm_engineering.domain.documents import ArticleDocument  # MongoDB ODM model for articles

from .base import BaseCrawler  # Abstract interface all crawlers must satisfy


class CustomArticleCrawler(BaseCrawler):
    """Generic article crawler used as a fallback for unregistered domains.

    Responsibilities:
    - Deduplicate: skip URLs already present in MongoDB.
    - Fetch: download raw HTML asynchronously.
    - Transform: convert HTML to structured plain text.
    - Persist: build and save an `ArticleDocument` with content and metadata.

    Design notes:
    - Inherits `BaseCrawler` to comply with the `extract()` contract, making
      it interchangeable with any other crawler in the dispatcher system.
    - Intentionally stateless beyond `model`; no Selenium, no credential
      management — this keeps it fast and dependency-light.
    """

    # Associate this crawler with the `ArticleDocument` ODM model. The
    # dispatcher and any orchestration code can inspect `crawler.model` to
    # know which MongoDB collection / document schema will be written.
    model = ArticleDocument

    def __init__(self) -> None:
        # Delegate to BaseCrawler.__init__. Even though BaseCrawler is
        # currently a no-op, explicit super().__init__() future-proofs this
        # class against base-class changes and keeps MRO explicit.
        super().__init__()

    def extract(self, link: str, **kwargs) -> None:
        """Download, parse, and persist a single article from `link`.

        Parameters
        ----------
        link : str
            Fully-qualified URL of the article to crawl.
        **kwargs : dict
            Must contain a `user` key holding the author/user domain object.
            Author metadata is stored with the document for traceability.
        """

        # --- Deduplication guard -------------------------------------------------
        # Query MongoDB for an existing document with this URL before doing any
        # network I/O. This prevents re-downloading and re-storing the same
        # article if the pipeline is re-run (e.g., after a partial failure).
        old_model = self.model.find(link=link)
        if old_model is not None:
            logger.info(f"Article already exists in the database: {link}")

            # Return early; nothing left to do for an already-ingested URL.
            return

        logger.info(f"Starting scrapping article: {link}")

        # --- Fetch HTML ----------------------------------------------------------
        # Wrap the URL in a list because AsyncHtmlLoader accepts a batch of
        # URLs, allowing future extension to multi-URL fetching without code
        # changes. It fetches HTML asynchronously (non-blocking I/O), which
        # is beneficial when called in a concurrent pipeline context.
        loader = AsyncHtmlLoader([link])
        docs = loader.load()  # Returns a list of LangChain Document objects with raw HTML content

        # --- Convert HTML to plain text ------------------------------------------
        # Html2TextTransformer processes the HTML Document list and returns a
        # new list of Documents where `page_content` is clean Markdown-like
        # plain text. Metadata fields like `title`, `description`, and
        # `language` are extracted from HTML <meta> tags during transformation.
        html2text = Html2TextTransformer()
        docs_transformed = html2text.transform_documents(docs)

        # We fetched exactly one URL, so we take the single result. Using [0]
        # is safe here; if the load failed, docs would be empty and we'd see
        # an IndexError, signalling a real problem rather than silently storing
        # an empty document.
        doc_transformed = docs_transformed[0]

        # --- Build structured content dict ---------------------------------------
        # Assemble a dictionary that maps semantic field names to extracted
        # values. Using a dict (rather than positional fields) makes the
        # schema self-documenting and easy to extend with new fields later.
        content = {
            # The page <title> tag — primary identifier for the article.
            "Title": doc_transformed.metadata.get("title"),

            # The <meta name="description"> tag — often the subtitle or
            # byline. `.get()` safely returns None if the tag is absent.
            "Subtitle": doc_transformed.metadata.get("description"),

            # Full body text produced by the HTML-to-text transformation.
            "Content": doc_transformed.page_content,

            # Detected/declared language (e.g., "en") for downstream
            # filtering or multilingual model routing.
            "language": doc_transformed.metadata.get("language"),
        }

        # --- Determine platform from URL -----------------------------------------
        # Parse the URL to extract `netloc` (the network location, i.e., the
        # domain: "example.com"). This is stored as `platform` so documents
        # can be grouped or filtered by source site without storing the full URL.
        parsed_url = urlparse(link)
        platform = parsed_url.netloc

        # --- Create and save the document ----------------------------------------
        # `user` carries author identity, passed in by the orchestration layer.
        # We store author_id and author_full_name together so the document is
        # fully self-contained for attribution and auditing.
        user = kwargs["user"]
        instance = self.model(
            content=content,
            link=link,           # Original URL — primary key for deduplication
            platform=platform,   # Derived domain — useful for filtering by source
            author_id=user.id,
            author_full_name=user.full_name,
        )
        # Persist to MongoDB. Downstream feature-engineering and RAG pipelines
        # will query this collection to retrieve article documents.
        instance.save()

        logger.info(f"Finished scrapping custom article: {link}")
