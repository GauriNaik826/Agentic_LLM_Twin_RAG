"""
medium.py — Selenium-based crawler for Medium articles
-------------------------------------------------------

WHY A DEDICATED MEDIUM CRAWLER?
--------------------------------
Medium renders its articles using JavaScript — the full article body, title,
and subtitle are not present in the raw HTML response; they are injected into
the DOM after JS executes. A simple HTTP request (like `AsyncHtmlLoader` in
`CustomArticleCrawler`) would only capture the shell of the page without any
article content. `MediumCrawler` solves this by using a real Chrome browser
(via Selenium) that executes JavaScript and waits for the DOM to fully load.

Additionally, Medium applies CSS classes like `pw-post-title` and
`pw-subtitle-paragraph` to its article elements. BeautifulSoup can target
these predictable class names directly, producing cleaner structured output
(Title, Subtitle, Content) compared to the generic HTML-to-text conversion
used by `CustomArticleCrawler`.
"""

# BeautifulSoup parses the fully-rendered HTML from the Selenium driver and
# lets us query the DOM using CSS class selectors. This is more reliable than
# regex over raw HTML because it understands the tree structure of the page.
from bs4 import BeautifulSoup
from loguru import logger  # Structured logging consistent with the rest of the codebase

# The persistence model — specifies that this crawler produces ArticleDocuments
# stored in the MongoDB "articles" collection.
from llm_engineering.domain.documents import ArticleDocument

# BaseSeleniumCrawler provides: Chrome WebDriver setup, headless browser
# configuration, and the scroll_page() helper for triggering lazy loading.
from .base import BaseSeleniumCrawler


class MediumCrawler(BaseSeleniumCrawler):
    """Crawls a single Medium article and persists it as an ArticleDocument.

    Inherits Chrome WebDriver setup and `scroll_page()` from
    `BaseSeleniumCrawler`. Overrides `set_extra_driver_options()` to use a
    specific Chrome profile and implements `extract()` with Medium-specific
    DOM parsing logic.
    """

    # Declare the target persistence model. Downstream code and the dispatcher
    # can inspect `MediumCrawler.model` to know which collection will be written.
    model = ArticleDocument

    def set_extra_driver_options(self, options) -> None:
        """Add Medium-specific Chrome options before the driver is started.

        Using a named Chrome profile ("Profile 2") allows the browser to
        reuse saved cookies and session state from a previous login. This is
        important for Medium because some articles are paywalled or require
        authentication to view the full content. By persisting the profile,
        we avoid having to log in on every crawl run.

        This method is called by `BaseSeleniumCrawler.__init__()` before
        `webdriver.Chrome(options=options)` is invoked, so the profile is
        active from the very first page load.
        """
        # r"..." (raw string) avoids accidental backslash interpretation on
        # Windows paths; it's a safe habit even on macOS/Linux.
        options.add_argument(r"--profile-directory=Profile 2")

    def extract(self, link: str, **kwargs) -> None:
        """Navigate to a Medium article URL, parse it, and save to MongoDB.

        Steps:
        1. Deduplicate — skip if the article already exists in the DB.
        2. Load the page in the headless Chrome browser.
        3. Scroll to trigger lazy-loaded content.
        4. Parse the rendered DOM with BeautifulSoup.
        5. Extract Title, Subtitle, and full-page text.
        6. Close the browser tab.
        7. Create and save an ArticleDocument.
        """

        # --- Deduplication guard -------------------------------------------------
        # Before doing any browser I/O, check MongoDB for an existing record
        # with this URL. Re-processing the same article wastes time and
        # produces duplicate entries in the data warehouse.
        old_model = self.model.find(link=link)
        if old_model is not None:
            logger.info(f"Article already exists in the database: {link}")

            # Early return — nothing to do; the article is already ingested.
            return

        logger.info(f"Starting scrapping Medium article: {link}")

        # --- Load the page -------------------------------------------------------
        # Instruct the Selenium-controlled Chrome browser to navigate to the
        # article URL. Because the browser executes JavaScript, Medium's
        # React/Next.js frontend will render the full article into the DOM.
        self.driver.get(link)

        # --- Scroll to load lazy content ----------------------------------------
        # Medium uses infinite-scroll and lazy loading for images and some
        # content sections. `scroll_page()` (inherited from BaseSeleniumCrawler)
        # repeatedly scrolls to the bottom of the page and waits, ensuring the
        # complete article body has been rendered before we read the DOM.
        self.scroll_page()

        # --- Parse the rendered DOM ---------------------------------------------
        # `self.driver.page_source` returns the *current* rendered HTML string
        # — this is the post-JavaScript DOM, not the original server response.
        # BeautifulSoup parses it into a navigable tree. We use "html.parser"
        # (Python's built-in parser) to avoid an extra lxml dependency.
        soup = BeautifulSoup(self.driver.page_source, "html.parser")

        # Medium consistently applies the CSS class "pw-post-title" to the
        # article's main <h1> heading. `find_all` returns a list; we expect
        # exactly one, but using find_all guards against edge cases where the
        # DOM has been modified or the article hasn't fully loaded.
        title = soup.find_all("h1", class_="pw-post-title")

        # Medium applies "pw-subtitle-paragraph" to the <h2> subtitle element.
        # Same defensive find_all pattern as above.
        subtitle = soup.find_all("h2", class_="pw-subtitle-paragraph")

        # --- Build the structured content dict ----------------------------------
        # We store a dict rather than a flat string so downstream tasks
        # (feature engineering, RAG chunking) can operate on individual
        # fields without re-parsing the raw text.
        data = {
            # `.string` extracts the direct text of the tag. The conditional
            # `if title` guards against an empty list (element not found),
            # returning None instead of raising an IndexError.
            "Title": title[0].string if title else None,

            # Same safe access pattern for the subtitle element.
            "Subtitle": subtitle[0].string if subtitle else None,

            # `soup.get_text()` dumps all visible text from the entire page.
            # This is a broad capture — it includes nav, footer, and
            # boilerplate alongside the article body. Cleaning/filtering
            # happens in the feature-engineering pipeline downstream.
            "Content": soup.get_text(),
        }

        # --- Release the browser tab --------------------------------------------
        # `driver.close()` closes the current tab (not the whole browser
        # process). This frees the tab's memory immediately rather than waiting
        # for garbage collection, which is important when crawling many articles
        # sequentially in a single pipeline run.
        self.driver.close()

        # --- Persist the document -----------------------------------------------
        # Retrieve the `user` domain object passed in by the orchestration layer.
        # We attach author identity to the document for attribution and
        # to allow filtering by author in downstream queries.
        user = kwargs["user"]
        instance = self.model(
            platform="medium",       # Hard-coded: all articles from this crawler come from Medium
            content=data,            # The structured dict built above
            link=link,               # Original URL — deduplication key on future runs
            author_id=user.id,
            author_full_name=user.full_name,
        )
        # Write the document to the MongoDB "articles" collection.
        instance.save()

        logger.info(f"Successfully scraped and saved article: {link}")
