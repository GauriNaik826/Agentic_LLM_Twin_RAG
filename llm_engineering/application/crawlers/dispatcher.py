"""
Crawler dispatcher
------------------

This module implements `CrawlerDispatcher`, a small routing layer that selects
the appropriate crawler implementation for a given URL. The dispatcher keeps
an internal registry which maps normalized domain patterns (regular
expressions) to concrete `BaseCrawler` subclasses. The intent is to decouple
URL/domain detection from crawling logic so new crawlers can be added without
modifying the core dispatching algorithm.

Why this design?
- Extensibility: new crawlers are registered via `register(...)` or the
  convenient builder-style methods like `register_medium()`; the core
  dispatch logic doesn't change when adding new sources (Open/Closed
  Principle).
- Separation of concerns: detection/selection lives here; crawling behaviour
  lives in dedicated crawler classes.
- Safety: when no pattern matches, we fall back to `CustomArticleCrawler` and
  log a warning so the pipeline can continue gracefully.

Notes:
- Domain patterns are stored as regex strings. We normalize domains using
  `urlparse` and `re.escape` before building a pattern; this avoids brittle
  string matching and allows matching URLs with or without `www` prefixes.
"""

import re
from urllib.parse import urlparse

from loguru import logger

from .base import BaseCrawler
from .custom_article import CustomArticleCrawler
from .github import GithubCrawler
from .linkedin import LinkedInCrawler
from .medium import MediumCrawler


class CrawlerDispatcher:
    """Selects and instantiates an appropriate crawler for a URL.

    The dispatcher acts as a tiny registry + factory. Use `build()` and the
    chained registration helpers to configure which crawlers should be used
    for which domains. Example:

        dispatcher = (
            CrawlerDispatcher.build()
            .register_medium()
            .register_github()
        )

    After configuration `get_crawler(url)` can be called to obtain an instance
    of the matching crawler. If there is no match, a `CustomArticleCrawler`
    instance is returned as a safe default.
    """

    def __init__(self) -> None:
        # Internal mapping of regex pattern -> crawler class. Patterns are
        # constructed in `register()` and are intended to match full URLs.
        self._crawlers = {}

    @classmethod
    def build(cls) -> "CrawlerDispatcher":
        # Builder-style entry point. Keeps construction expressive and allows
        # chaining registration calls immediately after build().
        dispatcher = cls()

        return dispatcher

    def register_medium(self) -> "CrawlerDispatcher":
        # Convenience registration for Medium domain. Calling this is the
        # same as calling `register('https://medium.com', MediumCrawler)`.
        self.register("https://medium.com", MediumCrawler)

        return self

    def register_linkedin(self) -> "CrawlerDispatcher":
        # Convenience registration for LinkedIn domain.
        self.register("https://linkedin.com", LinkedInCrawler)

        return self

    def register_github(self) -> "CrawlerDispatcher":
        # Convenience registration for GitHub domain.
        self.register("https://github.com", GithubCrawler)

        return self

    def register(self, domain: str, crawler: type[BaseCrawler]) -> None:
        """Register a crawler class for a given domain.

        Parameters
        - domain: a URL-like string (e.g. "https://medium.com" or
          "http://example.com"). We parse it with `urlparse` and extract the
          `netloc` to normalize the input and avoid relying on exact prefix
          matching.
        - crawler: a subclass of `BaseCrawler` which will be instantiated when
          a matching URL is requested.

        Implementation details:
        - We extract the domain's `netloc` and escape it with `re.escape`
          so that special characters in the domain don't break the regex.
        - The stored pattern includes an optional `www.` prefix and expects
          URLs that start with the scheme (https://). This keeps matching
          simple and explicit.
        """
        parsed_domain = urlparse(domain)
        domain = parsed_domain.netloc

        # Build a regex that matches URLs for the given domain. We escape the
        # domain to make the pattern robust against special characters and add
        # an optional 'www.' group so both 'example.com' and 'www.example.com'
        # match. The pattern is anchored at the scheme to avoid accidental
        # substring matches.
        self._crawlers[r"https://(www\.)?{}/*".format(re.escape(domain))] = crawler

    def get_crawler(self, url: str) -> BaseCrawler:
        """Return an instance of the registered crawler that matches `url`.

        We iterate over registered regex patterns and use `re.match` to find
        the first matching crawler. The dispatcher returns a fresh instance of
        the crawler class. If no pattern matches, a `CustomArticleCrawler`
        instance is returned and a warning is logged.
        """
        for pattern, crawler in self._crawlers.items():
            # Use re.match which attempts to match at the beginning of the
            # string. Patterns are intentionally constructed to match the
            # scheme and domain portion of the URL.
            if re.match(pattern, url):
                return crawler()
        else:
            # No registered crawler matched; provide a safe fallback so the
            # caller can still attempt to extract content.
            logger.warning(f"No crawler found for {url}. Defaulting to CustomArticleCrawler.")

            return CustomArticleCrawler()
