import time
from abc import ABC, abstractmethod
from tempfile import mkdtemp

import chromedriver_autoinstaller
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

from llm_engineering.domain.documents import NoSQLBaseDocument

# Ensure a compatible chromedriver binary is available for the runtime.
#
# `chromedriver_autoinstaller.install()` checks the local environment for a
# chromedriver matching the installed Chrome/Chromium version; if missing it
# downloads the appropriate binary and places it on PATH (or a location where
# webdriver.Chrome can find it). Doing this eagerly at module import time
# simplifies runtime setup for environments where the binary isn't preinstalled
# (e.g., ephemeral CI containers or dev machines). The downside is this will
# perform network I/O on import; that tradeoff is acceptable here for ease of
# use, but could be deferred to a lazy init if import-time side-effects are
# undesirable in some deployments.
chromedriver_autoinstaller.install()


class BaseCrawler(ABC):
    """Abstract base class that defines the crawling contract.

    - `model` is an attribute that subclasses should set to the specific
      `NoSQLBaseDocument` subclass that the crawler will produce. Typing this
      attribute makes the relationship explicit and helps downstream code
      know which document type to expect for persistence.
    - `extract(...)` is the required method all concrete crawlers must
      implement. It encapsulates the extraction logic for a single URL/link.

    This interface enforces a consistent API across different crawler
    implementations, enabling polymorphism: callers can invoke `extract` on
    any `BaseCrawler` without needing to know implementation details.
    """

    # Declared as a type attribute so concrete crawlers document what model
    # they produce. Not assigned here because it varies by implementation.
    model: type[NoSQLBaseDocument]

    @abstractmethod
    def extract(self, link: str, **kwargs) -> None: ...


class BaseSeleniumCrawler(BaseCrawler, ABC):
    """Reusable Selenium-based crawler functionality.

    Many websites require a browser environment (JS execution, dynamic
    loading) to render content. `BaseSeleniumCrawler` centralizes common
    Selenium setup and helpers so concrete crawlers (e.g., LinkedIn, Medium)
    don't duplicate configuration and can focus on extraction details.
    """

    def __init__(self, scroll_limit: int = 5) -> None:
        # Create Chrome options. We use webdriver.ChromeOptions rather than
        # a bare Options instance for maximum compatibility across selenium
        # versions and to access Chrome-specific flags.
        options = webdriver.ChromeOptions()

        # Add recommended flags for headless, hardened browser usage in
        # containerized or CI environments.
        options.add_argument("--no-sandbox")  # Required in some container runtimes
        options.add_argument("--headless=new")  # Run headless (no visible UI)
        options.add_argument("--disable-dev-shm-usage")  # Avoid /dev/shm size issues
        options.add_argument("--log-level=3")  # Reduce noise from Chrome logs
        options.add_argument("--disable-popup-blocking")  # Allow popups if needed
        options.add_argument("--disable-notifications")  # Disable site notifications
        options.add_argument("--disable-extensions")  # Disable installed extensions
        options.add_argument("--disable-background-networking")  # Limit background traffic
        options.add_argument("--ignore-certificate-errors")  # Allow self-signed certs

        # Use separate temporary directories for user data, data path and
        # disk cache. This avoids sharing state across concurrent crawler
        # instances and ensures ephemeral, clean profiles for each run.
        options.add_argument(f"--user-data-dir={mkdtemp()}")
        options.add_argument(f"--data-path={mkdtemp()}")
        options.add_argument(f"--disk-cache-dir={mkdtemp()}")

        # Open a remote debugging port to make troubleshooting possible and
        # to allow connecting devtools if necessary. Using an explicit port
        # here is optional but can be useful in advanced debugging scenarios.
        options.add_argument("--remote-debugging-port=9226")

        # Hook for subclasses to add or mutate options without overriding
        # `__init__`. This supports varying needs (e.g., enabling a proxy,
        # setting a specific user agent) while keeping common flags here.
        self.set_extra_driver_options(options)

        # How many times `scroll_page` should attempt to scroll before
        # considering the page fully loaded. Concrete crawlers may override
        # this when they expect longer or shorter dynamic content streams.
        self.scroll_limit = scroll_limit

        # Instantiate the Chrome WebDriver with the configured options. This
        # will start a browser process; callers are responsible for closing
        # `self.driver.quit()` when finished (not implemented here to keep
        # lifecycle management in the concrete crawler or orchestration code).
        self.driver = webdriver.Chrome(
            options=options,
        )

    def set_extra_driver_options(self, options: Options) -> None:
        """Extension hook for subclasses to customize Chrome options.

        Subclasses can override this method to add flags (e.g., proxy or
        user-agent configuration) without reimplementing the entire
        initialization sequence. The default implementation is a no-op.
        """
        pass

    def login(self) -> None:
        """Optional login helper for sites that require authentication.

        Concrete crawlers that must authenticate (e.g., LinkedIn) should
        implement this method. It is separated from `extract` so callers
        can choose when and how to handle auth (e.g., reuse cookies across
        multiple extractions). The default implementation is intentionally
        empty because not all crawlers require login.
        """
        pass

    def scroll_page(self) -> None:
        """Scroll the current page to trigger lazy/dynamic loading.

        Many modern pages load content incrementally as the viewport is
        scrolled. This helper repeatedly scrolls to the bottom of the page
        and waits briefly, stopping when the page height stops increasing or
        when the configured `scroll_limit` is reached. The method is
        conservative (fixed sleep) to balance reliability and speed; more
        advanced implementations could wait on specific DOM changes instead.
        """
        current_scroll = 0

        # Get the current full page height via JS execution inside the
        # controlled browser. This value changes as content is appended.
        last_height = self.driver.execute_script("return document.body.scrollHeight")

        while True:
            # Scroll to the bottom of the page to trigger lazy loading.
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

            # Sleep briefly to give the page time to fetch and render new
            # content. This is a simple approach; tests may reveal a better
            # tuned delay for specific sites.
            time.sleep(5)

            # Measure the new total height after scrolling and potential
            # content load.
            new_height = self.driver.execute_script("return document.body.scrollHeight")

            # Stop when no new content is added (height unchanged) or we hit
            # the configured scroll limit. The short-circuit `self.scroll_limit`
            # check allows a limit of 0 to mean "no-limit" if callers prefer
            # that convention; here a non-zero limit controls scroll attempts.
            if new_height == last_height or (self.scroll_limit and current_scroll >= self.scroll_limit):
                break

            # Prepare for the next loop iteration.
            last_height = new_height
            current_scroll += 1
