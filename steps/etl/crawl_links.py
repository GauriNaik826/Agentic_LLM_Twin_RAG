from urllib.parse import urlparse

from loguru import logger
from tqdm import tqdm
from typing_extensions import Annotated
from zenml import get_step_context, step

from llm_engineering.application.crawlers.dispatcher import CrawlerDispatcher
from llm_engineering.domain.documents import UserDocument


# Marks this function as a ZenML step.
@step
def crawl_links(user: UserDocument, links: list[str]) -> Annotated[list[str], "crawled_links"]:
    # Build crawler dispatcher and register supported platforms.
    dispatcher = CrawlerDispatcher.build().register_linkedin().register_medium().register_github()

    # Log how many URLs we are going to process.
    logger.info(f"Starting to crawl {len(links)} link(s).")

    # Track per-domain success/failure stats for step metadata.
    metadata = {}
    # Running count of successful crawls.
    successfull_crawls = 0
    # Iterate through links with a progress bar.
    for link in tqdm(links):
        # Crawl one link and get (success flag, domain).
        successfull_crawl, crawled_domain = _crawl_link(dispatcher, link, user)
        # Add this result to the global success counter.
        successfull_crawls += successfull_crawl

        # Update per-domain metadata summary.
        metadata = _add_to_metadata(metadata, crawled_domain, successfull_crawl)

    # Get the ZenML step context to attach metadata.
    step_context = get_step_context()
    # Persist crawl summary metadata in ZenML outputs.
    step_context.add_output_metadata(output_name="crawled_links", metadata=metadata)

    # Final success log for visibility.
    logger.info(f"Successfully crawled {successfull_crawls} / {len(links)} links.")

    # Return crawled links artifact (same list received as input).
    return links


def _crawl_link(dispatcher: CrawlerDispatcher, link: str, user: UserDocument) -> tuple[bool, str]:
    # Select crawler implementation based on URL domain/pattern.
    crawler = dispatcher.get_crawler(link)
    # Extract domain (e.g., medium.com) for metadata grouping.
    crawler_domain = urlparse(link).netloc

    try:
        # Execute platform-specific extraction and persistence.
        crawler.extract(link=link, user=user)

        # On success, return True + domain.
        return (True, crawler_domain)
    except Exception as e:
        # Log error and continue processing remaining links.
        logger.error(f"An error occurred while crowling: {e!s}")

        # On failure, return False + domain.
        return (False, crawler_domain)


def _add_to_metadata(metadata: dict, domain: str, successfull_crawl: bool) -> dict:
    # Initialize domain bucket the first time we see it.
    if domain not in metadata:
        metadata[domain] = {}
    # Increment successful count for this domain.
    metadata[domain]["successful"] = metadata[domain].get("successful", 0) + successfull_crawl
    # Increment total attempts for this domain.
    metadata[domain]["total"] = metadata[domain].get("total", 0) + 1

    # Return updated metadata dictionary.
    return metadata
