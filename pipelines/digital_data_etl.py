from zenml import pipeline

from steps.etl import crawl_links, get_or_create_user


# Marks this function as a ZenML pipeline (a workflow made of steps).
@pipeline
def digital_data_etl(user_full_name: str, links: list[str]) -> str:
    # Step 1: Find an existing user in the DB, or create a new one from full name.
    user = get_or_create_user(user_full_name)
    # Step 2: Crawl all provided links and store scraped data with this user as author.
    last_step = crawl_links(user=user, links=links)

    # Return the ZenML invocation ID of the last step so runs can be tracked externally.
    return last_step.invocation_id
