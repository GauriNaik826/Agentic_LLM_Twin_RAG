from loguru import logger
from typing_extensions import Annotated
from zenml import get_step_context, step

from llm_engineering.application import utils
from llm_engineering.domain.documents import UserDocument


# Marks this function as a ZenML step.
@step
def get_or_create_user(user_full_name: str) -> Annotated[UserDocument, "user"]:
    # Log which user name we are processing.
    logger.info(f"Getting or creating user: {user_full_name}")

    # Split a full name like "Maxime Labonne" into first and last name.
    first_name, last_name = utils.split_user_full_name(user_full_name)

    # Fetch the user if present; otherwise create and persist a new user record.
    user = UserDocument.get_or_create(first_name=first_name, last_name=last_name)

    # Get the runtime step context to attach metadata visible in ZenML UI.
    step_context = get_step_context()
    # Store input/output metadata for observability and debugging.
    step_context.add_output_metadata(output_name="user", metadata=_get_metadata(user_full_name, user))

    # Return a typed UserDocument artifact named "user".
    return user


def _get_metadata(user_full_name: str, user: UserDocument) -> dict:
    # Metadata payload recorded for this step execution.
    return {
        # Original query parameters.
        "query": {
            "user_full_name": user_full_name,
        },
        # Values actually retrieved/created from persistence.
        "retrieved": {
            "user_id": str(user.id),
            "first_name": user.first_name,
            "last_name": user.last_name,
        },
    }
