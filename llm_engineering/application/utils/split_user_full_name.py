"""
split_user_full_name.py — Utility for parsing a full name string into parts
----------------------------------------------------------------------------

WHY THIS UTILITY EXISTS
-----------------------
`UserDocument` (domain/documents.py) stores `first_name` and `last_name` as
separate fields. However, user identity is often supplied as a single full-name
string (e.g., from a YAML config file: "Paul Iusztin"). This utility bridges
that gap: it parses a free-form full-name string into the (first_name,
last_name) pair that `UserDocument` requires.

Centralising this logic in one function means:
- All callers (ETL steps, config parsers) parse names consistently.
- Edge cases (None, single-word names, multi-part given names) are handled once.
- Unit tests for name parsing live in one place.
"""

# ImproperlyConfigured is a domain-level sentinel exception used when the
# system receives bad configuration data (as opposed to a runtime error).
# Using it here signals that an empty/None user name is a misconfiguration
# of the pipeline (bad YAML config) rather than an unexpected runtime fault.
from llm_engineering.domain.exceptions import ImproperlyConfigured


def split_user_full_name(user: str | None) -> tuple[str, str]:
    """Parse a full-name string into (first_name, last_name) components.

    Parameters
    ----------
    user : str | None
        A full-name string, e.g. "Paul Iusztin" or "Maxime Labonne".
        May also be a single-token name like "Cher".

    Returns
    -------
    tuple[str, str]
        (first_name, last_name) where:
        - For multi-token names, everything except the last token is treated
          as the first name (handles compound given names like "Mary Ann"),
          and the last token is the last name.
        - For single-token names, both first_name and last_name are set to
          the same token, ensuring `UserDocument` can always be constructed.

    Raises
    ------
    ImproperlyConfigured
        If `user` is None or resolves to an empty string after splitting,
        indicating that no usable name was provided in the configuration.
    """

    # Guard against None being passed in. A None user name means the pipeline
    # step that reads the config file failed to supply a name — this is a
    # configuration error, not a data error, so we raise ImproperlyConfigured.
    if user is None:
        raise ImproperlyConfigured("User name is empty")

    # Split the full name on whitespace. A name like "Paul Iusztin" becomes
    # ["Paul", "Iusztin"]; "Mary Ann Smith" becomes ["Mary", "Ann", "Smith"].
    # The result is a list of tokens we can reason about by length.
    name_tokens = user.split(" ")

    if len(name_tokens) == 0:
        # split(" ") on an empty string actually returns [""] (a list with one
        # empty token), so this branch is a defensive fallback in case the
        # input is somehow a truly empty sequence after splitting.
        raise ImproperlyConfigured("User name is empty")
    elif len(name_tokens) == 1:
        # Single-token name (e.g., "Cher" or a username). There is no
        # meaningful first/last distinction, so we duplicate the token for
        # both fields. This prevents a downstream KeyError when constructing
        # `UserDocument(first_name=..., last_name=...)`.
        first_name, last_name = name_tokens[0], name_tokens[0]
    else:
        # Multi-token name: treat the last token as the family name and
        # everything before it as the given name. Joining with " " correctly
        # reconstructs compound given names (e.g., "Mary Ann" from
        # ["Mary", "Ann", "Smith"] → first_name="Mary Ann", last_name="Smith").
        first_name, last_name = " ".join(name_tokens[:-1]), name_tokens[-1]

    # Return the parsed pair. Callers unpack this directly into
    # `UserDocument(first_name=first_name, last_name=last_name)`.
    return first_name, last_name
