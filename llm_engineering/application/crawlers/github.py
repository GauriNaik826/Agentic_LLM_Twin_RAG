import os
import shutil
import subprocess
import tempfile

from loguru import logger

from llm_engineering.domain.documents import RepositoryDocument

from .base import BaseCrawler


class GithubCrawler(BaseCrawler):
    """Crawler that ingests a GitHub repository into a RepositoryDocument.

    Design notes:
    - Uses `git clone` to obtain repository contents rather than web scraping
    - Stores a mapping of file paths to file contents in `RepositoryDocument`
    - Skips typical binary or meta files via an ignore list
    - Ensures temporary resources are cleaned up on completion or error
    """

    # Declare the target persistence model for this crawler. Downstream code
    # can inspect `model` to know which document type will be produced.
    model = RepositoryDocument

    def __init__(self, ignore=(".git", ".toml", ".lock", ".png")) -> None:
        # Call BaseCrawler initializer (no-op but keeps MRO explicit).
        super().__init__()

        # Store a tuple of filename or directory suffixes to ignore while
        # building the repository tree. This prevents storing irrelevant
        # or large binary files and keeps the persisted document focused on
        # textual/code content useful for downstream tasks.
        self._ignore = ignore

    def extract(self, link: str, **kwargs) -> None:
        """Extract repository contents from `link` and persist them.

        Steps (high level):
        1. Check DB for existing repository to avoid duplicates.
        2. Clone the repo into a temporary directory.
        3. Walk the cloned tree, reading relevant files into a dict.
        4. Create and save a `RepositoryDocument` with collected content.
        5. Clean up temporary files.
        """

        # 1) Prevent duplicate ingestion: query existing documents by link.
        # `self.model.find(...)` is expected to be a convenience provided by
        # the document model (e.g., MongoODM) — if a matching document exists
        # we skip processing to save time and storage.
        old_model = self.model.find(link=link)
        if old_model is not None:
            logger.info(f"Repository already exists in the database: {link}")

            # Early return avoids re-downloading and re-processing the same repo.
            return

        logger.info(f"Starting scrapping GitHub repository: {link}")

        # Normalize repo name from URL. `rstrip` removes trailing slashes so
        # splitting on "/" yields the repo name as the last segment.
        repo_name = link.rstrip("/").split("/")[-1]

        # Create an isolated temporary directory for cloning. Using a temp
        # directory avoids polluting the caller's working directory and
        # ensures we can safely delete the workspace afterwards.
        local_temp = tempfile.mkdtemp()

        try:
            # Change into the temporary directory so `git clone` creates the
            # repository folder here. This keeps path handling simple.
            os.chdir(local_temp)

            # 2) Clone the repository using the system git. Cloning gives us a
            # faithful local copy of all files and directories without having
            # to rely on fragile HTML scraping. We use `subprocess.run` and
            # intentionally do not capture output here — in production you
            # might add timeout, stdout/stderr capture, and error handling.
            subprocess.run(["git", "clone", link])

            # After cloning, `local_temp` should contain a single directory
            # corresponding to the repo. We take the first entry from the
            # listing as the repository path. The noqa comment acknowledges
            # that path handling is intentionally simple here.
            repo_path = os.path.join(local_temp, os.listdir(local_temp)[0])  # noqa: PTH118

            # 3) Walk the repository tree and build a mapping of relative
            # file paths to file contents. `tree` preserves the repo
            # hierarchy which is useful for downstream processing.
            tree = {}
            for root, _, files in os.walk(repo_path):
                # Compute the relative directory path inside the repo.
                dir = root.replace(repo_path, "").lstrip("/")

                # Skip directories that start with any ignore pattern. This
                # keeps us from recursing into `.git` metadata or other
                # directories we don't want to include.
                if dir.startswith(self._ignore):
                    continue

                for file in files:
                    # Skip files with ignored suffixes (e.g., images,
                    # lockfiles). Using `endswith` against a tuple allows
                    # efficient multi-suffix checks.
                    if file.endswith(self._ignore):
                        continue

                    # Build a repository-relative file path key.
                    file_path = os.path.join(dir, file)  # noqa: PTH118

                    # Read file content in text mode; `errors='ignore'`
                    # ensures non-text bytes don't blow up the read. In
                    # production you may prefer to explicitly filter binary
                    # files instead of silencing errors.
                    with open(os.path.join(root, file), "r", errors="ignore") as f:  # noqa: PTH123, PTH118
                        # Normalize content by removing spaces. This was the
                        # existing behavior — note that aggressive
                        # normalization may remove semantic information; you
                        # may want to adjust normalization per use-case.
                        tree[file_path] = f.read().replace(" ", "")

            # 4) Construct and persist the document. `user` is expected to be
            # passed in kwargs by the caller (authentication/orchestration
            # layer). We capture author metadata and platform for traceability.
            user = kwargs["user"]
            instance = self.model(
                content=tree,
                name=repo_name,
                link=link,
                platform="github",
                author_id=user.id,
                author_full_name=user.full_name,
            )
            instance.save()

        except Exception:
            # Re-raise any exception after cleanup by the finally block.
            # In a more robust implementation we might log the error and
            # capture diagnostics (clone stdout/stderr) before re-raising.
            raise
        finally:
            # 5) Always attempt to remove the temporary directory to avoid
            # leaking filesystem state. `shutil.rmtree` will remove the
            # clone and any files created during processing.
            shutil.rmtree(local_temp)

        logger.info(f"Finished scrapping GitHub repository: {link}")
