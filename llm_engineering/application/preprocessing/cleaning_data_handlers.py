# What: Import ABC and abstractmethod to define the abstract base handler.
# Why: ABC prevents CleaningDataHandler from being instantiated directly;
#      abstractmethod forces every concrete subclass to implement clean(),
#      ensuring a consistent interface regardless of document type.
from abc import ABC, abstractmethod

# What: Import Generic and TypeVar for type-safe generics.
# Why: Generic[DocumentT, CleanedDocumentT] makes each concrete handler's
#      input/output types explicit (e.g. PostDocument → CleanedPostDocument),
#      giving static type checkers full visibility without losing the shared base class.
from typing import Generic, TypeVar

# What: Import the cleaned document domain models (output types for each handler).
# Why: Each handler returns a specific CleanedDocument subclass that gets stored
#      in Qdrant; importing them here makes the type mapping clear and explicit.
from llm_engineering.domain.cleaned_documents import (
    CleanedArticleDocument,
    CleanedDocument,
    CleanedPostDocument,
    CleanedRepositoryDocument,
)

# What: Import the raw document domain models (input types for each handler).
# Why: Each handler's clean() method accepts a specific raw document type from
#      MongoDB; importing them provides precise type annotations for IDEs and
#      static analysis tools.
from llm_engineering.domain.documents import (
    ArticleDocument,
    Document,
    PostDocument,
    RepositoryDocument,
)

# What: Import the clean_text utility function that does the actual text normalisation.
# Why: Separating the text-cleaning algorithm from the handler class keeps each file
#      single-responsibility. clean_text handles HTML stripping, whitespace normalisation,
#      special character removal, etc., and can be tested independently.
from .operations import clean_text

# What: TypeVar for the input raw document type, bound to Document.
# Why: Constrains the input to be a Document subclass, giving the type checker enough
#      information to catch passing the wrong document type to a handler.
DocumentT = TypeVar("DocumentT", bound=Document)

# What: TypeVar for the output cleaned document type, bound to CleanedDocument.
# Why: Pairs with DocumentT so each handler's clean() return type is precisely
#      typed — e.g. PostCleaningHandler always returns a CleanedPostDocument, not
#      just a generic CleanedDocument.
CleanedDocumentT = TypeVar("CleanedDocumentT", bound=CleanedDocument)


# What: Abstract base class that all concrete cleaning handlers must inherit from.
# Why: Defines the shared contract (a single clean() method) so the CleaningDispatcher
#      can call any handler through a unified interface. This is the Strategy pattern:
#      the cleaning algorithm is swapped at runtime depending on the document type,
#      while the dispatcher code stays unchanged.
class CleaningDataHandler(ABC, Generic[DocumentT, CleanedDocumentT]):
    """
    Abstract class for all cleaning data handlers.
    All data transformations logic for the cleaning step is done here
    """

    # What: Abstract method that every concrete handler must implement.
    # Why: Forces subclasses to define how their specific raw document type is
    #      transformed into a cleaned document. Declaring it abstract here means
    #      calling clean() on the base class raises a TypeError immediately,
    #      preventing accidental use of an unconfigured handler.
    @abstractmethod
    def clean(self, data_model: DocumentT) -> CleanedDocumentT:
        pass


# What: Concrete handler that cleans social-media post documents.
# Why: Posts store their content as a dict of sections (e.g. {"text": "...", "caption": "..."}).
#      This handler joins all sections and runs text normalisation to produce a single
#      clean string ready for chunking.
class PostCleaningHandler(CleaningDataHandler):

    def clean(self, data_model: PostDocument) -> CleanedPostDocument:
        """What: Join all content sections of the post with a #### separator, then clean the text.
        Why: Post content arrives as a dict because different platforms structure their
        payloads differently (title, body, caption, etc.). Joining with " #### " preserves
        the section boundaries as a visible delimiter so the chunker can see where one
        section ends and another begins, rather than merging them into an undifferentiated blob.
        clean_text then strips HTML, normalises whitespace, and removes noise.
        """
        return CleanedPostDocument(
            # What: Preserve the original document ID.
            # Why: Using the same UUID links the cleaned document back to its raw parent
            #      in MongoDB, maintaining full data lineage across pipeline stages.
            id=data_model.id,

            # What: Join all content dict values with a section separator, then clean.
            # Why: .values() extracts just the text payloads regardless of key names,
            #      making the handler robust to different platform content schemas.
            #      " #### " acts as a lightweight section marker that survives cleaning.
            content=clean_text(" #### ".join(data_model.content.values())),

            platform=data_model.platform,           # Source platform preserved for attribution
            author_id=data_model.author_id,         # Lineage: links to the UserDocument author
            author_full_name=data_model.author_full_name,  # Denormalised for downstream display
            # What: Carry the optional image URL forward, or set None if absent.
            # Why: Some posts have images; preserving the URL keeps it available for
            #      chunking and RAG display. The explicit None assignment is defensive
            #      against falsy values like empty strings.
            image=data_model.image if data_model.image else None,
        )


# What: Concrete handler that cleans long-form article documents (e.g. Medium articles).
# Why: Articles may contain sections where some content values are empty strings or None
#      (e.g. a missing subtitle). This handler filters those out before joining so the
#      cleaned content doesn't contain empty #### separators or blank lines.
class ArticleCleaningHandler(CleaningDataHandler):

    def clean(self, data_model: ArticleDocument) -> CleanedArticleDocument:
        """What: Filter out empty content sections, join the rest, then clean the text.
        Why: Articles scraped from the web often have optional fields (subtitle, author bio,
        pull quotes) that may be empty for a given article. Including empty values would
        produce " ####  #### " noise in the cleaned content, which could confuse the
        chunker and embed into meaningless vectors. The list comprehension drops falsy
        values before joining, keeping the output clean and meaningful.
        """
        # What: Keep only non-empty content values from the article's content dict.
        # Why: Falsy values (empty strings, None) would contribute nothing to the
        #      cleaned text but would add #### separators, degrading content quality.
        valid_content = [content for content in data_model.content.values() if content]

        return CleanedArticleDocument(
            id=data_model.id,

            # What: Join only the non-empty sections, then normalise the text.
            # Why: Using valid_content instead of all values ensures the cleaned
            #      content is dense with meaningful text and free of empty-section noise.
            content=clean_text(" #### ".join(valid_content)),

            platform=data_model.platform,
            # What: Article-specific canonical URL preserved from the raw document.
            # Why: Carried all the way to the embedded chunk so RAG responses can
            #      cite the exact source article URL.
            link=data_model.link,
            author_id=data_model.author_id,
            author_full_name=data_model.author_full_name,
        )


# What: Concrete handler that cleans code repository documents (e.g. GitHub READMEs).
# Why: Repository content (README sections, code file contents) is stored as a dict
#      of file/section names to their text. This handler joins all sections and cleans
#      the result, treating the full repository content as one document.
class RepositoryCleaningHandler(CleaningDataHandler):

    def clean(self, data_model: RepositoryDocument) -> CleanedRepositoryDocument:
        """What: Join all repository content sections with a #### separator, then clean.
        Why: A repository's content dict may contain multiple files or README sections.
        Joining them preserves all content in one document for chunking, which is
        appropriate because repository content should be searched holistically —
        a code snippet in one file often only makes sense with context from another.
        No empty-value filtering here (unlike articles) because repository file
        contents are expected to always be non-empty by the crawler.
        """
        return CleanedRepositoryDocument(
            id=data_model.id,
            content=clean_text(" #### ".join(data_model.content.values())),
            platform=data_model.platform,
            # What: Repository name carried forward from the raw document.
            # Why: Preserved all the way to the embedded chunk for attribution
            #      and citation in RAG responses.
            name=data_model.name,
            # What: Repository URL carried forward.
            # Why: Provides a direct link back to the source repo in RAG responses,
            #      allowing users to inspect the full context of retrieved code.
            link=data_model.link,
            author_id=data_model.author_id,
            author_full_name=data_model.author_full_name,
        )
