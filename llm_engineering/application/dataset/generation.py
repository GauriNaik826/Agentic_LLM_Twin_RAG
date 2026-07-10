from abc import ABC, abstractmethod

import tiktoken
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models.fake import FakeListLLM
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from loguru import logger

from llm_engineering import domain
from llm_engineering.application import utils
from llm_engineering.domain.cleaned_documents import CleanedDocument
from llm_engineering.domain.dataset import DatasetType, TrainTestSplit
from llm_engineering.domain.prompt import GenerateDatasetSamplesPrompt, Prompt
from llm_engineering.domain.types import DataCategory
from llm_engineering.settings import settings

from . import constants
from . import utils as generation_utils
from .output_parsers import ListPydanticOutputParser


# Abstract base class that defines the shared pipeline for both dataset generator types
# (InstructionDatasetGenerator and PreferenceDatasetGenerator).
# Using ABC forces subclasses to implement post_process_datasets().
class DatasetGenerator(ABC):
    # Shared tokenizer used to count and truncate prompt tokens before sending to the LLM.
    # Tied to the same model as the one used for generation so token counts are accurate.
    tokenizer = tiktoken.encoding_for_model(settings.OPENAI_MODEL_ID)

    # Subclasses set this to DatasetType.INSTRUCTION or DatasetType.PREFERENCE
    # so shared methods know which kind of data they are producing.
    dataset_type: DatasetType | None = None

    # The system-level instruction sent to the LLM at the start of every request.
    # {dataset_format} is filled in at runtime depending on dataset_type.
    system_prompt_template = """You are a helpful assistant who generates {dataset_format} based on the given context. \
Provide your response in JSON format.
"""
    # Subclasses override this with the actual user-facing prompt that includes the extract.
    prompt_template_str: str | None = None

    @classmethod
    def get_system_prompt(cls) -> Prompt:
        assert cls.dataset_type is not None, "Dataset type must be set before calling get_system_prompt()"

        # Pick the right human-readable label so the LLM knows what format to produce.
        # INSTRUCTION -> pairs (instruction + answer)
        # PREFERENCE  -> triples (instruction + chosen + rejected)
        dataset_format = (
            "instruction-answer pairs" if cls.dataset_type == DatasetType.INSTRUCTION else "instruction-answer triples"
        )
        input_variables = {
            "dataset_format": dataset_format,
        }
        system_prompt = cls.system_prompt_template.format(**input_variables)

        # Wrap in a Prompt domain object so the template, variables, and rendered
        # text are always stored together for logging/tracing purposes.
        return Prompt(
            template=cls.system_prompt_template,
            input_variables=input_variables,
            content=system_prompt,
        )

    @classmethod
    def get_prompts(cls, documents: list[CleanedDocument]) -> dict[DataCategory, list[GenerateDatasetSamplesPrompt]]:
        # Break long documents into overlapping substrings so each prompt fits
        # within the LLM's context window and every part of the document is covered.
        documents = generation_utils.extract_substrings(documents)

        # Group prompts by data category (article, post, repository, etc.) so
        # the caller can report statistics and store results per category.
        grouped_prompts = {}
        grouped_cleaned_documents = CleanedDocument.group_by_category(documents)
        for category, category_documents in grouped_cleaned_documents.items():
            category_prompts = [cls.get_prompt(document) for document in category_documents]
            grouped_prompts[category] = category_prompts

        return grouped_prompts

    @classmethod
    def get_prompt(cls, document: CleanedDocument) -> GenerateDatasetSamplesPrompt:
        assert cls.prompt_template_str is not None, "Prompt template must be set before calling get_prompt()"

        data_category = document.get_category()

        # Use Jinja2 templating so the {{extract}} placeholder in the subclass
        # prompt_template_str is replaced with the actual document content.
        prompt_template = PromptTemplate.from_template(
            template=cls.prompt_template_str,
            template_format="jinja2",
        )
        input_variables = {
            "extract": document.content,
        }
        prompt = prompt_template.format(**input_variables)

        # Tokenise the rendered prompt to enforce the model's max context window.
        # If it exceeds the limit, hard-truncate at the token level (not character
        # level) to avoid cutting in the middle of a multi-byte character.
        prompt_tokens = cls.tokenizer.encode(prompt)
        if len(prompt_tokens) > settings.OPENAI_MAX_TOKEN_WINDOW:
            prompt_tokens = prompt_tokens[: settings.OPENAI_MAX_TOKEN_WINDOW]
            prompt = cls.tokenizer.decode(prompt_tokens)

        # Wrap everything in a domain object so downstream steps can access the
        # original template, the variables, the rendered text, and metadata
        # (token count, category, source document) in one place.
        prompt = GenerateDatasetSamplesPrompt(
            template=prompt_template.template,
            input_variables=input_variables,
            content=prompt,
            num_tokens=len(prompt_tokens),
            data_category=data_category,
            document=document,
        )

        return prompt

    @classmethod
    def generate(
        cls,
        prompts: dict[DataCategory, list[GenerateDatasetSamplesPrompt]],
        test_size: float = 0.2,
        mock: bool = False,
    ) -> TrainTestSplit:
        # Main entry point: send all prompts to the LLM, parse the JSON responses
        # into typed dataset samples, then split the results into train / test sets.
        assert cls.dataset_type is not None, "Dataset type must be set before calling generate()"

        # Helper: converts our domain prompt into the [system, human] message list
        # format that LangChain's ChatOpenAI expects.
        def _to_langchain(
            prompt: GenerateDatasetSamplesPrompt,
        ) -> list[BaseMessage]:
            messages = [
                SystemMessage(content=cls.get_system_prompt().content),
                HumanMessage(content=prompt.content),
            ]

            return messages

        # When mock=True, use a fake LLM that returns hard-coded responses.
        # This is useful for unit tests and local development without burning API credits.
        if mock:
            llm = FakeListLLM(responses=[constants.get_mocked_response(cls.dataset_type)])
        else:
            assert settings.OPENAI_API_KEY is not None, "OpenAI API key must be set to generate datasets"

            # Preference triples are longer (instruction + chosen + rejected),
            # so they need a higher max_tokens budget than instruction pairs.
            llm = ChatOpenAI(
                model=settings.OPENAI_MODEL_ID,
                api_key=settings.OPENAI_API_KEY,
                max_tokens=2000 if cls.dataset_type == DatasetType.PREFERENCE else 1200,
                temperature=0.7,
            )

        # The parser validates and converts the LLM's raw JSON output into a list
        # of the appropriate Pydantic model (InstructDatasetSample or PreferenceDatasetSample).
        parser = ListPydanticOutputParser(pydantic_object=cls._get_dataset_sample_type())

        # LangChain pipe: LLM call -> JSON parse -> typed Pydantic objects.
        chain = llm | parser

        datasets = {}
        for category, category_prompts in prompts.items():
            langchain_category_prompts = [_to_langchain(prompt) for prompt in category_prompts]

            # Process prompts in batches of 24 to avoid overwhelming the API
            # with too many concurrent requests while still being efficient.
            batches = utils.misc.batch(langchain_category_prompts, size=24)

            flattened_instruct_dataset_samples = []
            for batch in batches:
                try:
                    # chain.batch() sends all prompts in the batch concurrently
                    # and returns a list of lists (one inner list per prompt).
                    batched_dataset_samples = chain.batch(batch, stop=None)

                    # Flatten the list-of-lists into a single flat list of samples.
                    for instruct_dataset_sample_batch in batched_dataset_samples:
                        flattened_instruct_dataset_samples.extend(instruct_dataset_sample_batch)
                except OutputParserException:
                    # If the LLM returns malformed JSON for a whole batch, log and skip
                    # rather than crashing — partial data is better than no data.
                    logger.exception(f"Failed to parse the output JSON for a batch for category {category}")

            dataset = domain.dataset.build_dataset(
                dataset_type=cls.dataset_type, category=category, samples=flattened_instruct_dataset_samples
            )
            datasets[category] = dataset
            logger.info(f"Generated {len(dataset.samples)} samples for category '{category}'.")

        # Delegate train/test splitting (and any filtering) to the subclass
        # because instruction and preference datasets have different quality checks.
        processed_datasets = cls.post_process_datasets(datasets, test_size=test_size)

        return processed_datasets

    @classmethod
    def _get_dataset_sample_type(
        cls,
    ) -> type[domain.dataset.InstructDatasetSample] | type[domain.dataset.PreferenceDatasetSample]:
        # Returns the correct Pydantic model so the output parser knows which
        # fields to validate (instruction+answer vs instruction+chosen+rejected).
        return (
            domain.dataset.InstructDatasetSample
            if cls.dataset_type == DatasetType.INSTRUCTION
            else domain.dataset.PreferenceDatasetSample
        )

    @classmethod
    @abstractmethod
    def post_process_datasets(
        cls, datasets: dict[DataCategory, domain.dataset.InstructDataset], test_size: float
    ) -> TrainTestSplit:
        # Subclasses implement this to apply dataset-specific filtering and
        # to split samples into train / test sets.
        pass


# Generates supervised fine-tuning (SFT) data: each sample is an (instruction, answer) pair
# where both the question and the answer are written in the author's own voice.
class InstructionDatasetGenerator(DatasetGenerator):
    dataset_type = DatasetType.INSTRUCTION

    # The prompt asks the LLM to produce 5 pairs per extract.
    # Key constraints:
    #   - instructions must be self-contained (no references to "the extract")
    #   - answers must mimic the writing style of the source text
    # This produces SFT data that teaches the fine-tuned model to write like the original author.
    prompt_template_str = """Based on the following extract, generate five instruction-answer pairs. Each instruction \
must ask to write about a specific topic contained in the context. Each answer \
must provide a relevant paragraph based on the information found in the \
context. Only use concepts from the context to generate the instructions. \
Instructions must never explicitly mention a context, a system, a course, or an extract. \
Instructions must be self-contained and general. \
Answers must imitate the writing style of the context. \
    
Example instruction: Explain the concept of an LLM Twin. \
Example answer: An LLM Twin is essentially an AI character that mimics your writing style, personality, and voice. \
It's designed to write just like you by incorporating these elements into a language model. \
The idea is to create a digital replica of your writing habits using advanced AI techniques. \

Structure the answer in JSON format, ready to be loaded in Python by json.loads(), as a list of objects.
Do not add any extra characters and provide your response in JSON format with the following structure:
[
    {"instruction": "...", "answer": "..."},
    ...
]

Extract:
{{extract}}
"""

    @classmethod
    def post_process_datasets(
        cls, datasets: dict[DataCategory, domain.dataset.InstructDataset], test_size: float
    ) -> TrainTestSplit:
        # No extra filtering needed for instruction data — just split into train/test.
        # random_state=42 ensures reproducible splits across runs.
        train_test_split = generation_utils.create_instruct_train_test_split(
            datasets, test_size=test_size, random_state=42
        )

        return train_test_split


# Generates preference / alignment data used for DPO (Direct Preference Optimization).
# Each sample is a triple: (instruction, chosen, rejected).
#   chosen   = verbatim text from the source document (ground truth)
#   rejected = an LLM-generated answer (plausible but not as good)
# Training on this data teaches the model to prefer the author's real writing over generic answers.
class PreferenceDatasetGenerator(DatasetGenerator):
    dataset_type = DatasetType.PREFERENCE

    prompt_template_str = """Based on the following extract, generate five instruction-answer triples. Each triple should consist of:
1. An instruction asking about a specific topic in the context.
2. A generated answer that attempts to answer the instruction based on the context, named as 'rejected'.
3. An extracted answer that is a relevant excerpt directly from the given context, named as 'chosen'.

Instructions must be self-contained and general, without explicitly mentioning a context, system, course, or extract.

Important:
- Ensure that the extracted answer, the chosen one, is a verbatim copy from the context, including all punctuation and apostrophes.
- Do not add any ellipsis (...) or [...]  to indicate skipped text in the extracted answer.
- If the relevant text is not continuous, use two separate sentences from the context instead of skipping text.

Structure the answer in JSON format, ready to be loaded in Python by json.loads(), as a list of objects.
Do not add any extra characters and provide your response in JSON format with the following structure:
[
    {
        "instruction": "...",
        "rejected": "...",
        "chosen": "..."
    },
    ...
]

Extract:
{{extract}}
"""

    @classmethod
    def post_process_datasets(
        cls, datasets: dict[DataCategory, domain.dataset.PreferenceDataset], test_size: float
    ) -> TrainTestSplit:
        # Preference data needs extra quality checks before splitting:
        # 1. Remove samples where chosen/rejected answers are too short to be meaningful.
        # 2. Remove samples where the format is wrong (e.g., the LLM added "..." instead of
        #    copying verbatim text), because DPO training is sensitive to data quality.
        datasets = generation_utils.filter_short_answers(datasets)
        datasets = generation_utils.filter_answer_format(datasets)

        remaining_samples = sum([dataset.num_samples for dataset in datasets.values()])
        logger.info(
            f"Filtered out short answers and answers with incorrect format. Remaining samples: {remaining_samples}"
        )

        train_test_split = generation_utils.create_preference_train_test_split(
            datasets, test_size=test_size, random_state=42
        )

        return train_test_split


# Factory function: returns the right generator class for the requested dataset type.
# Callers use this instead of importing the subclasses directly, keeping coupling low.
def get_dataset_generator(dataset_type: DatasetType) -> type[DatasetGenerator]:
    if dataset_type == DatasetType.INSTRUCTION:
        return InstructionDatasetGenerator
    elif dataset_type == DatasetType.PREFERENCE:
        return PreferenceDatasetGenerator
    else:
        raise ValueError(f"Invalid dataset type: {dataset_type}")
