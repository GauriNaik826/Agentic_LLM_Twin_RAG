from enum import Enum

from loguru import logger

# Hugging Face `datasets` is an optional heavy dependency.
# Wrapping the import in try/except lets the rest of the codebase load even
# when `datasets` is not installed (e.g. during inference-only deployments).
try:
    from datasets import Dataset, DatasetDict, concatenate_datasets
except ImportError:
    logger.warning("Huggingface datasets not installed. Install with `pip install datasets`")


from llm_engineering.domain.base import VectorBaseDocument
from llm_engineering.domain.types import DataCategory


# Enum that distinguishes the two dataset formats used for fine-tuning:
#   INSTRUCTION -> supervised fine-tuning (SFT): (instruction, answer) pairs
#   PREFERENCE  -> alignment / DPO training: (instruction, chosen, rejected) triples
class DatasetType(Enum):
    INSTRUCTION = "instruction"
    PREFERENCE = "preference"


# Represents one (instruction, answer) pair produced by the LLM during dataset generation.
# Stored in the vector DB so it can be retrieved, versioned, and pushed to Hugging Face.
class InstructDatasetSample(VectorBaseDocument):
    instruction: str
    answer: str

    class Config:
        category = DataCategory.INSTRUCT_DATASET_SAMPLES


# Represents one DPO triple: the instruction, a generated (rejected) answer, and a
# verbatim excerpt from the source document (chosen).
# `chosen` is always better than `rejected` — that contrast is what DPO trains on.
class PreferenceDatasetSample(VectorBaseDocument):
    instruction: str
    rejected: str  # LLM-generated answer (plausible but not ideal)
    chosen: str    # Verbatim text from the author's own writing (ground truth)

    class Config:
        category = DataCategory.PREFERENCE_DATASET_SAMPLES


# Container that groups all InstructDatasetSamples for a single data category
# (e.g. all article samples together, all post samples together).
# Keeping samples category-scoped lets downstream code report per-category statistics
# and apply category-specific filtering if needed.
class InstructDataset(VectorBaseDocument):
    category: DataCategory
    samples: list[InstructDatasetSample]

    class Config:
        category = DataCategory.INSTRUCT_DATASET

    @property
    def num_samples(self) -> int:
        return len(self.samples)

    def to_huggingface(self) -> "Dataset":
        # Convert Pydantic models to plain dicts first, then reshape into the
        # column-oriented format that Hugging Face Dataset.from_dict() expects.
        # The column is renamed from "answer" -> "output" to match the Alpaca
        # SFT format that most fine-tuning frameworks expect.
        data = [sample.model_dump() for sample in self.samples]

        return Dataset.from_dict(
            {"instruction": [d["instruction"] for d in data], "output": [d["answer"] for d in data]}
        )


# Holds the train/test split for either dataset type.
# Using a base class here avoids duplicating the to_huggingface() conversion logic;
# the concrete subclasses (InstructTrainTestSplit, PreferenceTrainTestSplit) just
# narrow the type annotations so callers get proper type-checking.
class TrainTestSplit(VectorBaseDocument):
    train: dict
    test: dict
    test_split_size: float  # e.g. 0.2 means 20 % of samples go to the test set

    def to_huggingface(self, flatten: bool = False) -> "DatasetDict":
        # Convert every per-category dataset into a Hugging Face Dataset object.
        train_datasets = {category.value: dataset.to_huggingface() for category, dataset in self.train.items()}
        test_datasets = {category.value: dataset.to_huggingface() for category, dataset in self.test.items()}

        if flatten:
            # flatten=True merges all categories into one big dataset.
            # Useful when you want to train on all data regardless of source type.
            train_datasets = concatenate_datasets(list(train_datasets.values()))
            test_datasets = concatenate_datasets(list(test_datasets.values()))
        else:
            # flatten=False keeps categories as separate columns inside
            # a single Dataset, preserving per-category metadata.
            train_datasets = Dataset.from_dict(train_datasets)
            test_datasets = Dataset.from_dict(test_datasets)

        # Wrap in DatasetDict so the result can be pushed directly to the
        # Hugging Face Hub with dataset.push_to_hub(), which expects this format.
        return DatasetDict({"train": train_datasets, "test": test_datasets})


# Typed specialisation of TrainTestSplit for instruction (SFT) data.
# Narrowing the dict types from `dict` to `dict[DataCategory, InstructDataset]`
# gives static type checkers and IDE auto-complete full visibility into the contents.
class InstructTrainTestSplit(TrainTestSplit):
    train: dict[DataCategory, InstructDataset]
    test: dict[DataCategory, InstructDataset]
    test_split_size: float

    class Config:
        category = DataCategory.INSTRUCT_DATASET


# Container for all PreferenceDatasetSamples for one data category.
# Mirrors InstructDataset but holds triples instead of pairs.
class PreferenceDataset(VectorBaseDocument):
    category: DataCategory
    samples: list[PreferenceDatasetSample]

    class Config:
        category = DataCategory.PREFERENCE_DATASET

    @property
    def num_samples(self) -> int:
        return len(self.samples)

    def to_huggingface(self) -> "Dataset":
        # Reshape Pydantic models into the column-oriented dict that Dataset.from_dict() needs.
        # The column is renamed from "instruction" -> "prompt" to match the TRL / DPO
        # trainer convention, which expects columns named "prompt", "chosen", "rejected".
        data = [sample.model_dump() for sample in self.samples]

        return Dataset.from_dict(
            {
                "prompt": [d["instruction"] for d in data],
                "rejected": [d["rejected"] for d in data],
                "chosen": [d["chosen"] for d in data],
            }
        )


# Typed specialisation of TrainTestSplit for preference (DPO) data.
class PreferenceTrainTestSplit(TrainTestSplit):
    train: dict[DataCategory, PreferenceDataset]
    test: dict[DataCategory, PreferenceDataset]
    test_split_size: float

    class Config:
        category = DataCategory.PREFERENCE_DATASET


# Factory function: constructs the correct dataset container based on dataset_type.
# Why a factory instead of calling InstructDataset / PreferenceDataset directly?
# It keeps callers decoupled from the concrete classes — they only need to pass
# the DatasetType enum value, and the factory picks the right implementation.
def build_dataset(dataset_type, *args, **kwargs) -> InstructDataset | PreferenceDataset:
    if dataset_type == DatasetType.INSTRUCTION:
        return InstructDataset(*args, **kwargs)
    elif dataset_type == DatasetType.PREFERENCE:
        return PreferenceDataset(*args, **kwargs)
    else:
        raise ValueError(f"Invalid dataset type: {dataset_type}")
