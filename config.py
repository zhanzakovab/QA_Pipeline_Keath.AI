import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent

# Data directories
DATA_DIR = BASE_DIR / "dataset"
TEST_DIR = DATA_DIR / "test_preprocessed"
TEST_DISCUSSION_DIR = DATA_DIR / "test_discussion"

# Input files
INPUT_FILE_REAL = DATA_DIR / "keath_dataset.json"
INPUT_FILE_SAMPLE = DATA_DIR / "keath_dataset_sample.json"

# Output files
OUTPUT_FILE_REAL = DATA_DIR / "data_with_ids.json"
DUPLICATES_FILE_REAL = DATA_DIR / "discussion" / "duplicates_grouped.json"
OUTPUT_FILE_TEST = TEST_DIR / "sample_data_with_ids.json"
DUPLICATES_FILE_TEST = TEST_DISCUSSION_DIR / "sample_duplicates.json"

# Evaluation files
EVALUATIONS_DIR = DATA_DIR / "evaluations"
TEST_EVALUATIONS_DIR = DATA_DIR / "test_evaluations"

# Script paths
TEST_EVALUATION_SCRIPT = BASE_DIR / "scripts" / "test_evaluation.py"

# Evaluation settings
EVALUATION_THRESHOLD = 0.7
BATCH_SIZE = 10

# Safety settings
USE_SAMPLE_DATA_BY_DEFAULT = True