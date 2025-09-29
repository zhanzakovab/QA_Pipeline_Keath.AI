import json
import os
import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('keath_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_json_data(filepath: str) -> List[Dict[str, Any]]:
    # Load JSON data from a file
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            report_data = json.load(f)

        if report_data:
            if isinstance(report_data, list):
                logger.info(f"Successfully loaded JSON. Found {len(report_data)} records.")
            else:
                logger.info(f"Loaded JSON, but it's not a list. Type: {type(report_data)}. Handling appropriately.")
                report_data = [report_data] if isinstance(report_data, dict) else []
        else:
            logger.warning("JSON file was loaded but appears to be empty or None.")
            report_data = []
        return report_data

    except FileNotFoundError:
        logger.error(f"File not found at '{filepath}'. Please check the path.")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Could not decode JSON from file {filepath}. The file may be corrupted. Error: {e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred during file loading: {e}")
        raise


def validate_data(df: pd.DataFrame) -> List[int]:
    # Validate data and identify non-substantive records
    # Check: Content Uniqueness
    if 'paper_content' in df.columns:
        total_records = len(df)
        unique_content_records = df['paper_content'].nunique()
        if total_records == unique_content_records:
            logger.info("Uniqueness Check: All 'paper_content' entries are unique.")
        else:
            duplicate_count = total_records - unique_content_records
            logger.info(f"Uniqueness Check: Found {duplicate_count} duplicate 'paper_content' entries.")
    else:
        logger.error("'paper_content' column not found for uniqueness check.")

    # Check: Substantive Content
    flagged_for_review = []
    for i, row in df.iterrows():
        content = row.get('paper_content', '')
        declared_match = re.search(r'word count:?\s*(\d{1,5})', content, re.IGNORECASE)
        declared_count = int(declared_match.group(1)) if declared_match else None
        actual_count = len(re.findall(r'\w+', content))
        
        is_substantive = True
        reason = ""

        if declared_count is not None:
            difference_ratio = actual_count / declared_count if declared_count > 0 else 0
            if difference_ratio < 0.80:
                is_substantive = False
                reason = f"Huge discrepancy -> Declared: {declared_count}, Actual: {actual_count}"
        elif actual_count < 300:
            is_substantive = False
            reason = f"No declared word count and content is very short -> Actual: {actual_count}"
        
        if not is_substantive:
            flagged_for_review.append({'index': i, 'reason': reason})
            
    if flagged_for_review:
        logger.info(f"Substantive Content Check: Found {len(flagged_for_review)} records that may be non-substantive.")
        logger.info("Details of Flagged Non-Substantive Records")
        for flag in flagged_for_review:
             logger.info(f"  - Index {flag['index']}: {flag['reason']}")
    else:
        logger.info("Substantive Content Check: All records appear to have substantive content.")
    
    # Return just the list of indices to be dropped
    return [flag['index'] for flag in flagged_for_review]


def deduplicate_data(df: pd.DataFrame) -> pd.DataFrame:
    # Remove duplicate records based on 'paper_content' column
    if 'paper_content' not in df.columns:
        logger.error("Cannot de-duplicate: 'paper_content' column missing.")
        return df
        
    records_before = len(df)
    clean_df = df.drop_duplicates(subset=['paper_content'], keep='first').copy()
    records_after = len(clean_df)
    logger.info(f"Removed {records_before - records_after} duplicate records. {records_after} unique records remain.")
    return clean_df


def assign_sequential_ids(df: pd.DataFrame) -> pd.DataFrame:
    # Assign sequential IDs to each record
    logger.info("Assigning Final Sequential IDs")
    # Reset the index to ensure it's a clean sequence from 0 to N-1
    df_with_new_index = df.reset_index(drop=True)
    # Assign the new index as the 'assignment_id'
    df_with_new_index['assignment_id'] = df_with_new_index.index
    logger.info(f"Assigned new 'assignment_id' to {len(df_with_new_index)} records.")
    return df_with_new_index


def export_duplicates(df: pd.DataFrame, output_path: str) -> None:
    # Export duplicate records for analysis
    logger.info("Exporting Duplicates for Analysis")
    duplicate_mask = df.duplicated(subset=['paper_content'], keep=False)
    duplicates_df = df[duplicate_mask]

    if not duplicates_df.empty:
        sorted_duplicates_df = duplicates_df.sort_values(by='paper_content').reset_index(drop=True)
        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        sorted_duplicates_df.to_json(output_path, orient='records', indent=4)
        logger.info(f"Analysis file of {len(sorted_duplicates_df)} duplicate records saved to '{output_path}'")
    else:
        logger.info("No duplicate records found to export.")


def process_data_pipeline(
    input_file: str = "dataset/keath_dataset_sample.json",
    output_file: str = "dataset/test_preprocessed/sample_data_with_ids.json",
    duplicates_file: Optional[str] = "dataset/test_discussion/sample_duplicates.json"
) -> pd.DataFrame:
    # Run the complete data preprocessing pipeline
    logger.info("Starting data preprocessing pipeline...")
    logger.info(f"Input file: {input_file}")
    logger.info(f"Output file: {output_file}")
    
    # Load the raw data
    raw_data = load_json_data(input_file)
    
    if not raw_data:
        raise ValueError("No data loaded from input file")
    
    main_df = pd.DataFrame(raw_data)
    logger.info(f"Created DataFrame with {len(main_df)} records")
    
    # Perform validation checks and get indices of non-substantive records
    non_substantive_indices = validate_data(main_df)
    
    # Filter out non-substantive records
    logger.info("Filtering Non-Substantive Records")
    if non_substantive_indices:
        substantive_df = main_df.drop(non_substantive_indices).copy()
        logger.info(f"Removed {len(non_substantive_indices)} records. Proceeding with {len(substantive_df)} records.")
    else:
        substantive_df = main_df.copy()
        logger.info("No non-substantive records to remove.")

    # Export duplicates for analysis if requested
    if duplicates_file:
        export_duplicates(substantive_df, duplicates_file)

    # Clean the data by de-duplicating
    deduplicated_df = deduplicate_data(substantive_df)
    
    # Assign final IDs to the clean, de-duplicated data
    final_df = assign_sequential_ids(deduplicated_df)

    # Save the final clean file
    try:
        # Ensure output directory exists
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        final_df.to_json(output_file, orient='records', indent=4)
        logger.info(f"Final clean data successfully saved to: '{output_file}'")
    except Exception as e:
        logger.error(f"Could not save the clean file. Error: {e}")
        raise
    
    logger.info("Data preprocessing pipeline completed successfully!")
    return final_df


def main():
    # Main function to run the data preprocessing pipeline with sample data
    logger.info("KEATH Data Preprocessing Pipeline")
    logger.info("Using sample data by default for safety...")
    
    try:
        final_df = process_data_pipeline()
        logger.info(f"Success! Processed {len(final_df)} records.")
        logger.info("Output saved to: dataset/test_preprocessed/sample_data_with_ids.json")
        logger.info("Duplicates analysis saved to: dataset/test_discussion/sample_duplicates.json")
        
        return 0
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())