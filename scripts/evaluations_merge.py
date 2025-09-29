import json
import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

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


def prepare_evaluation_dataframe(df: pd.DataFrame, metric_name: str) -> pd.DataFrame:
    # Prepare evaluation DataFrame for merging by renaming columns
    df_copy = df.copy()
    
    # Rename columns based on the metric
    if metric_name in ['actionability', 'specificity', 'rubric_alignment']:
        df_copy.rename(columns={'actual_output': 'comment'}, inplace=True)
    elif metric_name == 'factual_consistency':
        df_copy.rename(columns={'input': 'comment'}, inplace=True)
    
    # Add metric name column
    df_copy['metric'] = metric_name
    
    return df_copy


def merge_evaluation_with_master(
    evaluation_df: pd.DataFrame, 
    master_df: pd.DataFrame, 
    metric_name: str
) -> pd.DataFrame:
    # Merge evaluation results with master data
    logger.info(f"Merging {metric_name} results with master data...")
    
    # Prepare the evaluation DataFrame
    prepared_eval_df = prepare_evaluation_dataframe(evaluation_df, metric_name)
    
    # Merge on 'comment' column
    merged_df = pd.merge(prepared_eval_df, master_df, on='comment', how='left')
    
    # Sort and reset index
    merged_df.sort_values(['assignment_id', 'item'], inplace=True)
    merged_df.reset_index(drop=True, inplace=True)
    
    # Rename score columns for clarity
    if 'score_x' in merged_df.columns and 'score_y' in merged_df.columns:
        merged_df.rename(columns={'score_x': 'judge_score'}, inplace=True)
        merged_df.rename(columns={'score_y': 'grader_score'}, inplace=True)
    
    logger.info(f"Successfully merged {len(merged_df)} {metric_name} records")
    return merged_df


def clean_merged_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    # Clean and standardize merged DataFrame columns
    # Define final columns to keep
    final_columns_to_keep = [
        'assignment_id',
        'paper_content',
        'item',
        'type', 
        'grader_score',
        'comment',
        'evidence',
        'score_range',
        'rubric',
        'metric',
        'judge_score',
        'status',
        'reason'
    ]
    
    # Keep only the columns that exist in the DataFrame
    available_columns = [col for col in final_columns_to_keep if col in df.columns]
    cleaned_df = df[available_columns].copy()
    
    return cleaned_df


def save_merged_results(merged_results: Dict[str, pd.DataFrame], output_dir: str) -> None:
    # Save merged results to JSON files
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for metric_name, df in merged_results.items():
        output_file = output_path / f"merged_{metric_name}.json"
        try:
            df.to_json(output_file, orient='records', indent=4)
            logger.info(f"Saved {metric_name} results to: {output_file}")
        except Exception as e:
            logger.error(f"Could not save {metric_name} results. Error: {e}")


def merge_evaluation_results(
    master_data_dir: str = "dataset/preprocessed",
    evaluations_dir: str = "dataset/evaluations",
    output_dir: str = "dataset/merged",
    metrics: Optional[List[str]] = None
) -> Dict[str, pd.DataFrame]:
    # Merge evaluation results with master data
    logger.info("Starting evaluation results merging...")
    
    # Default metrics to process
    if metrics is None:
        metrics = ['actionability', 'specificity', 'rubric_alignment', 'factual_consistency']
    
    # File paths
    master_df_file = os.path.join(master_data_dir, 'master_df.json')
    master_second_df_file = os.path.join(master_data_dir, 'master_second_df.json')
    
    # Load master data
    logger.info("Loading master data...")
    try:
        master_df = pd.DataFrame(load_json_data(master_df_file))
        master_second_df = pd.DataFrame(load_json_data(master_second_df_file))
        logger.info(f"Loaded master data: {len(master_df)} and {len(master_second_df)} records")
    except FileNotFoundError as e:
        logger.error("Master data files not found. Please run preprocessing first.")
        raise
    
    merged_results = {}
    
    # Process each metric
    for metric in metrics:
        try:
            # Load evaluation results
            eval_file = os.path.join(evaluations_dir, f'{metric}_results.json')
            evaluation_df = pd.DataFrame(load_json_data(eval_file))
            
            # Choose appropriate master data based on metric
            if metric in ['actionability', 'specificity']:
                master_data = master_df
            else:  # rubric_alignment, factual_consistency
                master_data = master_second_df
            
            # Merge the data
            merged_df = merge_evaluation_with_master(evaluation_df, master_data, metric)
            
            # Clean the merged data
            cleaned_df = clean_merged_dataframe(merged_df)
            
            merged_results[metric] = cleaned_df
            
            # Display info
            logger.info(f"{metric.upper()} Results:")
            logger.info(f"  Records: {len(cleaned_df)}")
            logger.info(f"  Columns: {list(cleaned_df.columns)}")
            logger.info("")
            
        except FileNotFoundError:
            logger.warning(f"{metric} results file not found, skipping...")
        except Exception as e:
            logger.error(f"Failed to merge {metric} results: {e}")
    
    # Save results
    if merged_results:
        save_merged_results(merged_results, output_dir)
        logger.info(f"Successfully merged {len(merged_results)} metrics")
    else:
        logger.info("No results to merge")
    
    return merged_results


def main():
    # Main function to run the evaluation merging pipeline
    logger.info("KEATH Evaluation Results Merging Pipeline")
    
    try:
        results = merge_evaluation_results()
        
        if results:
            logger.info(f"Success! Merged {len(results)} evaluation metrics.")
            logger.info("Results saved to: dataset/merged/")
            logger.info("Merged files:")
            for metric in results.keys():
                logger.info(f"  - merged_{metric}.json")
        else:
            logger.info("No results were merged. Check if evaluation files exist.")
            return 1
        
        return 0
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())