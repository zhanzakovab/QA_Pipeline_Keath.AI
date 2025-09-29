#!/usr/bin/env python3

import argparse
import subprocess
import sys
import logging
from pathlib import Path

# Add the scripts directory to the path
sys.path.append(str(Path(__file__).parent))

from data_preprocessing import process_data_pipeline
from evaluations_merge import merge_evaluation_results
import config

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


def run_command(command: list, description: str) -> bool:
    # Run a command and return success status
    logger.info(f"{description}...")
    logger.info(f"Running: {' '.join(command)}")
    
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        logger.info(f"{description} completed successfully")
        if result.stdout:
            logger.info(f"Output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"{description} failed")
        logger.error(f"Error: {e.stderr}")
        return False
    except FileNotFoundError:
        logger.error(f"Command not found: {command[0]}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run complete KEATH pipeline")
    
    parser.add_argument(
        "--real-data",
        action="store_true",
        help="Use real dataset instead of sample data (overwrites existing files)"
    )
    
    parser.add_argument(
        "--skip-preprocessing",
        action="store_true",
        help="Skip data preprocessing step"
    )
    
    parser.add_argument(
        "--skip-evaluations",
        action="store_true",
        help="Skip evaluation step"
    )
    
    parser.add_argument(
        "--skip-merge",
        action="store_true",
        help="Skip merge step"
    )
    
    args = parser.parse_args()
    
    logger.info("Starting KEATH Complete Pipeline")
    logger.info("=" * 50)
    
    if args.real_data:
        logger.warning("Using real dataset - this may overwrite existing files!")
        input("Press Enter to continue or Ctrl+C to cancel...")
        data_mode = "real"
    else:
        logger.info("Using sample data (safe mode)")
        data_mode = "sample"
    
    success_count = 0
    total_steps = 0
    
    # Step 1: Data Preprocessing
    if not args.skip_preprocessing:
        total_steps += 1
        logger.info(f"Step 1: Data Preprocessing ({data_mode} data)")
        
        try:
            if data_mode == "real":
                final_df = process_data_pipeline(
                    input_file="dataset/keath_dataset.json",
                    output_file="dataset/data_with_ids.json",
                    duplicates_file="dataset/discussion/duplicates_grouped.json"
                )
            else:
                final_df = process_data_pipeline()
            
            logger.info(f"Preprocessing completed: {len(final_df)} records processed")
            success_count += 1
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            return 1
    else:
        logger.info("Skipping data preprocessing")
    
    # Step 2: Run Evaluations
    if not args.skip_evaluations:
        total_steps += 1
        logger.info("Step 2: Running Evaluations")
        
        # Check if test_evaluation.py exists
        if not config.TEST_EVALUATION_SCRIPT.exists():
            logger.error("test_evaluation.py not found in scripts directory")
            return 1
        
        if run_command(["python", str(config.TEST_EVALUATION_SCRIPT)], "Running evaluations"):
            success_count += 1
        else:
            logger.error("Evaluations failed")
            return 1
    else:
        logger.info("Skipping evaluations")
    
    # Step 3: Merge Results
    if not args.skip_merge:
        total_steps += 1
        logger.info("Step 3: Merging Results")
        
        try:
            if data_mode == "real":
                results = merge_evaluation_results(
                    master_data_dir="dataset/preprocessed",
                    evaluations_dir="dataset/evaluations",
                    output_dir="dataset/merged"
                )
            else:
                results = merge_evaluation_results(
                    master_data_dir="dataset/test_preprocessed",
                    evaluations_dir="dataset/test_evaluations",
                    output_dir="dataset/test_merged"
                )
            
            if results:
                logger.info(f"Merging completed: {len(results)} metrics merged")
                success_count += 1
            else:
                logger.info("No results to merge")
                return 1
        except Exception as e:
            logger.error(f"Merging failed: {e}")
            return 1
    else:
        logger.info("Skipping merge")
    
    # Summary
    logger.info("=" * 50)
    logger.info("PIPELINE SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Completed: {success_count}/{total_steps} steps")
    
    if success_count == total_steps:
        logger.info("Pipeline completed successfully!")
        logger.info("Output files:")
        
        if data_mode == "real":
            logger.info("  - dataset/data_with_ids.json")
            logger.info("  - dataset/evaluations/*_results.json")
            logger.info("  - dataset/merged/merged_*.json")
        else:
            logger.info("  - dataset/test_preprocessed/sample_data_with_ids.json")
            logger.info("  - dataset/test_evaluations/*_results.json")
            logger.info("  - dataset/test_merged/merged_*.json")
        
        logger.info("Next steps:")
        logger.info("  1. Review results in the merged files")
        logger.info("  2. Run analysis: jupyter notebook notebooks/evaluations_analysis.ipynb")
        logger.info("  3. Generate reports and insights")
        
        return 0
    else:
        logger.error("Pipeline completed with errors")
        return 1


if __name__ == "__main__":
    exit(main())