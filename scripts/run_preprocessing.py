#!/usr/bin/env python3

import argparse
import sys
import logging
from pathlib import Path

# Add the scripts directory to the path so we can import our module
sys.path.append(str(Path(__file__).parent))

from data_preprocessing import process_data_pipeline

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


def main():
    parser = argparse.ArgumentParser(description="Run KEATH data preprocessing pipeline")
    
    parser.add_argument(
        "--input", 
        default="dataset/keath_dataset_sample.json",
        help="Input JSON file path (default: dataset/keath_dataset_sample.json - SAFE)"
    )
    
    parser.add_argument(
        "--output", 
        default="dataset/test_preprocessed/sample_data_with_ids.json",
        help="Output JSON file path (default: dataset/test_preprocessed/sample_data_with_ids.json)"
    )
    
    parser.add_argument(
        "--duplicates", 
        default="dataset/test_discussion/sample_duplicates.json",
        help="Duplicates analysis file path (default: dataset/test_discussion/sample_duplicates.json)"
    )
    
    parser.add_argument(
        "--no-duplicates", 
        action="store_true",
        help="Skip exporting duplicates analysis"
    )
    
    parser.add_argument(
        "--force-real-data",
        action="store_true",
        help="Allow processing of real dataset (overwrites existing files)"
    )
    
    args = parser.parse_args()
    
    # Safety check for real data
    if "keath_dataset.json" in args.input and not args.force_real_data:
        logger.warning("You're about to process the real dataset!")
        logger.warning("   This might overwrite existing evaluation results.")
        logger.warning("   Use --force-real-data flag if you're sure.")
        logger.warning("   Or use sample data: python scripts/run_preprocessing.py")
        return 1
    
    # Check if input file exists
    if not Path(args.input).exists():
        logger.error(f"Input file '{args.input}' does not exist!")
        logger.error("   Available files:")
        dataset_dir = Path("dataset")
        if dataset_dir.exists():
            for file in dataset_dir.glob("*.json"):
                logger.error(f"   - {file}")
        return 1
    
    try:
        logger.info("Starting KEATH data preprocessing...")
        logger.info(f"Input: {args.input}")
        logger.info(f"Output: {args.output}")
        
        # Run the pipeline
        final_df = process_data_pipeline(
            input_file=args.input,
            output_file=args.output,
            duplicates_file=None if args.no_duplicates else args.duplicates
        )
        
        logger.info(f"Success! Processed {len(final_df)} records.")
        logger.info(f"Output saved to: {args.output}")
        if not args.no_duplicates:
            logger.info(f"Duplicates analysis saved to: {args.duplicates}")
        
        logger.info("Next steps:")
        logger.info("   1. Test with sample data first")
        logger.info("   2. Run evaluations: python scripts/test_evaluation.py")
        logger.info("   3. Merge results: python scripts/run_merge.py")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())