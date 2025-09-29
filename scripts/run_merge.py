#!/usr/bin/env python3

import argparse
import sys
import logging
from pathlib import Path

# Add the scripts directory to the path so we can import our module
sys.path.append(str(Path(__file__).parent))

from evaluations_merge import merge_evaluation_results

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
    parser = argparse.ArgumentParser(description="Merge KEATH evaluation results with master data")
    
    parser.add_argument(
        "--master-dir",
        default="dataset/preprocessed",
        help="Directory containing master data files (default: dataset/preprocessed)"
    )
    
    parser.add_argument(
        "--evaluations-dir",
        default="dataset/evaluations", 
        help="Directory containing evaluation results (default: dataset/evaluations)"
    )
    
    parser.add_argument(
        "--output-dir",
        default="dataset/merged",
        help="Directory to save merged results (default: dataset/merged)"
    )
    
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=['actionability', 'specificity', 'rubric_alignment', 'factual_consistency'],
        help="Specific metrics to merge (default: all available)"
    )
    
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Use test data directories (dataset/test_*)"
    )
    
    args = parser.parse_args()
    
    # Adjust paths for test mode
    if args.test_mode:
        args.master_dir = "dataset/test_preprocessed"
        args.evaluations_dir = "dataset/test_evaluations"
        args.output_dir = "dataset/test_merged"
        logger.info("Test mode: Using test data directories")
    
    # Check if directories exist
    master_path = Path(args.master_dir)
    evaluations_path = Path(args.evaluations_dir)
    
    if not master_path.exists():
        logger.error(f"Master data directory '{args.master_dir}' does not exist!")
        logger.error("   Run preprocessing first: python scripts/run_preprocessing.py")
        return 1
    
    if not evaluations_path.exists():
        logger.error(f"Evaluations directory '{args.evaluations_dir}' does not exist!")
        logger.error("   Run evaluations first: python scripts/test_evaluation.py")
        return 1
    
    try:
        logger.info("Starting KEATH evaluation merging...")
        logger.info(f"Master data: {args.master_dir}")
        logger.info(f"Evaluations: {args.evaluations_dir}")
        logger.info(f"Output: {args.output_dir}")
        
        if args.metrics:
            logger.info(f"Metrics: {', '.join(args.metrics)}")
        else:
            logger.info("Metrics: all available")
        
        # Run the merging
        results = merge_evaluation_results(
            master_data_dir=args.master_dir,
            evaluations_dir=args.evaluations_dir,
            output_dir=args.output_dir,
            metrics=args.metrics
        )
        
        if results:
            logger.info(f"Success! Merged {len(results)} evaluation metrics.")
            logger.info(f"Results saved to: {args.output_dir}/")
            logger.info("Merged files:")
            for metric in results.keys():
                logger.info(f"  - merged_{metric}.json ({len(results[metric])} records)")
            
            logger.info("Next steps:")
            logger.info("   1. Review merged results")
            logger.info("   2. Run analysis: python notebooks/evaluations_analysis.ipynb")
            logger.info("   3. Generate reports")
        else:
            logger.info("No results were merged.")
            logger.info("   Check if evaluation files exist in the evaluations directory")
            return 1
        
        return 0
        
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())