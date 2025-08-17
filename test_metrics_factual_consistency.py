import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import GEval
from deepeval.dataset import EvaluationDataset
from deepeval.test_case import LLMTestCaseParams
from dotenv import load_dotenv
from deepeval.metrics import HallucinationMetric
from deepeval.evaluate import evaluate
import pandas as pd
import os

# Setup 
load_dotenv()

# Load the dataset
df = pd.read_json("dataset/preprocessed/master_second_df_preview.json", orient='records')


# --- Function to create test cases from the DataFrame ---
# This function can be reused for different metrics if needed
def create_test_cases(df: pd.DataFrame, 
                      input_col: str = None, 
                      actual_output_col: str = None, 
                      context_cols: list = None,
                      retrieval_context_cols: list = None) -> list[LLMTestCase]:
  
    test_cases = []
    for index, row in df.iterrows():
        
        # Build the arguments for the LLMTestCase dynamically
        case_args = {}
        
        if input_col:
            case_args['input'] = row[input_col]
        if actual_output_col:
            case_args['actual_output'] = row[actual_output_col]
        
        # If a list of context columns is provided, combine them
        if context_cols:
            case_args['context'] = [row[col] for col in context_cols]
            
        # Do the same for retrieval_context
        if retrieval_context_cols:
            case_args['retrieval_context'] = [row[col] for col in retrieval_context_cols]

        # Create the test case with whatever arguments we've built
        test_case = LLMTestCase(**case_args)
        test_cases.append(test_case)
        
    return test_cases


# --- FACTUAL CONSISTENCY ---


# Create test cases 
factual_consistency_test_cases = create_test_cases(
    df=df,
    input_col='comment',
    actual_output_col='evidence',
    context_cols=['paper_content']
)


# Define the Metric
factual_consistency_metric = GEval(
    name="Factual Consistency",
    evaluation_steps=[
         'You are a meticulous educational assessor. Your job is to verify the accuracy of the feedback.',
         'First, read the feedback evidence provided in the "actual_output" and i identify the key factual claims it makes.',
         'Second, review the submitted paper provided in "context" and locate the specific information related to each claim.',
         'Third, for each factual claim, determine whether it is fully supported by, directly extracted from, or correctly paraphrased based on the source document.',
         'Based on the three checks above, synthesize your findings into a single, fine-grained score from 0.0 to 1.0. A score of 1.0 requires the claim to be direct, faithful, AND accurate representation of the source text. A score of 0.0 should be given IF there is a significant contradiction.',
         'Explain your reasoning in a clear, professional sentence. CRITICAL: Do NOT use the words "actual_output", "context", or "output" in your justification. Instead, use human-readable terms like "The feedback evidence" and "The student paper".'
    ],
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.CONTEXT],
    model="gpt-4o",
    threshold=0.7  
)

# Run the evaluation
factual_consistency_results = evaluate(
    test_cases=factual_consistency_test_cases,
    metrics=[factual_consistency_metric]
)

# --- Process the Results to Create a Clean DataFrame ---
def process_results(results_list: list) -> pd.DataFrame:
    results_data = []

    # Handle different possible top-level result structures
    test_results = results_list.test_results if hasattr(results_list, 'test_results') else results
    
    for i, test_result in enumerate(test_results):
        # This handles the issue where results might be a list of tuples
        result_object = test_result[0] if isinstance(test_result, tuple) else test_result

        # Check if it's a valid result object
        if not hasattr(result_object, 'input'):
            print(f"Skipping malformed result at index {i}: {result_object}")
            continue

        # --- Base data from the test case ---
        row_data = {
            'input': result_object.input,
            'actual_output': result_object.actual_output,
        }

        # Check for context and retrieval_context 
        if hasattr(result_object, 'context') and result_object.context is not None:
            # Join list into a single string for easier CSV viewing
            row_data['context'] = " | ".join(result_object.context)
            
        if hasattr(result_object, 'retrieval_context') and result_object.retrieval_context is not None:
            row_data['retrieval_context'] = " | ".join(result_object.retrieval_context)


        # Metric-specific results 
        if hasattr(result_object, 'metrics_data'):
            for metric_data in result_object.metrics_data:
                metric_name = metric_data.name.replace(" ", "_").lower()
                row_data[f'metric'] = metric_data.name
                row_data[f'score'] = metric_data.score
                row_data[f'status'] = "passed" if metric_data.success else "failed"
                row_data[f'reason'] = metric_data.reason
        
        results_data.append(row_data)

    # Return the clean DataFrame
    return pd.DataFrame(results_data)


# Call the new function to get your results DataFrame
factual_consistency_results_df = process_results(factual_consistency_results)


# Hallucination Results to JSON
factual_consistency_results_df.to_json('dataset/factual_consistency_results.json', orient='records', indent=4)
print("Report successfully saved")
