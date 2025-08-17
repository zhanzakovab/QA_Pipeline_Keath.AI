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
df = pd.read_json("dataset/master_df.json", orient='records')


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


# # --- RUBRIC ALIGNMENT METRIC ---


# Create test cases 
rubric_alignment_test_cases = create_test_cases(
    df=df,
    input_col='paper_content',
    actual_output_col='comment',
    context_cols=['rubric'],
)

# Define the Metric
rubric_alignment_metric = GEval(
    name="Rubric Alignment",
    evaluation_steps=[
         'You are a meticulous educational assessor. Your job is to ensure fairness and consistency of the feedback by strictly adhering to the provided rubric.',
         'First, begin by carefully reading the rubric description in the "context", which defines the non-negotiable standard for this evaluation.',
         'Second, read the qualitative feedback comment given by an AI Grader in the "actual_output"',
         'Third, critically assess whether the feedback comment is logically consistent with the rubric description. For example, feedback aligned to a high (e.g., "Excellent") rubric level should primarily highlight strengths and avoid focusing on deficiencies. Conversely, feedback aligned to a low rubric level should not be predominantly positive or uncritical.',
         'Based on the three checks above, synthesize your findings into a single, fine-grained score from 0.0 to 1.0. A score of 1.0 requires the feedback to be fair AND logical representation of the rubric description. A score of 0.0 should be given IF there is a significant contradiction.',
         'Provide a brief, one-sentence justification for your verdict. CRITICAL: Do NOT use the words "actual_output" or "context" in your justification; instead, use terms like "the feedback comment" and "the rubric description".'
    ],
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.CONTEXT],
    model="gpt-4o",
    threshold=0.7  
)

# Run the evaluation
rubric_results = evaluate(
    test_cases=rubric_alignment_test_cases,
    metrics=[rubric_alignment_metric]
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
rubric_alignment_results_df = process_results(rubric_results)


# Rubric Alignment Results to JSON
rubric_alignment_results_df.to_json('dataset/rubric_alignment_results.json', orient='records', indent=4)
print("Report successfully saved")

