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



# --- SPECIFICITY ---

# Create test cases 
specificity_test_cases = create_test_cases(
    df=df,
    input_col='paper_content',
    actual_output_col='comment',
    # context_cols=['paper_content', 'evidence']
)


# Define the metric
specificity_metric = GEval(
    name="Specificity",
    evaluation_steps=[
        'You are a meticulous educational assessor. Your goal is to judge whether the feedback comment is concise, direct, and clearly linked to specific details in the student submission.',
        'First, review the feedback comment and the student submission. Determine if the comment directly references specific content, examples, or details contained in the student work.',
        'Second, assess whether the feedback is stated succinctly, with no unnecessary elaboration or filler—ideally limited to one or two sentences, unless every part directly addresses identifiable aspects of the submission.',
        'Based on the two checks above, synthesize your findings into a single, fine-grained score from 0.0 to 1.0. A score of 1.0 requires the feedback to be succinct AND points directly to something present in the student work. A score of 0.0 should be given if the feedback is vague, generic, excessively wordy, OR does not reference specific details from the submission.',
        'Provide a brief, one-sentence justification for your verdict, explaining the specific reason for the decision. CRITICAL: Do NOT use the variable name "actual_output" in your justification; refer to it as "the feedback comment" instead.'
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    model="gpt-4o",
    threshold=0.7 
)


# Run the evaluation
specificity_results = evaluate(
    test_cases=specificity_test_cases,
    metrics=[specificity_metric]
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
specificity_results_df = process_results(specificity_results)


# Specificity Results to JSON
specificity_results_df.to_json('dataset/specificity_results.json', orient='records', indent=4)
print("Report successfully saved")
