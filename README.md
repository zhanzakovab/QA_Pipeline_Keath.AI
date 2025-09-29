# KEATH.AI QA Pipeline
**Automated AI Feedback Quality Assurance System**

## The Problem
KEATH.AI's AI feedback felt unreliable — vague, inconsistent, sometimes misleading. This threatened both trust (teachers and students wouldn't adopt) and growth (hard to sell without proof of quality).

## The Solution
An evaluation pipeline where a Judge LLM scores feedback across four metrics: **Actionability**, **Specificity**, **Factual Consistency**, and **Rubric Alignment**. This turns "quality" from an opinion into a repeatable measurement system.

## What Changed
KEATH.AI now has baseline product metrics:
- **Actionability**: 65–70%
- **Specificity**: ≈70% 
- **Factual Consistency & Rubric Alignment**: 80–85%

They can track these numbers across assignments, cohorts, and updates — the same way SaaS companies track retention or NPS as standard product health metrics. This gives the product team both a marketing position ("we don't just give feedback — we measure its quality") and a growth lever (clear targets to improve trust, adoption, and sales).

## Key Features
- **Robust Data Preprocessing**: Clean, normalize, and structure raw JSON data
- **Hybrid Evaluation Framework**: Custom metrics measuring qualitative and quantitative aspects
- **LLM-as-a-Judge**: GPT-4o with G-Eval prompting for nuanced academic scoring
- **Production-Ready Pipeline**: Logging, error handling, and Docker containerization
- **Automated Testing**: DeepEval framework for systematic evaluation and reporting

## Quick Start

### With Docker (Recommended)
```bash
# Run full pipeline
docker-compose up keath-pipeline

# Run just preprocessing
docker-compose up keath-preprocessing
```

### Manual Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Run full pipeline
python scripts/run_full_pipeline.py

# Run individual steps
python scripts/run_preprocessing.py
python scripts/test_evaluation.py
python scripts/run_merge.py
```

## Project Structure
```
keath_pipeline/
├── scripts/           # Core pipeline modules
├── dataset/           # Data and evaluation results
├── config.py         # Centralized configuration
├── Dockerfile        # Container configuration
└── docker-compose.yml # Easy deployment
```

## Current Challenges
- **Cost**: LLM evaluation expenses
- **Latency**: Processing time for large datasets
- **Human Evaluation**: Validation methodology refinement needed
