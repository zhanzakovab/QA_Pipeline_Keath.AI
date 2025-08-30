# AI Quality Assurance in EdTech: Evaluating Automated Feedback  

This project develops and evaluates a **scalable AI Quality Assurance (QA) pipeline** for automated feedback systems in education, built in collaboration with **KEATH.AI**. The goal is to ensure that AI-generated feedback is **actionable, specific, rubric-aligned, and factually consistent** — moving beyond grading speed to **feedback quality at scale**.  

## What It Does  
- Processes **8,600+ AI-generated feedback items** across multiple assignments  
- Evaluates feedback quality using four key metrics:  
  **Actionability · Specificity · Rubric Alignment · Factual Consistency**  
- Leverages **Judge LLMs** to systematically assess outputs from **Grader LLMs**  
- Provides detailed analytics: pass/fail breakdowns, correlations, and failure patterns  
- Designed to be **extensible** for future QA workflows and multi-LLM evaluation setups  

## Tech Highlights  
- **Python-based evaluation framework**: `pandas`, `deepeval`, `GEval`, `HallucinationMetric`  
- Custom **data pipelines** for merging, cleaning, and analyzing ~8.6k records  
- Advanced **visualizations**: stacked bar charts, correlation heatmaps, bias analysis  
- Fully reproducible, modular codebase — easy to adapt to other EdTech QA contexts  

## Why It Matters  
AI feedback systems are becoming integral to learning, but **low-quality feedback can undermine trust and widen learning gaps**.  
This project demonstrates a **practical, scalable solution**: automating QA for AI-generated feedback while highlighting **systemic biases**, dataset risks, and improvement opportunities.  
