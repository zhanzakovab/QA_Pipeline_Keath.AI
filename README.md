# KEATH.AI QA Pipeline
AI Feedback Quality Assurance Pipeline
This project is an end-to-end system for the automated evaluation of LLM-generated educational feedback. It serves as the basis for a Master's dissertation in Business Analytics at UCL and as a proof-of-concept for a scalable QA system for the EdTech company, Keath.AI.

The pipeline ingests raw, complex JSON data containing student work, AI-generated feedback, and associated rubrics. It then uses a sophisticated "LLM-as-a-Judge" (GPT-4o) within the DeepEval framework to audit the feedback against a hybrid evaluation framework grounded in pedagogical theory.

**Key Features**
Robust Data Preprocessing: A Python pipeline using Pandas to clean, normalize, and structure raw, nested JSON data into a test-ready format.

Hybrid Evaluation Framework: A suite of custom metrics designed to measure both the qualitative and quantitative aspects of AI feedback.

LLM-as-a-Judge: Utilizes GPT-4o and the G-Eval prompting technique to score feedback on nuanced, academic criteria.

Automated Testing: Leverages the DeepEval framework to systematically run evaluations, track results, and provide detailed reports.