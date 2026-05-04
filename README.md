# WFT Digital Medicine

*Last Update: 2026-05*

## Project Overview

This project provides a series of Jupyter notebooks for **medical students** learning Python programming, data science, NLP, and AI applications in healthcare. All notebooks are designed to run on **Google Colab** (free tier) without requiring API keys.

## Quick Start

1. Open the [Starting Notebook](https://colab.research.google.com/github/MoLue/wft_digital_medicine/blob/main/starting_notebook.ipynb) in Google Colab
2. Pick a learning path based on your experience level
3. Run cells top-to-bottom (Shift+Enter)

## Notebook Catalog

| # | Notebook | Level | Duration | Tags |
|---|----------|-------|----------|------|
| 1 | [Python Basics (Compact)](https://colab.research.google.com/github/MoLue/wft_digital_medicine/blob/main/python_basics_compact.ipynb) | Beginner | 30–60 min | `python`, `pandas`, `visualization`, `compact` |
| 2 | [Medical Data Science Fundamentals](https://colab.research.google.com/github/MoLue/wft_digital_medicine/blob/main/medical_data_science_fundamentals.ipynb) | Beginner | 90–180+ min | `python`, `pandas`, `data-cleaning`, `logistic-regression` |
| 3 | [Heart Disease Prediction Analysis](https://colab.research.google.com/github/MoLue/wft_digital_medicine/blob/main/heart_disease_prediction_analysis.ipynb) | Intermediate | 60–120 min | `ml-classification`, `visualization`, `sklearn`, `xgboost` |
| 4 | [NLP and Transformers](https://colab.research.google.com/github/MoLue/wft_digital_medicine/blob/main/dm_nlp.ipynb) | Intermediate | 45–90 min | `nlp`, `spacy`, `transformers`, `ner`, `qa` |
| 5 | [Introduction to LLMs in Healthcare](https://colab.research.google.com/github/MoLue/wft_digital_medicine/blob/main/llms_in_healthcare.ipynb) | Advanced | 60–120 min | `llm`, `biogpt`, `prompt-engineering`, `rag` |
| 6 | [Agentic AI in Medicine](https://colab.research.google.com/github/MoLue/wft_digital_medicine/blob/main/agentic_ai_medical.ipynb) | Mixed | 30–60 min | `agentic-ai`, `agents`, `rag`, `multi-agent`, `safety`, `compact` |
| 7 | [LLM Agents for Healthcare Admin](https://colab.research.google.com/github/MoLue/wft_digital_medicine/blob/main/llm_agents_healthcare_agents.ipynb) | Advanced | 60–120+ min | `agents`, `biogpt`, `biobert`, `routing`, `guardrails` |

> **Note on duration:** The lower end is for experienced programmers; the upper end accounts for exploring exercises, experimenting, and reading background material. Take your time — there's no rush!

**Level Guide:**
- **Beginner** — No programming experience required
- **Intermediate** — Basic Python knowledge helpful
- **Advanced** — Comfortable with Python and libraries like pandas/sklearn
- **Mixed** — Layered difficulty; beginners focus on early parts, advanced go further

### Compact vs. Full Notebooks

- **Compact** (#1, #6): Shorter, focused introductions — great for getting started quickly
- **Full** (#2, #3, #4, #5, #7): In-depth self-paced learning with more exercises and exploration

## Data

- `data/heart.csv` — Heart disease dataset from Kaggle (ODbL license)
- `data/synthetic_diabetes.csv` — Synthetic diabetes dataset for educational purposes
- `medical_kb/` — Medical condition texts for RAG demonstrations

## Additional Resources

### Streamlit Demo
`streamlit_multi_agents_diabetes/` — A multi-agent Streamlit demo for diabetes case conferences

### Cheat Sheets
- [Pandas](https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf)
- [Seaborn Charts](https://www.kaggle.com/code/themlphdstudent/cheat-sheet-seaborn-charts)

## Technical Requirements

All notebooks run on **Google Colab free tier**:
- No local installation needed
- No API keys required (open-source models only)
- GPU optional but recommended for LLM notebooks

> **Disclaimer:** All notebooks are for educational purposes only and must not be used for real patient care or clinical decision-making.