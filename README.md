# Spam Filter Project

Project structure:
- data/: raw and processed data
- spam_filter_project/: python package with reusable code
- notebooks/: EDA and model development notebooks
- models/: saved model/vectorizer (do not include large binaries in git)
- reports/: final report and figures

## Quick start

1. Create and activate venv:
   `python -m venv .venv; .\.venv\Scripts\Activate.ps1`
2. Install requirements:
   `pip install -r requirements.txt`
3. Run notebooks (JupyterLab) or run streamlit demo:
   `streamlit run spam_filter_project/deployment.py`
