# 🧳 Wellness Tourism Package — Purchase Prediction (MLOps Pipeline)

<!-- Replace the URL below with your own: go to your repo's Actions tab → click the
     workflow → "..." menu → "Create status badge" → copy the markdown it gives you. -->
[![MLOps Pipeline](https://github.com/<your-username>/<your-repo>/actions/workflows/pipeline.yml/badge.svg)](https://github.com/benitabenony/wellness-tourism-mlops/actions/workflows/pipeline.yml)

Predicts, before a salesperson contacts a customer, whether that customer is likely to
purchase "Visit with Us"'s new **Wellness Tourism Package** — a fully automated GitHub
Actions MLOps pipeline with a Streamlit deployment.

🔗 **Live app:** `<paste your Streamlit app URL here>`

## Everything is documented in one notebook

**`wellness_tourism_full_project.ipynb`** is the complete, executed, end-to-end project
report: business context, data registration, data cleaning, model tuning with MLflow
tracking (every parameter combination logged), model evaluation, the Streamlit
deployment code, and the GitHub Actions pipeline. Every `%%writefile` cell in the
notebook writes the actual pipeline file to disk, so running it top to bottom in Colab
produces a ready-to-push repository.

## App preview

<!-- Drop a screenshot of the deployed app here, e.g.: -->
<!-- ![App screenshot](docs/app_screenshot.png) -->

## Repository structure

```
tourism_project/
├── data/tourism.csv
├── model_building/
│   ├── data_register.py
│   ├── prep.py
│   └── train.py
├── deployment/
│   ├── app.py
│   ├── requirements.txt
│   ├── model.joblib
│   └── metrics.json
├── requirements.txt
.github/workflows/pipeline.yml
wellness_tourism_full_project.ipynb
README.md
```

## Results (Bagging selected as best model)

| Metric | Score |
|---|---|
| Accuracy | 0.930 |
| Precision | 0.916 |
| Recall | 0.703 |
| **F1** | **0.796** |
| ROC-AUC | 0.972 |

Selected from 6 tuned candidate algorithms — Decision Tree, Bagging, Random Forest,
AdaBoost, Gradient Boosting, XGBoost — with every parameter combination each grid search
tried logged to MLflow (44 total runs).

## Pipeline overview

| Stage | What happens |
|---|---|
| **Data Registration** | Validates the dataset schema against the business data dictionary, prints a summary |
| **Data Preparation** | Cleans the data, creates a stratified train/test split, uploads it as a workflow artifact |
| **Model Building & Tracking** | Downloads the artifact, tunes 6 models, logs every run to MLflow, commits the best model back to `main` |
| **Deployment** | Streamlit Community Cloud auto-redeploys whenever `main` updates — including every automated model refresh |

## Run it locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r tourism_project/requirements.txt

python tourism_project/model_building/data_register.py
python tourism_project/model_building/prep.py
python tourism_project/model_building/train.py

streamlit run tourism_project/deployment/app.py
```

Or open and run `wellness_tourism_full_project.ipynb` top to bottom in Colab.

## Deploying this yourself

1. Push to GitHub, confirm the **Actions** tab goes green.
2. Deploy on **Streamlit Community Cloud**: repo → branch `main` → main file path
   `tourism_project/deployment/app.py` → Python 3.11.
