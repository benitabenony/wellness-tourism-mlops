# Wellness Tourism Package — Purchase Prediction (MLOps Pipeline)

Predicts, before a salesperson contacts a customer, whether that customer is likely to
purchase "Visit with Us"'s new **Wellness Tourism Package** — a fully automated GitHub
Actions MLOps pipeline with a Streamlit deployment.

## Everything is documented in one notebook

**`wellness_tourism_full_project.ipynb`** is the complete, executed, end-to-end project
report: business context, data registration, data cleaning, model tuning with MLflow
tracking (every parameter combination logged), model evaluation, the Streamlit
deployment code, and the GitHub Actions pipeline. Every `%%writefile` cell in the
notebook writes the actual pipeline file to disk, so running it top to bottom in Colab
produces a ready-to-push repository.

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

## Deploying this yourself

1. Push to GitHub, confirm the Actions tab goes green.
2. Deploy on Streamlit Community Cloud: repo → branch `main` → main file path
   `tourism_project/deployment/app.py`.
