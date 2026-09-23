"""Model Building with Experiment Tracking: tune 6 models, log to MLflow, save the best."""
import json, joblib, pandas as pd, mlflow
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (BaggingClassifier, RandomForestClassifier,
                               AdaBoostClassifier, GradientBoostingClassifier)
from xgboost import XGBClassifier

RANDOM_STATE = 42
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("wellness-tourism-prodtaken")

def build_preprocessor(X):
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X.select_dtypes(exclude=["object"]).columns.tolist()
    return ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])

def get_model_grid():
    return {
        "DecisionTree": (DecisionTreeClassifier(random_state=RANDOM_STATE),
            {"model__max_depth": [4, 6, 8, None], "model__min_samples_split": [2, 5, 10]}),
        "Bagging": (BaggingClassifier(random_state=RANDOM_STATE),
            {"model__n_estimators": [50, 100], "model__max_samples": [0.7, 1.0]}),
        "RandomForest": (RandomForestClassifier(random_state=RANDOM_STATE),
            {"model__n_estimators": [100, 200], "model__max_depth": [6, 10, None]}),
        "AdaBoost": (AdaBoostClassifier(random_state=RANDOM_STATE),
            {"model__n_estimators": [50, 100], "model__learning_rate": [0.5, 1.0]}),
        "GradientBoosting": (GradientBoostingClassifier(random_state=RANDOM_STATE),
            {"model__n_estimators": [100, 200], "model__learning_rate": [0.05, 0.1]}),
        "XGBoost": (XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss"),
            {"model__n_estimators": [100, 200], "model__max_depth": [3, 5],
             "model__learning_rate": [0.05, 0.1]}),
    }

def evaluate(y_true, y_pred, y_proba):
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred), 4),
        "recall": round(recall_score(y_true, y_pred), 4),
        "f1": round(f1_score(y_true, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_true, y_proba), 4),
    }

def main():
    Xtrain = pd.read_csv("Xtrain.csv"); Xtest = pd.read_csv("Xtest.csv")
    ytrain = pd.read_csv("ytrain.csv").squeeze()
    ytest = pd.read_csv("ytest.csv").squeeze()
    preprocessor = build_preprocessor(Xtrain)

    best = {"name": None, "f1": -1, "pipeline": None, "metrics": None, "params": None}
    all_results = []

    for name, (estimator, grid) in get_model_grid().items():
        pipe = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
        with mlflow.start_run(run_name=name):
            search = GridSearchCV(pipe, grid, cv=3, scoring="f1", n_jobs=-1)
            search.fit(Xtrain, ytrain)

            cv_results = search.cv_results_
            for combo_idx, combo_params in enumerate(cv_results["params"]):
                with mlflow.start_run(run_name=f"{name}_combo_{combo_idx}", nested=True):
                    mlflow.log_params(combo_params)
                    mlflow.log_metric("mean_cv_f1", cv_results["mean_test_score"][combo_idx])
                    mlflow.log_metric("std_cv_f1", cv_results["std_test_score"][combo_idx])
                    mlflow.log_metric("rank_cv_f1", int(cv_results["rank_test_score"][combo_idx]))

            bp = search.best_estimator_
            pred = bp.predict(Xtest)
            proba = bp.predict_proba(Xtest)[:, 1]
            metrics = evaluate(ytest, pred, proba)
            mlflow.log_params(search.best_params_)
            mlflow.log_metrics(metrics)
            mlflow.set_tag("model_family", name)
            mlflow.log_metric("n_param_combinations_tried", len(cv_results["params"]))
            print(f"[{name}] tried {len(cv_results['params'])} combinations -> "
                  f"best {search.best_params_} -> {metrics}")
            all_results.append({"model": name, "best_params": search.best_params_, **metrics})
            if metrics["f1"] > best["f1"]:
                best = {"name": name, "f1": metrics["f1"], "pipeline": bp,
                        "metrics": metrics, "params": search.best_params_}

    print(f"BEST MODEL: {best['name']} (F1={best['f1']})")
    joblib.dump(best["pipeline"], "tourism_project/deployment/model.joblib")
    with open("tourism_project/deployment/metrics.json", "w") as f:
        json.dump({"best_model": best["name"], "best_params": best["params"],
                    "metrics": best["metrics"], "all_results": all_results},
                   f, indent=2, default=str)

if __name__ == "__main__":
    main()
