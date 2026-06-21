# Databricks notebook source
# MAGIC %md
# MAGIC # 03 — Training & Experiment Tracking
# MAGIC
# MAGIC **Project:** Customer Churn — Production ML with MLflow
# MAGIC
# MAGIC **Input:** `churn.features_train`, `churn.features_val`
# MAGIC
# MAGIC **Output:** 4 MLflow-tracked model runs + 1 comparison summary run
# MAGIC
# MAGIC ### Models
# MAGIC Logistic Regression, Random Forest, XGBoost, LightGBM — chosen to span the
# MAGIC spectrum from interpretable linear baseline to gradient-boosted ensembles,
# MAGIC which is exactly the range a churn team would compare before picking a
# MAGIC production candidate.
# MAGIC
# MAGIC ### Tracking strategy
# MAGIC We briefly demo `mlflow.autolog()` on a quick baseline — useful for fast
# MAGIC exploration — then **disable it** and switch to manual logging for the
# MAGIC systematic comparison. Manual logging gives us consistent metric names
# MAGIC across model families (autolog's logged metric names differ between
# MAGIC sklearn/XGBoost/LightGBM), custom artifacts (confusion matrix, ROC curve),
# MAGIC and full control over what gets registered later.
# MAGIC
# MAGIC ### Hyperparameter search
# MAGIC `RandomizedSearchCV`, 3-fold `StratifiedKFold`, 20 iterations, scored on
# MAGIC ROC-AUC. Stratified CV matters here — with a 26.5% churn rate, a random
# MAGIC (non-stratified) fold could end up with a meaningfully different positive
# MAGIC rate, making CV scores noisy.
# MAGIC
# MAGIC ### Metrics logged (per model, on validation set)
# MAGIC ROC-AUC, PR-AUC (average precision — more informative than ROC-AUC under
# MAGIC class imbalance), F1, precision, recall.

# COMMAND ----------

# MAGIC %md ## 0. Imports & Config

# COMMAND ----------

# MAGIC %pip install xgboost lightgbm --quiet
# MAGIC %pip install --upgrade scikit-learn=1.9.0 --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, confusion_matrix,
    ConfusionMatrixDisplay, roc_curve
)
from scipy.stats import randint, uniform, loguniform

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment("/Workspace/Users/deshpande.ajay.us@gmail.com/churn-ops/03_training_and_tracking")

DB_NAME = "churn"
RANDOM_STATE = 42

# COMMAND ----------

# MAGIC %md ## 1. Load Data

# COMMAND ----------

train_pd = spark.table(f"{DB_NAME}.features_train").toPandas()
val_pd   = spark.table(f"{DB_NAME}.features_val").toPandas()

print(f"Train: {train_pd.shape}  |  Val: {val_pd.shape}")

# COMMAND ----------

# MAGIC %md ## 2. Define Feature Groups & Preprocessing Pipeline
# MAGIC
# MAGIC One `ColumnTransformer` shared across all 4 models:
# MAGIC - **One-hot encode** multi-category columns (`handle_unknown="ignore"` so the
# MAGIC   pipeline doesn't break if val/test contains a category unseen in train)
# MAGIC - **Standard-scale** continuous numerics (matters for LR; harmless for tree
# MAGIC   models, so a single shared pipeline keeps the comparison apples-to-apples)
# MAGIC - **Pass through** binary 0/1 flags unchanged

# COMMAND ----------

categorical_for_onehot = [
    "InternetService", "MultipleLines", "PaymentMethod",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
    "tenure_bucket",
]

numeric_features = [
    "tenure", "MonthlyCharges", "TotalCharges",
    "contract_encoded", "num_addon_services",
    "avg_monthly_spend", "charge_increase_ratio",
]

binary_features = [
    "SeniorCitizen", "gender_flag", "Partner_flag",
    "Dependents_flag", "PhoneService_flag", "PaperlessBilling_flag",
]

target_col = "Churn_flag"

preprocessor = ColumnTransformer(
    transformers=[
        ("onehot", OneHotEncoder(handle_unknown="ignore"), categorical_for_onehot),
        ("scale", StandardScaler(), numeric_features),
        ("passthrough", "passthrough", binary_features),
    ]
)

# COMMAND ----------

X_train = train_pd[categorical_for_onehot + numeric_features + binary_features]
y_train = train_pd[target_col]

X_val = val_pd[categorical_for_onehot + numeric_features + binary_features]
y_val = val_pd[target_col]

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"scale_pos_weight (neg/pos ratio): {scale_pos_weight:.3f}")

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

# COMMAND ----------

# MAGIC %md ## 3. Quick Baseline with `mlflow.autolog()`
# MAGIC
# MAGIC A single default-hyperparameter Logistic Regression, autologged. This is
# MAGIC the "I just want to see if this works" experiment a data scientist runs
# MAGIC before committing to a full search.

# COMMAND ----------

mlflow.sklearn.autolog()

with mlflow.start_run(run_name="baseline_lr_autolog") as run:
    baseline_pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
    ])
    baseline_pipeline.fit(X_train, y_train)

    # autolog captures training params + sklearn's default .score() (accuracy);
    # we additionally log val ROC-AUC manually since that's our primary metric
    val_proba = baseline_pipeline.predict_proba(X_val)[:, 1]
    mlflow.log_metric("val_roc_auc", roc_auc_score(y_val, val_proba))

    print(f"Baseline LR val ROC-AUC: {roc_auc_score(y_val, val_proba):.4f}")
    print(f"Run ID: {run.info.run_id}")

mlflow.sklearn.autolog(disable=True)

# COMMAND ----------

# MAGIC %md ## 4. Helper Functions — Plots & Training Loop

# COMMAND ----------

def save_confusion_matrix(y_true, y_pred, model_name, path):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Churn", "Churn"])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"{model_name} — Confusion Matrix (Validation)")
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close(fig)


def save_roc_curve(y_true, y_proba, model_name, path):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{model_name} — ROC Curve (Validation)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close(fig)

# COMMAND ----------

def train_and_log_model(model_name, base_estimator, param_distributions, n_iter=20):
    """
    Runs RandomizedSearchCV over a Pipeline(preprocess + classifier), logs the
    best params, validation metrics, confusion matrix, ROC curve, and the
    fitted pipeline to MLflow. Returns (run_id, metrics_dict, fitted_pipeline).
    """
    with mlflow.start_run(run_name=f"{model_name}_search") as run:

        pipeline = Pipeline([
            ("preprocess", preprocessor),
            ("classifier", base_estimator),
        ])

        search = RandomizedSearchCV(
            pipeline,
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=cv,
            scoring="roc_auc",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=0,
        )
        search.fit(X_train, y_train)
        best_model = search.best_estimator_

        # ── Log best hyperparameters (strip "classifier__" prefix) ──────────
        best_params = {k.replace("classifier__", ""): v for k, v in search.best_params_.items()}
        mlflow.log_params(best_params)
        mlflow.log_param("n_search_iter", n_iter)
        mlflow.log_param("cv_folds", cv.n_splits)
        mlflow.log_metric("cv_best_roc_auc", search.best_score_)

        # ── Validation metrics ───────────────────────────────────────────────
        y_val_proba = best_model.predict_proba(X_val)[:, 1]
        y_val_pred = best_model.predict(X_val)

        metrics = {
            "val_roc_auc": roc_auc_score(y_val, y_val_proba),
            "val_pr_auc": average_precision_score(y_val, y_val_proba),
            "val_f1": f1_score(y_val, y_val_pred),
            "val_precision": precision_score(y_val, y_val_pred),
            "val_recall": recall_score(y_val, y_val_pred),
        }
        mlflow.log_metrics(metrics)

        # ── Artifacts: confusion matrix + ROC curve ──────────────────────────
        cm_path = f"/Workspace/Users/deshpande.ajay.us@gmail.com/churn-ops/assets/plots/eda/{model_name}_confusion_matrix.png"
        roc_path = f"/Workspace/Users/deshpande.ajay.us@gmail.com/churn-ops/assets/plots/eda/{model_name}_roc_curve.png"
        save_confusion_matrix(y_val, y_val_pred, model_name, cm_path)
        save_roc_curve(y_val, y_val_proba, model_name, roc_path)
        mlflow.log_artifact(cm_path, artifact_path="plots")
        mlflow.log_artifact(roc_path, artifact_path="plots")

        # ── Log model (sklearn flavor — pipeline wraps XGB/LGBM too) ─────────
        # Note: for a standalone XGBoost/LightGBM deployment, the native
        # mlflow.xgboost / mlflow.lightgbm flavors give richer metadata
        # (e.g. native booster format). Since these are wrapped in a shared
        # sklearn Pipeline for unified preprocessing, we use the sklearn
        # flavor consistently across all 4 models.
        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="model",
            input_example=X_train.head(5),
        )

        mlflow.set_tag("model_family", model_name)
        mlflow.set_tag("notebook", "03_training_and_tracking")

        print(f"{model_name:18s} | val_roc_auc={metrics['val_roc_auc']:.4f} "
              f"| val_pr_auc={metrics['val_pr_auc']:.4f} "
              f"| val_f1={metrics['val_f1']:.4f}")

        return run.info.run_id, metrics, best_model

# COMMAND ----------

# MAGIC %md ## 5. Train All 4 Models

# COMMAND ----------

# MAGIC %md ### 5.1 Logistic Regression

# COMMAND ----------

param_dist_lr = {
    "classifier__C": loguniform(1e-3, 1e2),
    "classifier__penalty": ["l1", "l2"],
    "classifier__class_weight": [None, "balanced"],
}

lr_estimator = LogisticRegression(solver="liblinear", max_iter=1000, random_state=RANDOM_STATE)

lr_run_id, lr_metrics, lr_model = train_and_log_model(
    "logistic_regression", lr_estimator, param_dist_lr, n_iter=20
)

# COMMAND ----------

# MAGIC %md ### 5.2 Random Forest

# COMMAND ----------

param_dist_rf = {
    "classifier__n_estimators": randint(100, 500),
    "classifier__max_depth": randint(3, 20),
    "classifier__min_samples_split": randint(2, 20),
    "classifier__min_samples_leaf": randint(1, 10),
    "classifier__max_features": ["sqrt", "log2", None],
    "classifier__class_weight": [None, "balanced"],
}

rf_estimator = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)

rf_run_id, rf_metrics, rf_model = train_and_log_model(
    "random_forest", rf_estimator, param_dist_rf, n_iter=20
)

# COMMAND ----------

# MAGIC %md ### 5.3 XGBoost

# COMMAND ----------

param_dist_xgb = {
    "classifier__n_estimators": randint(100, 500),
    "classifier__max_depth": randint(3, 10),
    "classifier__learning_rate": loguniform(0.01, 0.3),
    "classifier__subsample": uniform(0.6, 0.4),
    "classifier__colsample_bytree": uniform(0.6, 0.4),
    "classifier__scale_pos_weight": [1, scale_pos_weight],
}

xgb_estimator = XGBClassifier(
    random_state=RANDOM_STATE, eval_metric="logloss", use_label_encoder=False
)

xgb_run_id, xgb_metrics, xgb_model = train_and_log_model(
    "xgboost", xgb_estimator, param_dist_xgb, n_iter=20
)

# COMMAND ----------

# MAGIC %md ### 5.4 LightGBM

# COMMAND ----------

param_dist_lgbm = {
    "classifier__n_estimators": randint(100, 500),
    "classifier__max_depth": randint(3, 12),
    "classifier__learning_rate": loguniform(0.01, 0.3),
    "classifier__num_leaves": randint(20, 150),
    "classifier__subsample": uniform(0.6, 0.4),
    "classifier__colsample_bytree": uniform(0.6, 0.4),
    "classifier__class_weight": [None, "balanced"],
}

lgbm_estimator = LGBMClassifier(random_state=RANDOM_STATE, verbose=-1)

lgbm_run_id, lgbm_metrics, lgbm_model = train_and_log_model(
    "lightgbm", lgbm_estimator, param_dist_lgbm, n_iter=20
)

# COMMAND ----------

# MAGIC %md ## 6. Model Comparison Summary

# COMMAND ----------

results = {
    "logistic_regression": (lr_run_id, lr_metrics),
    "random_forest": (rf_run_id, rf_metrics),
    "xgboost": (xgb_run_id, xgb_metrics),
    "lightgbm": (lgbm_run_id, lgbm_metrics),
}

comparison_df = pd.DataFrame({
    model: metrics for model, (_, metrics) in results.items()
}).T
comparison_df.index.name = "model"
comparison_df = comparison_df.sort_values("val_roc_auc", ascending=False)

print(comparison_df.round(4))

best_model_name = comparison_df.index[0]
print(f"\nBest model by val_roc_auc: {best_model_name} "
      f"({comparison_df.loc[best_model_name, 'val_roc_auc']:.4f})")

# COMMAND ----------

# Bar chart of val ROC-AUC across models
fig, ax = plt.subplots(figsize=(7, 4))
comparison_df["val_roc_auc"].plot(kind="bar", ax=ax, color="steelblue")
ax.set_ylabel("Validation ROC-AUC")
ax.set_title("Model Comparison — Validation ROC-AUC")
ax.set_ylim(0.5, 1.0)
for i, v in enumerate(comparison_df["val_roc_auc"]):
    ax.text(i, v + 0.01, f"{v:.3f}", ha="center")
plt.tight_layout()
plt.savefig("/Workspace/Users/deshpande.ajay.us@gmail.com/churn-ops/assets/plots/eda/model_comparison_roc_auc.png", dpi=100)
plt.show()

# COMMAND ----------

with mlflow.start_run(run_name="model_comparison_summary") as run:

    for model_name, (run_id, metrics) in results.items():
        for metric_name, value in metrics.items():
            mlflow.log_metric(f"{model_name}__{metric_name}", value)
        mlflow.set_tag(f"{model_name}__run_id", run_id)

    mlflow.set_tag("best_model", best_model_name)
    mlflow.set_tag("notebook", "03_training_and_tracking")

    comparison_df.to_csv("/tmp/model_comparison.csv")
    mlflow.log_artifact("/tmp/model_comparison.csv", artifact_path="comparison")
    mlflow.log_artifact("/Workspace/Users/deshpande.ajay.us@gmail.com/churn-ops/assets/plots/eda/logistic_regression_roc_curve.png", artifact_path="comparison")

    print(f"\n✓ Summary run logged: {run.info.run_id}")
    print(f"✓ Best model run_id (for notebook 04): {results[best_model_name][0]}")

# COMMAND ----------

# MAGIC %md
# MAGIC **Next:** `04_model_registry.py` — register `{best_model_name}`'s run as a
# MAGIC model version, validate against the held-out test set, and promote
# MAGIC Staging → Production.
