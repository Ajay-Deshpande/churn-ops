# Databricks notebook source
# MAGIC %md
# MAGIC # 04 — Model Registry: Register, Validate, Promote
# MAGIC
# MAGIC **Project:** Customer Churn — Production ML with MLflow
# MAGIC **Input:** Best run from `03_training_and_tracking` experiment, `churn.features_test`
# MAGIC **Output:** Validated "champion" model referenced by a stable `runs:/` URI
# MAGIC
# MAGIC **Workaround used in this notebook:** rather than a registered model
# MAGIC version, we treat the *validated run's artifact URI*
# MAGIC (`runs:/<run_id>/model`) as the stable production reference. The
# MAGIC validation gate, promotion logic, and tagging conventions are identical
# MAGIC to what a registered-model workflow would do — only the storage/lookup
# MAGIC mechanism differs. Downstream notebooks (05, 07) load the model via this
# MAGIC run URI instead of `models:/name@alias`.
# MAGIC
# MAGIC In a workspace with a properly provisioned UC metastore, swapping back
# MAGIC to true registration is a ~5 line change (see commented block in
# MAGIC Section 3) — no changes needed elsewhere.
# MAGIC
# MAGIC ### Why query MLflow instead of reusing notebook 03's in-memory results?
# MAGIC This notebook is **decoupled** from notebook 03 — it finds the best run by
# MAGIC querying the experiment directly. This is how a real CI/CD pipeline works:
# MAGIC the "promote" stage runs independently and shouldn't depend on Python
# MAGIC variables from a training notebook's session.
# MAGIC
# MAGIC ### Validation gate
# MAGIC Before promoting challenger → champion, we score the **held-out test
# MAGIC set** (never touched until now) and require `test_roc_auc >= 0.80`. A
# MAGIC model only ships if it passes an objective threshold on data it has
# MAGIC never influenced.

# COMMAND ----------

# MAGIC %md ## 0. Imports & Config

# COMMAND ----------

# MAGIC %pip install lightgbm --quiet
# MAGIC %pip install --upgrade scikit-learn=1.9.0 --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import mlflow
import mlflow.sklearn
import pandas as pd
import json
import matplotlib.pyplot as plt
from mlflow.tracking import MlflowClient
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
)

mlflow.set_registry_uri("databricks-uc")
client = MlflowClient(registry_uri="databricks-uc")

TRAINING_EXPERIMENT = "/Workspace/Users/deshpande.ajay.us@gmail.com/churn-ops/03_training_and_tracking"
REGISTRY_EXPERIMENT = "/Workspace/Users/deshpande.ajay.us@gmail.com/churn-ops/04_model_registry"
MODEL_NAME = "churn_classifier"  # logical name only — not a registered-model name here
TEST_ROC_AUC_THRESHOLD = 0.80

mlflow.set_experiment(REGISTRY_EXPERIMENT)

DB_NAME = "churn"

# COMMAND ----------

# MAGIC %md ## 1. Find the Best Run from Notebook 03

# COMMAND ----------

runs_df = mlflow.search_runs(
    experiment_names=[TRAINING_EXPERIMENT],
    filter_string="tags.notebook = '03_training_and_tracking'",
    order_by=["metrics.val_roc_auc DESC"],
    max_results=4,
)

display(runs_df[["run_id", "tags.model_family", "metrics.val_roc_auc", "metrics.val_pr_auc"]])

best_run = runs_df.iloc[0]
best_run_id = best_run["run_id"]
best_model_family = best_run["tags.model_family"]
best_val_roc_auc = best_run["metrics.val_roc_auc"]

print(f"\nBest model: {best_model_family}  |  run_id: {best_run_id}  |  val_roc_auc: {best_val_roc_auc:.4f}")

# COMMAND ----------

# MAGIC %md ## 2. Mark as "Challenger"
# MAGIC
# MAGIC In a true registry workflow, this step would call
# MAGIC `mlflow.register_model()` and set a `@challenger` alias. Here we record
# MAGIC the same intent as run tags directly on the source run — the run_id
# MAGIC itself *is* the artifact reference, so tagging it is the equivalent
# MAGIC operation.

# COMMAND ----------

client.set_tag(best_run_id, "registry_status", "challenger")
client.set_tag(best_run_id, "model_logical_name", MODEL_NAME)
print(f"✓ Run {best_run_id} tagged as challenger for '{MODEL_NAME}'")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Reference: true UC registration code (untested — blocked by free-tier storage)
# MAGIC ```python
# MAGIC mlflow.set_registry_uri("databricks-uc")
# MAGIC uc_client = MlflowClient(registry_uri="databricks-uc")
# MAGIC CATALOG, SCHEMA = "workspace", "churn"
# MAGIC full_name = f"{CATALOG}.{SCHEMA}.{MODEL_NAME}"
# MAGIC
# MAGIC spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
# MAGIC uc_client.create_registered_model(full_name, description="...")
# MAGIC registration = mlflow.register_model(f"runs:/{best_run_id}/model", full_name)
# MAGIC uc_client.set_registered_model_alias(full_name, "challenger", registration.version)
# MAGIC # ... then later, after the gate: uc_client.set_registered_model_alias(full_name, "champion", registration.version)
# MAGIC ```

# COMMAND ----------

# MAGIC %md ## 3. Validation Gate — Score the Held-Out Test Set
# MAGIC
# MAGIC This is the **first and only time** the test set is used. We load the
# MAGIC model directly from its run artifact via the `sklearn` flavor (not
# MAGIC `pyfunc`) so we have access to `predict_proba` for ROC-AUC / PR-AUC.

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
feature_cols = categorical_for_onehot + numeric_features + binary_features

test_pd = spark.table(f"{DB_NAME}.features_test").toPandas()
X_test = test_pd[feature_cols]
y_test = test_pd["Churn_flag"]

print(f"Test set: {len(test_pd):,} rows  |  churn rate: {y_test.mean():.4f}")

# COMMAND ----------

challenger_model_uri = f"runs:/{best_run_id}/model"
challenger_model = mlflow.sklearn.load_model(challenger_model_uri)

y_test_proba = challenger_model.predict_proba(X_test)[:, 1]
y_test_pred = challenger_model.predict(X_test)

test_metrics = {
    "test_roc_auc": roc_auc_score(y_test, y_test_proba),
    "test_pr_auc": average_precision_score(y_test, y_test_proba),
    "test_f1": f1_score(y_test, y_test_pred),
    "test_precision": precision_score(y_test, y_test_pred),
    "test_recall": recall_score(y_test, y_test_pred),
}

for k, v in test_metrics.items():
    print(f"  {k}: {v:.4f}")

gate_passed = test_metrics["test_roc_auc"] >= TEST_ROC_AUC_THRESHOLD
print(f"\nValidation gate (test_roc_auc >= {TEST_ROC_AUC_THRESHOLD}): "
      f"{'PASS' if gate_passed else 'FAIL'}")

# COMMAND ----------

cm = confusion_matrix(y_test, y_test_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Churn", "Churn"])
fig, ax = plt.subplots(figsize=(5, 4))
disp.plot(ax=ax, cmap="Greens", colorbar=False)
ax.set_title(f"{best_model_family} (challenger) — Confusion Matrix (TEST)")
plt.tight_layout()
plt.savefig("/Workspace/Users/deshpande.ajay.us@gmail.com/churn-ops/assets/plots/eda/test_confusion_matrix.png", dpi=100)
plt.show()

# COMMAND ----------

# MAGIC %md ## 4. Log Validation Run

# COMMAND ----------

with mlflow.start_run(run_name=f"validate_{MODEL_NAME}_{best_model_family}") as run:
    mlflow.log_param("model_logical_name", MODEL_NAME)
    mlflow.log_param("model_family", best_model_family)
    mlflow.log_param("source_run_id", best_run_id)
    mlflow.log_param("threshold_test_roc_auc", TEST_ROC_AUC_THRESHOLD)

    mlflow.log_metrics(test_metrics)
    mlflow.log_metric("gate_passed", int(gate_passed))

    mlflow.log_artifact("/Workspace/Users/deshpande.ajay.us@gmail.com/churn-ops/assets/plots/eda/test_confusion_matrix.png", artifact_path="plots")

    mlflow.set_tag("notebook", "04_model_registry")
    mlflow.set_tag("source_run_id", best_run_id)
    mlflow.set_tag("gate_result", "PASS" if gate_passed else "FAIL")

    validation_run_id = run.info.run_id
    print(f"✓ Validation run logged: {validation_run_id}")

# COMMAND ----------

# MAGIC %md ## 5. Promote to "Champion" (if gate passed)
# MAGIC
# MAGIC Promotion = tagging the source run as champion and persisting its
# MAGIC run_id to a small JSON pointer file in DBFS. Downstream notebooks
# MAGIC (05 SHAP, 07 batch scoring) read this pointer rather than hardcoding
# MAGIC a run_id — the same indirection a `models:/name@champion` URI would
# MAGIC give you, just implemented without the registry.

# COMMAND ----------

CHAMPION_POINTER_PATH = "/Workspace/Users/deshpande.ajay.us@gmail.com/churn-ops/champion_model.json"

if gate_passed:
    client.set_tag(best_run_id, "registry_status", "champion")
    client.set_tag(best_run_id, "test_roc_auc", f"{test_metrics['test_roc_auc']:.4f}")
    client.set_tag(best_run_id, "validation_run_id", validation_run_id)

    champion_pointer = {
        "model_logical_name": MODEL_NAME,
        "run_id": best_run_id,
        "model_uri": f"runs:/{best_run_id}/model",
        "model_family": best_model_family,
        "test_roc_auc": test_metrics["test_roc_auc"],
        "val_roc_auc": best_val_roc_auc,
        "validation_run_id": validation_run_id,
    }
    
    import os
    
    dbutils.fs.mkdirs(os.path.dirname(CHAMPION_POINTER_PATH))
    
    with open(CHAMPION_POINTER_PATH, "w") as f:
        json.dump(champion_pointer, f, indent=2)

    print(f"✓ Run {best_run_id} → champion (test_roc_auc={test_metrics['test_roc_auc']:.4f})")
    print(f"✓ Champion pointer written: {CHAMPION_POINTER_PATH}")
    print(json.dumps(champion_pointer, indent=2))
else:
    client.set_tag(best_run_id, "registry_status", "failed_validation")
    print(f"✗ Run {best_run_id} failed validation gate — no champion pointer written.")

# COMMAND ----------

# MAGIC %md ## 6. Registry State Summary

# COMMAND ----------

tagged_runs = mlflow.search_runs(
    experiment_names=[TRAINING_EXPERIMENT],
    filter_string="tags.registry_status != ''",
    order_by=["start_time DESC"],
)

if len(tagged_runs):
    summary_cols = ["run_id", "tags.model_family", "tags.registry_status",
                     "tags.val_roc_auc" if "tags.val_roc_auc" in tagged_runs.columns else "metrics.val_roc_auc",
                     "tags.test_roc_auc"]
    summary_cols = [c for c in summary_cols if c in tagged_runs.columns]
    display(tagged_runs[summary_cols])
else:
    print("No tagged runs found yet.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC | Step | Result |
# MAGIC |---|---|
# MAGIC | Best candidate (by val ROC-AUC) | *(model_family, run_id — see output above)* |
# MAGIC | Test ROC-AUC | *(see output above)* |
# MAGIC | Validation gate | PASS / FAIL |
# MAGIC | Champion reference | `champion_model.json` pointer → `runs:/<run_id>/model` |
# MAGIC
# MAGIC **Next:** `05_shap_explainability.py` — read `champion_model.json`, load
# MAGIC the model via its `runs:/` URI, and generate global (SHAP beeswarm) and
# MAGIC local (waterfall) explanations.
