# Databricks notebook source
# MAGIC %md
# MAGIC # 07 — Batch Scoring
# MAGIC
# MAGIC **Project:** Customer Churn — Production ML with MLflow
# MAGIC **Input:** Current champion model pointer, a batch of customers to score
# MAGIC **Output:** `churn.predictions_batch` Delta table with probabilities and risk tiers
# MAGIC
# MAGIC ### Role in the pipeline
# MAGIC This is the notebook that runs **on a schedule** in production — daily
# MAGIC or weekly, scoring whatever new/existing customer batch needs fresh
# MAGIC churn predictions. Notebooks 01-06 are the "build and maintain the
# MAGIC model" lifecycle; this is the "use the model" step that actually
# MAGIC delivers business value.
# MAGIC
# MAGIC ### Design choices
# MAGIC - **Always reads the champion pointer fresh** — never hardcodes a
# MAGIC   run_id, so this notebook automatically picks up whichever model
# MAGIC   (v1 or v2, post-drift-retrain) is currently champion. No code change
# MAGIC   needed when notebook 06 promotes a new champion.
# MAGIC - **Idempotent writes**: each run appends a batch tagged with a
# MAGIC   `scored_at` timestamp and `model_run_id` — you can always trace which
# MAGIC   model version produced which predictions, and re-running for the same
# MAGIC   day is safe (creates a new batch, doesn't silently overwrite).
# MAGIC - **Risk tiers**: raw probabilities are useful for ranking, but a
# MAGIC   retention team works off discrete tiers (who gets a call today vs.
# MAGIC   an email vs. nothing) — so we bucket into High/Medium/Low.

# COMMAND ----------

# MAGIC %md ## 0. Imports & Config

# COMMAND ----------

# MAGIC %pip install --upgrade scikit-learn=1.9.0 lightgbm --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import mlflow
import mlflow.sklearn
import pandas as pd
import json
from pyspark.sql import functions as F
from datetime import datetime

mlflow.set_registry_uri('uc-databricks')
mlflow.set_experiment("/Workspace/Users/deshpande.ajay.us@gmail.com/churn-ops/07_batch_scoring")

DB_NAME = "churn"
CHAMPION_POINTER_PATH = "/Workspace/Users/deshpande.ajay.us@gmail.com/churn-ops/champion_model.json"

# Risk tier thresholds — tunable based on retention team capacity
HIGH_RISK_THRESHOLD = 0.60
MEDIUM_RISK_THRESHOLD = 0.30

# COMMAND ----------

# MAGIC %md ## 1. Load Current Champion
# MAGIC
# MAGIC Always read fresh — this is what makes the notebook safe to schedule
# MAGIC without manual intervention after a retrain promotes a new champion.

# COMMAND ----------

with open(CHAMPION_POINTER_PATH) as f:
    champion_info = json.load(f)

champion_model = mlflow.sklearn.load_model(champion_info["model_uri"])

print(f"Scoring with champion model:")
print(f"  model_family : {champion_info['model_family']}")
print(f"  run_id       : {champion_info['run_id']}")
print(f"  test_roc_auc : {champion_info['test_roc_auc']:.4f}")
if champion_info.get("retrained_due_to_drift"):
    print(f"  (retrained due to drift on {champion_info.get('previous_champion_run_id', 'N/A')[:8]}...)")

# COMMAND ----------

# MAGIC %md ## 2. Load Batch to Score
# MAGIC
# MAGIC In production this would be a fresh extract of active customers from
# MAGIC an operational system, landed in a Delta table by an upstream
# MAGIC ingestion job. For this portfolio demo, we score the held-out test
# MAGIC set — customers the model has been validated against but whose
# MAGIC predictions haven't yet been written to a "production" output table.

# COMMAND ----------

batch_pd = spark.table(f"{DB_NAME}.features_test").toPandas()
print(f"Batch to score: {len(batch_pd):,} customers")

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

X_batch = batch_pd[feature_cols]

# COMMAND ----------

# MAGIC %md ## 3. Score the Batch

# COMMAND ----------

churn_proba = champion_model.predict_proba(X_batch)[:, 1]
churn_pred = champion_model.predict(X_batch)

print(f"Scored {len(churn_proba):,} customers")
print(f"Mean predicted churn probability: {churn_proba.mean():.4f}")
print(f"Predicted churners (class=1): {churn_pred.sum():,} ({churn_pred.mean():.2%})")

# COMMAND ----------

# MAGIC %md ## 4. Assign Risk Tiers

# COMMAND ----------

def assign_risk_tier(proba):
    if proba >= HIGH_RISK_THRESHOLD:
        return "High"
    elif proba >= MEDIUM_RISK_THRESHOLD:
        return "Medium"
    return "Low"

risk_tiers = [assign_risk_tier(p) for p in churn_proba]

tier_counts = pd.Series(risk_tiers).value_counts()
print("Risk tier distribution:")
print(tier_counts)
print(f"\nTier thresholds — High: >={HIGH_RISK_THRESHOLD}, Medium: >={MEDIUM_RISK_THRESHOLD}, Low: <{MEDIUM_RISK_THRESHOLD}")

# COMMAND ----------

# MAGIC %md ## 5. Assemble Output Table

# COMMAND ----------

scored_at = datetime.utcnow()

predictions_pd = pd.DataFrame({
    "customer_id": batch_pd["customer_id"],
    "churn_probability": churn_proba,
    "predicted_churn": churn_pred,
    "risk_tier": risk_tiers,
    "model_run_id": champion_info["run_id"],
    "model_family": champion_info["model_family"],
    "scored_at": scored_at,
})

predictions_pd.head(10)

# COMMAND ----------

# MAGIC %md ## 6. Write to Delta
# MAGIC
# MAGIC Append mode — each scoring run adds a new batch, never overwrites
# MAGIC history. `model_run_id` and `scored_at` let you reconstruct exactly
# MAGIC which model produced which predictions on which date, which matters
# MAGIC both for auditability and for measuring realized vs. predicted churn
# MAGIC over time once outcomes are known.

# COMMAND ----------

predictions_sdf = spark.createDataFrame(predictions_pd)

(
    predictions_sdf.write
    .format("delta")
    .mode("append")
    .option("mergeSchema", "true")
    .saveAsTable(f"{DB_NAME}.predictions_batch")
)

print(f"✓ Appended {len(predictions_pd):,} predictions to {DB_NAME}.predictions_batch")

# COMMAND ----------

# MAGIC %md ## 7. Sanity Check — Read Back

# COMMAND ----------

display(spark.table(f"{DB_NAME}.predictions_batch") \
    .orderBy(F.desc("churn_probability")) \
    .take(10))

# COMMAND ----------

# MAGIC %md ## 8. Log Scoring Run to MLflow

# COMMAND ----------

with mlflow.start_run(run_name="batch_scoring") as run:
    mlflow.log_param("model_run_id", champion_info["run_id"])
    mlflow.log_param("model_family", champion_info["model_family"])
    mlflow.log_param("scored_at", scored_at.isoformat())
    mlflow.log_param("high_risk_threshold", HIGH_RISK_THRESHOLD)
    mlflow.log_param("medium_risk_threshold", MEDIUM_RISK_THRESHOLD)

    mlflow.log_metric("n_customers_scored", len(predictions_pd))
    mlflow.log_metric("mean_churn_probability", churn_proba.mean())
    mlflow.log_metric("n_high_risk", int((pd.Series(risk_tiers) == "High").sum()))
    mlflow.log_metric("n_medium_risk", int((pd.Series(risk_tiers) == "Medium").sum()))
    mlflow.log_metric("n_low_risk", int((pd.Series(risk_tiers) == "Low").sum()))

    mlflow.set_tag("notebook", "07_batch_scoring")
    mlflow.set_tag("champion_run_id", champion_info["run_id"])

    print(f"✓ Scoring run logged: {run.info.run_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC | Metric | Value |
# MAGIC |---|---|
# MAGIC | Customers scored | *(see output above)* |
# MAGIC | High-risk count | *(see tier distribution above)* |
# MAGIC | Model used | *(model_family, run_id from champion pointer)* |
# MAGIC
# MAGIC **This notebook is the scheduled job task** — see `job_config.json` in
# MAGIC the repo root for the Databricks job definition that chains notebooks
# MAGIC 01-07 and runs this scoring step on a recurring schedule.
