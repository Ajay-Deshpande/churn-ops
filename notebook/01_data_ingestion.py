# Databricks notebook source
# MAGIC %md
# MAGIC # 01 — Data Ingestion
# MAGIC
# MAGIC **Project:** Customer Churn — Production ML with MLflow
# MAGIC **Purpose:** Download the dataset, validate schema/quality, write to Delta, log ingestion metadata to MLflow.
# MAGIC
# MAGIC ### Dataset
# MAGIC | Dataset | Source | Role |
# MAGIC |---|---|---|
# MAGIC | IBM Telco Churn | Kaggle `blastchar/telco-customer-churn` | Training, evaluation, and drift-simulation base |
# MAGIC
# MAGIC ### Design note: single dataset, synthetic drift
# MAGIC We initially scoped a second real dataset (Cell2Cell, a different telecom carrier) for
# MAGIC drift detection. Two issues ruled it out:
# MAGIC 1. **Schema incompatibility** — only ~7 features map cleanly across both datasets,
# MAGIC    dropping the strongest churn predictors (`Contract`, `InternetService`, `TechSupport`,
# MAGIC    `PaymentMethod`) and degrading expected AUC from ~0.84 to ~0.70.
# MAGIC 2. **Unlabeled holdout** — Cell2Cell's holdout split is 100% `Churn = NA`, unusable for
# MAGIC    retrain validation.
# MAGIC
# MAGIC Instead, we use **synthetic drift injection** on the Telco dataset in notebook 06 —
# MAGIC the standard pattern used by Evidently AI and Databricks MLOps reference
# MAGIC implementations to validate that monitoring and retrain-trigger logic actually fires.
# MAGIC This keeps the full, high-signal feature set for modeling and SHAP.
# MAGIC
# MAGIC ### Outputs
# MAGIC - Delta table: `churn.telco_raw`
# MAGIC - MLflow run: ingestion metadata (row count, null rates, schema, target distribution)

# COMMAND ----------

# MAGIC %md ## 0. Imports & Config

# COMMAND ----------

# MAGIC %pip install kagglehub --quiet
# MAGIC %pip install --upgrade scikit-learn=1.9.0 --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import kagglehub
import mlflow
import pandas as pd
import os
from pyspark.sql import functions as F

# ── MLflow setup ──────────────────────────────────────────────────────────────
# Workspace Model Registry (free-tier Databricks; enterprise equivalent uses
# mlflow.set_registry_uri("databricks-uc") with Unity Catalog)
mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment("/Workspace/Users/deshpande.ajay.us@gmail.com/churn-ops/01_data_ingestion")

# ── Delta database ────────────────────────────────────────────────────────────
DB_NAME = "churn"
spark.sql(f"CREATE DATABASE IF NOT EXISTS {DB_NAME}")
print(f"Database '{DB_NAME}' ready.")

# COMMAND ----------

# MAGIC %md ## 1. Download Dataset via KaggleHub

# COMMAND ----------

telco_path = kagglehub.dataset_download("blastchar/telco-customer-churn", output_dir='/Workspace/Users/deshpande.ajay.us@gmail.com/churn-ops/downloads')
print("Telco path:", telco_path)

for f in os.listdir(telco_path):
    print(" ", f)

# COMMAND ----------

# MAGIC %md ## 2. Load Raw CSV into Spark DataFrame

# COMMAND ----------

telco_csv = os.path.join(telco_path, "WA_Fn-UseC_-Telco-Customer-Churn.csv")

telco_raw = (
    spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .csv(telco_csv)
)

print(f"Telco rows: {telco_raw.count():,}  |  cols: {len(telco_raw.columns)}")
telco_raw.printSchema()

# COMMAND ----------

# MAGIC %md ## 3. Schema & Quality Validation
# MAGIC
# MAGIC - **Row count** — catch empty or truncated downloads
# MAGIC - **Null rate per column** — flag anything > 20% missing
# MAGIC - **Target distribution** — churn rate; severe imbalance affects model choice
# MAGIC   (Telco churn rate ~26.5% — moderate imbalance, PR-AUC will matter alongside ROC-AUC)

# COMMAND ----------

def profile_dataframe(df, name, target_col):
    """
    Returns a summary dict with row count, null rates, and target distribution.
    Prints a readable report.
    """
    n_rows = df.count()
    n_cols = len(df.columns)

    null_counts = df.select(
        [F.round(F.mean(F.col(c).isNull().cast("int")) * 100, 2).alias(c)
         for c in df.columns]
    ).collect()[0].asDict()

    high_null_cols = {k: v for k, v in null_counts.items() if v > 20}

    if target_col in df.columns:
        target_dist = (
            df.groupBy(target_col)
            .count()
            .withColumn("pct", F.round(F.col("count") / n_rows * 100, 2))
            .toPandas()
            .set_index(target_col)
            .to_dict()
        )
    else:
        target_dist = {"error": f"Column '{target_col}' not found"}

    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")
    print(f"  Rows : {n_rows:,}")
    print(f"  Cols : {n_cols}")
    print(f"  High-null columns (>20%): {high_null_cols if high_null_cols else 'None'}")
    print(f"  Target distribution:\n{pd.DataFrame(target_dist)}\n")

    return {
        "dataset": name,
        "n_rows": n_rows,
        "n_cols": n_cols,
        "high_null_cols": list(high_null_cols.keys()),
        "null_rates": null_counts,
        "target_distribution": target_dist,
    }

# COMMAND ----------

telco_profile = profile_dataframe(telco_raw, "IBM Telco", "Churn")

# COMMAND ----------

# MAGIC %md ## 4. Light Standardisation Before Writing to Delta
# MAGIC
# MAGIC No feature engineering yet (that's notebook 02). We only:
# MAGIC - Rename `customerID` → `customer_id`
# MAGIC - Cast `TotalCharges` to double (ships as string due to 11 blank values for
# MAGIC   customers with `tenure = 0`)
# MAGIC - Add a `source` and `ingested_at` column for lineage

# COMMAND ----------

telco_clean = (
    telco_raw
    .withColumnRenamed("customerID", "customer_id")
    .withColumn("TotalCharges",
                F.when(F.trim(F.col("TotalCharges")) == "", None)
                 .otherwise(F.col("TotalCharges").cast("double")))
    .withColumn("source", F.lit("telco"))
    .withColumn("ingested_at", F.current_timestamp())
)

# Confirm the TotalCharges cast didn't silently introduce unexpected nulls
n_null_total_charges = telco_clean.filter(F.col("TotalCharges").isNull()).count()
print(f"Rows with null TotalCharges after cast: {n_null_total_charges} "
      f"(expected: 11, all tenure==0)")

# COMMAND ----------

# MAGIC %md ## 5. Write to Delta

# COMMAND ----------

(
    telco_clean
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(f"{DB_NAME}.telco_raw")
)
print("✓ Written: churn.telco_raw")

# COMMAND ----------

# MAGIC %md ## 6. Log Ingestion Run to MLflow
# MAGIC
# MAGIC Even ingestion gets an MLflow run — this is how data lineage is tracked in
# MAGIC production. You can always answer "which data version trained this model?"

# COMMAND ----------

with mlflow.start_run(run_name="data_ingestion_v1") as run:

    mlflow.log_param("source", "blastchar/telco-customer-churn")

    mlflow.log_metric("n_rows", telco_profile["n_rows"])
    mlflow.log_metric("n_cols", telco_profile["n_cols"])
    mlflow.log_metric("null_total_charges_post_cast", n_null_total_charges)

    dist = telco_profile["target_distribution"]["pct"]
    mlflow.log_metric("churn_rate_pct", float(dist.get("Yes", 0.0)))

    mlflow.set_tag("delta.telco_raw", f"{DB_NAME}.telco_raw")
    mlflow.set_tag("notebook", "01_data_ingestion")
    mlflow.set_tag("drift_strategy", "synthetic_injection_notebook_06")

    schema_df = pd.DataFrame({
        "column": telco_raw.columns,
        "dtype": [str(t) for _, t in telco_raw.dtypes],
    })
    schema_df.to_csv("/tmp/telco_schema.csv", index=False)
    mlflow.log_artifact("/tmp/telco_schema.csv", artifact_path="schemas")

    print(f"\n✓ MLflow run logged: {run.info.run_id}")

# COMMAND ----------

# MAGIC %md ## 7. Sanity Check — Read Back from Delta

# COMMAND ----------

print("=== churn.telco_raw ===")
display(spark.table("churn.telco_raw").select(
    "customer_id", "tenure", "Contract", "MonthlyCharges", "TotalCharges", "Churn", "source"
).take(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC | Table | Rows | Purpose |
# MAGIC |---|---|---|
# MAGIC | `churn.telco_raw` | 7,043 | Training, evaluation, SHAP, and synthetic-drift base |
# MAGIC
# MAGIC **Next:** `02_feature_engineering.py` — tenure buckets, usage ratios, contract encoding,
# MAGIC train/val/test split.
