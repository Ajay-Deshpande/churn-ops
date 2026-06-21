# Databricks notebook source
# MAGIC %md
# MAGIC # 02 — Feature Engineering
# MAGIC
# MAGIC **Project:** Customer Churn — Production ML with MLflow
# MAGIC **Input:** `churn.telco_raw`
# MAGIC **Output:** `churn.features_train`, `churn.features_val`, `churn.features_test`
# MAGIC
# MAGIC ### Approach
# MAGIC We build a **feature table** containing both raw categorical columns (for
# MAGIC one-hot encoding inside the model pipeline in notebook 03) and **engineered
# MAGIC features** that capture signal the raw columns don't expose directly.
# MAGIC Encoding choices (one-hot vs. ordinal vs. scaling) are deferred to a
# MAGIC `sklearn` `ColumnTransformer` in notebook 03 — this notebook's job is to
# MAGIC produce clean, well-defined columns, not model-ready arrays. This mirrors
# MAGIC the feature-table / training-pipeline split used in real feature stores.
# MAGIC
# MAGIC ### Engineered Features
# MAGIC | Feature | Logic | Why it matters |
# MAGIC |---|---|---|
# MAGIC | `tenure_bucket` | `0-12 / 13-24 / 25-48 / 49+` months | Churn risk is highly non-linear in tenure — new customers churn far more than long-tenured ones; binning lets LR capture this without needing splines |
# MAGIC | `contract_encoded` | Month-to-month=0, One year=1, Two year=2 | Ordinal — contract length is monotonically related to commitment/switching cost |
# MAGIC | `num_addon_services` | Count of "Yes" across 6 add-on service columns | A single "engagement depth" signal — customers with more add-ons are more entrenched |
# MAGIC | `avg_monthly_spend` | `TotalCharges / max(tenure, 1)` | Normalizes total spend by tenure — comparable across customers regardless of how long they've been a customer |
# MAGIC | `charge_increase_ratio` | `MonthlyCharges / avg_monthly_spend` | >1 = current bill is higher than historical average (recent price increase) — a known churn trigger |
# MAGIC
# MAGIC ### Split Strategy
# MAGIC 70/15/15 train/val/test, **stratified on `Churn`** to preserve the ~26.5% churn
# MAGIC rate across all three splits (critical for imbalanced classification — an
# MAGIC unstratified split could leave the test set with a meaningfully different
# MAGIC base rate, distorting metric comparisons).

# COMMAND ----------

# MAGIC %md ## 0. Imports & Config

# COMMAND ----------

# MAGIC %pip install --upgrade scikit-learn=1.9.0 --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import mlflow
import pandas as pd
import json
from pyspark.sql import functions as F
from sklearn.model_selection import train_test_split

mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment("/Workspace/Users/deshpande.ajay.us@gmail.com/churn-ops/02_feature_engineering")

DB_NAME = "churn"

# COMMAND ----------

# MAGIC %md ## 1. Load Raw Data

# COMMAND ----------

df = spark.table(f"{DB_NAME}.telco_raw")
print(f"Rows: {df.count():,}  |  Cols: {len(df.columns)}")

# COMMAND ----------

# MAGIC %md ## 2. Engineered Features

# COMMAND ----------

# MAGIC %md ### 2.1 `tenure_bucket`

# COMMAND ----------

df = df.withColumn(
    "tenure_bucket",
    F.when(F.col("tenure") <= 12, "0-12")
     .when(F.col("tenure") <= 24, "13-24")
     .when(F.col("tenure") <= 48, "25-48")
     .otherwise("49+")
)

df.groupBy("tenure_bucket").count().orderBy("tenure_bucket").show()

# COMMAND ----------

# MAGIC %md ### 2.2 `contract_encoded` (ordinal)

# COMMAND ----------

df = df.withColumn("TotalCharges", 
                F.when(F.col("TotalCharges").isNull(), 0.0
                    ).otherwise(F.col("TotalCharges")))

df = df.withColumn(
    "contract_encoded",
    F.when(F.col("Contract") == "Month-to-month", 0)
     .when(F.col("Contract") == "One year", 1)
     .when(F.col("Contract") == "Two year", 2)
     .otherwise(None)
)

df.groupBy("Contract", "contract_encoded").count().show()

# COMMAND ----------

# MAGIC %md ### 2.3 `num_addon_services`
# MAGIC
# MAGIC Counts "Yes" across the 6 add-on columns. Values of "No internet service" /
# MAGIC "No phone service" count as 0 (not "Yes"), so this naturally reflects both
# MAGIC "doesn't have internet/phone" and "has internet but declined the add-on."

# COMMAND ----------

addon_cols = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies"
]

addon_expr = sum(F.when(F.col(c) == "Yes", 1).otherwise(0) for c in addon_cols)
df = df.withColumn("num_addon_services", addon_expr)

df.groupBy("num_addon_services").count().orderBy("num_addon_services").show()

# COMMAND ----------

# MAGIC %md ### 2.4 `avg_monthly_spend` and `charge_increase_ratio`
# MAGIC
# MAGIC For the 11 customers with `tenure == 0` (and thus null `TotalCharges`),
# MAGIC we treat `avg_monthly_spend` as their current `MonthlyCharges` — they have
# MAGIC exactly one month of history, so the average *is* the current charge.
# MAGIC This makes `charge_increase_ratio == 1.0` for these customers, correctly
# MAGIC signaling "no change yet observed."

# COMMAND ----------

df = df.withColumn(
    "avg_monthly_spend",
    F.when(F.col("tenure") == 0, F.col("MonthlyCharges"))
     .otherwise(F.col("TotalCharges") / F.col("tenure"))
)

df = df.withColumn(
    "charge_increase_ratio",
    F.round(F.col("MonthlyCharges") / F.col("avg_monthly_spend"), 4)
)

df.select("tenure", "MonthlyCharges", "TotalCharges",
          "avg_monthly_spend", "charge_increase_ratio").show(5)

# Sanity: ratio should be 1.0 for all tenure==0 rows
df.filter(F.col("tenure") == 0).select("charge_increase_ratio").distinct().show()

# COMMAND ----------

# MAGIC %md ### 2.5 Binary Yes/No → 0/1 Encoding
# MAGIC
# MAGIC Simple binary columns get encoded now (no ambiguity, no information loss).
# MAGIC Multi-category columns (`InternetService`, `PaymentMethod`, `MultipleLines`,
# MAGIC and the 6 add-on columns) are left as strings for one-hot encoding in the
# MAGIC model pipeline — collapsing them here would lose the "No internet service"
# MAGIC vs "No" distinction, which the model pipeline's encoder should see explicitly.

# COMMAND ----------

binary_map_cols = ["Partner", "Dependents", "PhoneService", "PaperlessBilling", "Churn"]

for c in binary_map_cols:
    df = df.withColumn(c + "_flag", F.when(F.col(c) == "Yes", 1).otherwise(0))

df = df.withColumn("gender_flag", F.when(F.col("gender") == "Male", 1).otherwise(0))

df.select("Partner", "Partner_flag", "Churn", "Churn_flag", "gender", "gender_flag").show(5)

# COMMAND ----------

# MAGIC %md ## 3. Final Feature Table

# COMMAND ----------

# MAGIC %md
# MAGIC Final column set: identifiers, raw categoricals kept for one-hot encoding,
# MAGIC engineered numerics, binary flags, and the target.

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

final_cols = ["customer_id"] + categorical_for_onehot + numeric_features + binary_features + [target_col]

features_df = df.select(*final_cols)

print(f"Final feature table: {len(final_cols)} columns")
features_df.printSchema()

# COMMAND ----------

# MAGIC %md ## 4. Train / Val / Test Split (Stratified)

# COMMAND ----------

# Convert to pandas for sklearn's stratified split (dataset is small — 7K rows)
features_pd = features_df.toPandas()

train_pd, temp_pd = train_test_split(
    features_pd, test_size=0.30, stratify=features_pd[target_col], random_state=42
)
val_pd, test_pd = train_test_split(
    temp_pd, test_size=0.50, stratify=temp_pd[target_col], random_state=42
)

print(f"Train: {len(train_pd):,} rows  |  churn rate: {train_pd[target_col].mean():.4f}")
print(f"Val:   {len(val_pd):,} rows  |  churn rate: {val_pd[target_col].mean():.4f}")
print(f"Test:  {len(test_pd):,} rows  |  churn rate: {test_pd[target_col].mean():.4f}")

# COMMAND ----------

# MAGIC %md ## 5. Write Splits to Delta

# COMMAND ----------

for name, pdf in [("train", train_pd), ("val", val_pd), ("test", test_pd)]:
    sdf = spark.createDataFrame(pdf)
    (
        sdf.write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(f"{DB_NAME}.features_{name}")
    )
    print(f"✓ Written: {DB_NAME}.features_{name}  ({len(pdf):,} rows)")

# COMMAND ----------

# MAGIC %md ## 6. Log Feature Engineering Run to MLflow

# COMMAND ----------

feature_metadata = {
    "categorical_for_onehot": categorical_for_onehot,
    "numeric_features": numeric_features,
    "binary_features": binary_features,
    "target_col": target_col,
    "engineered_features": {
        "tenure_bucket": "0-12 / 13-24 / 25-48 / 49+ months",
        "contract_encoded": "Month-to-month=0, One year=1, Two year=2",
        "num_addon_services": "count of Yes across 6 add-on service columns",
        "avg_monthly_spend": "TotalCharges / max(tenure, 1)",
        "charge_increase_ratio": "MonthlyCharges / avg_monthly_spend",
    },
}

with open("/tmp/feature_metadata.json", "w") as f:
    json.dump(feature_metadata, f, indent=2)

with mlflow.start_run(run_name="feature_engineering_v1") as run:

    mlflow.log_param("split_strategy", "70/15/15 stratified on Churn")
    mlflow.log_param("random_state", 42)
    mlflow.log_param("n_engineered_features", len(feature_metadata["engineered_features"]))

    mlflow.log_metric("train_rows", len(train_pd))
    mlflow.log_metric("val_rows", len(val_pd))
    mlflow.log_metric("test_rows", len(test_pd))
    mlflow.log_metric("train_churn_rate", train_pd[target_col].mean())
    mlflow.log_metric("val_churn_rate", val_pd[target_col].mean())
    mlflow.log_metric("test_churn_rate", test_pd[target_col].mean())

    mlflow.set_tag("delta.features_train", f"{DB_NAME}.features_train")
    mlflow.set_tag("delta.features_val", f"{DB_NAME}.features_val")
    mlflow.set_tag("delta.features_test", f"{DB_NAME}.features_test")
    mlflow.set_tag("notebook", "02_feature_engineering")

    mlflow.log_artifact("/tmp/feature_metadata.json", artifact_path="feature_metadata")

    print(f"\n✓ MLflow run logged: {run.info.run_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC | Table | Rows | Churn Rate |
# MAGIC |---|---|---|
# MAGIC | `churn.features_train` | ~4,930 | ~26.5% |
# MAGIC | `churn.features_val` | ~1,056 | ~26.5% |
# MAGIC | `churn.features_test` | ~1,057 | ~26.5% |
# MAGIC
# MAGIC **Feature counts:** 10 one-hot categoricals, 7 engineered/raw numerics, 6 binary flags, 1 target.
# MAGIC
# MAGIC **Next:** `03_training_and_tracking.py` — build the `ColumnTransformer` pipeline,
# MAGIC train Logistic Regression, Random Forest, XGBoost, and LightGBM, and log every
# MAGIC run to MLflow.
