# Spark & MLflow — Interview Quick Reference

---

## PYSPARK

### Reading/Writing Data
```python
spark.read.option("header","true").option("inferSchema","true").csv(path)
df.write.format("delta").mode("overwrite"/"append").option("overwriteSchema","true").saveAsTable("db.table")
spark.table("db.table")
spark.sql("CREATE DATABASE IF NOT EXISTS db")
spark.sql("CREATE SCHEMA IF NOT EXISTS catalog.schema")
```
**Why:** Delta = ACID transactions, time travel, schema enforcement on a data lake. `overwriteSchema` needed when column types/structure change between runs.

### Column Operations — `pyspark.sql.functions as F`
```python
F.col("x"), F.lit(value), F.when(cond, val).otherwise(val)
F.trim(), F.round(col, n)
df.withColumn("new_col", expr)
df.withColumnRenamed("old", "new")
df.select(*cols)
```
**Why `when/otherwise`:** Spark-native conditional logic — vectorized, avoids UDF overhead (UDFs are slow, not optimized by Catalyst).

### Aggregation / Profiling
```python
df.groupBy("col").count()
df.select([F.mean(F.col(c).isNull().cast("int")) for c in cols])  # null rate per column
df.count(), len(df.columns)
```
**Why:** Used for data quality checks (null %, target distribution) before training — standard ingestion validation step.

### Conversions
```python
df.toPandas()         # Spark -> pandas (small/medium data, sklearn needs pandas)
spark.createDataFrame(pandas_df)   # pandas -> Spark (to write back to Delta)
```
**Why:** sklearn/SHAP only work on pandas/numpy. Spark handles ingestion + feature engineering at scale; modeling drops to pandas once data is small enough (this project: ~7K rows).

### Key concept to articulate in interviews
- **Spark = distributed, lazy evaluation, used for ETL/feature engineering at scale**
- **pandas/sklearn = single-node, used for modeling once data fits in memory**
- Common interview Q: "Why not Spark ML for modeling?" → Spark ML has weaker model APIs (no native LightGBM, limited SHAP support); sklearn ecosystem is richer for tabular ML at small-medium scale.

### Why Spark at all (conceptual)
- **Problem it solves:** single-machine tools (pandas) can't process data that doesn't fit in one machine's RAM. Spark partitions data across a cluster and processes partitions in parallel.
- **Lazy evaluation:** transformations (`select`, `filter`, `withColumn`) build a logical plan; nothing executes until an action (`count`, `show`, `write`, `collect`) is called. Lets Catalyst optimizer rewrite/optimize the whole plan before running — e.g. predicate pushdown, column pruning.
- **Catalyst optimizer / Tungsten:** Catalyst = query plan optimizer (like a SQL engine's). Tungsten = physical execution engine (memory management, codegen). Why Spark SQL/DataFrame API is faster than raw RDDs.
- **DataFrame API vs RDD:** DataFrames are higher-level, schema-aware, optimized by Catalyst. RDDs = low-level, no built-in optimization. Always prefer DataFrame/SQL API unless you need fine-grained control Spark doesn't expose.
- **Partitions:** data is split into partitions, each processed by a task on an executor. Too few partitions = underutilized cluster; too many = scheduling overhead. (`df.rdd.getNumPartitions()`, `df.repartition(n)`)
- **Why Delta Lake over plain Parquet/CSV:** ACID transactions on a data lake (safe concurrent writes), schema enforcement/evolution, time travel (query old table versions), upserts via `MERGE`. Plain files have none of this — a failed write can corrupt/duplicate data.
- **Where Spark fits in this project specifically:** ingestion (reading raw CSV, validating schema/nulls at scale) and feature engineering (column transforms across the full dataset) — i.e., the steps that would need to scale if the dataset were millions of rows instead of 7K. Modeling drops to pandas because sklearn/SHAP need single-node arrays and the post-feature-engineering data is small.

---

## MLFLOW

### Setup
```python
mlflow.set_tracking_uri(...)        # where runs are logged
mlflow.set_registry_uri("databricks-uc")   # UC registry
mlflow.set_experiment("/path/to/experiment")
```

### Logging a Run
```python
with mlflow.start_run(run_name="..."):
    mlflow.log_param("key", value)
    mlflow.log_params({...})          # bulk
    mlflow.log_metric("key", value)
    mlflow.log_metrics({...})         # bulk
    mlflow.set_tag("key", value)
    mlflow.log_artifact("/tmp/file.png", artifact_path="plots")
    mlflow.sklearn.log_model(sk_model=model, artifact_path="model", input_example=X.head())
```
**param vs metric vs tag:**
- `param` = input/config (hyperparameters, thresholds) — immutable per run
- `metric` = output/numeric result (AUC, F1) — can log multiple steps/timesteps
- `tag` = metadata/labels (model_family, status) — used for filtering/search

### Autolog
```python
mlflow.sklearn.autolog()
# ... train ...
mlflow.sklearn.autolog(disable=True)
```
**Why/when:** fast exploration, auto-captures params + default metrics. Switch to manual logging when you need **consistent metric names across model families** (autolog names differ between sklearn/XGBoost/LightGBM) or custom artifacts.

### Querying Runs (decoupled promotion pattern)
```python
mlflow.search_runs(
    experiment_names=[...],
    filter_string="tags.notebook = 'x' and metrics.val_roc_auc > 0.8",
    order_by=["metrics.val_roc_auc DESC"],
    max_results=4,
)
```
**Why:** real promotion/CI step queries the tracking server fresh rather than reusing in-memory variables from training — decouples training and deployment as separate processes/sessions.

### Loading a Model
```python
mlflow.sklearn.load_model("runs:/<run_id>/model")
mlflow.sklearn.load_model("models:/<name>/Staging")          # legacy registry
mlflow.sklearn.load_model("models:/<catalog>.<schema>.<name>@champion")  # UC registry
mlflow.pyfunc.load_model(...)   # generic flavor — .predict() only, no predict_proba
```
**Why sklearn flavor over pyfunc:** need `predict_proba` for ROC-AUC/PR-AUC; pyfunc only exposes generic `.predict()`.

### Model Registry (MlflowClient)
```python
from mlflow.tracking import MlflowClient
client = MlflowClient()

client.create_registered_model(name, description="...")
mlflow.register_model(model_uri, name)              # returns ModelVersion(version=...)
client.update_model_version(name, version, description="...")
client.set_model_version_tag(name, version, key, value)

# Legacy registry (Staging/Production)
client.transition_model_version_stage(name, version, stage="Staging"/"Production", archive_existing_versions=True)

# Unity Catalog registry (aliases — current standard)
client.set_registered_model_alias(name, alias="champion"/"challenger", version)
client.search_model_versions(f"name='{name}'")   # returns list w/ .version, .current_stage, .tags, .aliases
```
**Why aliases > stages:** UC has no Staging/Production concept; aliases are flexible labels, exclusive per name (setting `@champion` on a new version auto-moves it off the old one — no manual archive step needed).

### Why MLflow at all (conceptual)
- **Problem it solves:** without it, experiment results live in scattered notebook print statements / memory — unreproducible, unsearchable, no audit trail. MLflow gives every training run a persistent, queryable record.
- **4 components:** Tracking (log params/metrics/artifacts per run), Models (standard packaging format — "flavors" — so any model can be loaded/served the same way regardless of library), Model Registry (versioning + stage/alias-based promotion), Projects (packaging code for reproducible runs — less commonly used day-to-day).
- **"Flavor" concept:** MLflow saves a model in a generic format (`pyfunc`) plus library-specific format (`sklearn`, `xgboost`, etc). `pyfunc` lets you load ANY MLflow model the same way (`.predict()`) without knowing what trained it — this is what model-serving infra relies on. Library-specific flavor (e.g. `mlflow.sklearn`) gives back the native object with full API (`predict_proba`, etc).
- **Why tracking server is decoupled from training code:** mirrors real CI/CD — a separate "promote" process queries the tracking server for the best run rather than depending on Python variables from a training session that may no longer exist.
- **Where it fits in this project:** every notebook (even ingestion) logs a run — establishes lineage ("which data/model produced this artifact?") across the entire pipeline, not just the training step.


- **Experiment vs Run:** experiment = container/project; run = single execution (one training job)
- **Why log artifacts not just metrics:** reproducibility — confusion matrices/ROC curves let you visually audit a run later without rerunning code
- **Model Registry purpose:** central versioned store + stage/alias-based promotion workflow = decouples "which model is live" from "which code trained it"
- **Validation gate pattern:** register → score on untouched holdout → promote only if threshold met (e.g. `test_roc_auc >= 0.80`) — this is the ML equivalent of a CI/CD quality gate
- **Why `runs:/` URI as fallback:** if registry is unavailable (permissions/infra), a run's artifact URI is still a stable, addressable reference — same lineage, no registry-specific API needed
