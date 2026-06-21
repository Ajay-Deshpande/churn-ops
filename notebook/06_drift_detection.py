# Databricks notebook source
# MAGIC %md
# MAGIC # 06 — Drift Detection
# MAGIC
# MAGIC **Project:** Customer Churn — Production ML with MLflow
# MAGIC **Input:** `churn.features_test`, champion model pointer
# MAGIC **Output:** Drift report (PSI + KS), retrain decision, v2 model if triggered
# MAGIC
# MAGIC ### Why synthetic drift (recap from notebook 01)
# MAGIC We use synthetic drift injection rather than a second real dataset —
# MAGIC the standard pattern (used by Evidently AI and Databricks MLOps
# MAGIC reference implementations) for testing that monitoring and retrain-
# MAGIC trigger logic actually fires, since real drift is rare and slow to
# MAGIC accumulate naturally.
# MAGIC
# MAGIC ### Simulated scenario
# MAGIC "6 months later" — three realistic business changes:
# MAGIC 1. **Pricing change**: monthly charges increased ~15% (inflation /
# MAGIC    price hike)
# MAGIC 2. **Marketing shift**: a promo campaign brought in a wave of new,
# MAGIC    shorter-tenure customers — tenure distribution skews younger
# MAGIC 3. **Contract mix shift**: more customers on month-to-month plans
# MAGIC    (reflecting reduced uptake of annual contracts post price-hike)
# MAGIC
# MAGIC ### Drift metrics
# MAGIC - **PSI (Population Stability Index)** on numeric features — industry
# MAGIC   standard thresholds: PSI < 0.1 = no significant shift, 0.1-0.2 =
# MAGIC   moderate, > 0.2 = severe drift, investigate/retrain
# MAGIC - **KS test** (Kolmogorov-Smirnov) on the same numeric features — a
# MAGIC   complementary statistical test; p < 0.05 indicates the two
# MAGIC   distributions are significantly different
# MAGIC
# MAGIC ### Retrain trigger
# MAGIC If **any** feature shows PSI > 0.2, we retrain on the combined
# MAGIC (original + drifted) data and compare v1 vs. v2 in MLflow.

# COMMAND ----------

# MAGIC %pip install --upgrade scikit-learn=1.9.0 lightgbm --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md ## 0. Imports & Config

# COMMAND ----------

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score
from scipy.stats import randint, uniform, loguniform
from lightgbm import LGBMClassifier

mlflow.set_registry_uri("uc-databricks")
mlflow.set_experiment("/Workspace/Users/deshpande.ajay.us@gmail.com/churn-ops/06_drift_detection")

DB_NAME = "churn"
CHAMPION_POINTER_PATH = "/Workspace/Users/deshpande.ajay.us@gmail.com/churn-ops/champion_model.json"
RANDOM_STATE = 42
PSI_RETRAIN_THRESHOLD = 0.2

# COMMAND ----------

# MAGIC %md ## 1. Load Champion Model & Reference Data

# COMMAND ----------

with open(CHAMPION_POINTER_PATH) as f:
    champion_info = json.load(f)

champion_pipeline = mlflow.sklearn.load_model(champion_info["model_uri"])
print(f"✓ Loaded champion: {champion_info['model_family']} (test_roc_auc={champion_info['test_roc_auc']:.4f})")

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
target_col = "Churn_flag"

# Reference distribution = the test set the champion was validated on
reference_pd = spark.table(f"{DB_NAME}.features_test").toPandas().reset_index(drop=True)
print(f"Reference (test) set: {reference_pd.shape}")

# COMMAND ----------

# MAGIC %md ## 2. Inject Synthetic Drift
# MAGIC
# MAGIC We drift a **fresh copy of the training set** (not the test set) to
# MAGIC create the "month 6" batch — this also gives us a properly-sized batch
# MAGIC for the eventual retrain step, with the same underlying customer
# MAGIC population the model was originally trained on, just shifted.

# COMMAND ----------

train_pd = spark.table(f"{DB_NAME}.features_train").toPandas().reset_index(drop=True)
drift_pd = train_pd.copy()

np.random.seed(RANDOM_STATE)

# ── 1. Pricing change: +15% monthly charges, recompute dependent features ──
drift_pd["MonthlyCharges"] = drift_pd["MonthlyCharges"] * 1.15

# ── 2. Marketing shift: skew tenure younger ─────────────────────────────────
# Multiply tenure by a random factor < 1 for ~40% of customers (new promo cohort)
promo_cohort_mask = np.random.rand(len(drift_pd)) < 0.40
tenure_shrink_factor = np.random.uniform(0.1, 0.5, size=promo_cohort_mask.sum())
drift_pd.loc[promo_cohort_mask, "tenure"] = (
    drift_pd.loc[promo_cohort_mask, "tenure"] * tenure_shrink_factor
).round().astype(int).clip(lower=0)

# ── 3. Contract mix shift: push some long-contract customers to month-to-month ──
long_contract_mask = drift_pd["contract_encoded"] > 0
shift_to_mtm_mask = long_contract_mask & (np.random.rand(len(drift_pd)) < 0.25)
drift_pd.loc[shift_to_mtm_mask, "contract_encoded"] = 0

# ── Recompute dependent engineered features after the above shifts ──────────
drift_pd["tenure_bucket"] = pd.cut(
    drift_pd["tenure"], bins=[-1, 12, 24, 48, np.inf],
    labels=["0-12", "13-24", "25-48", "49+"]
).astype(str)

# TotalCharges and avg_monthly_spend recomputed to stay internally consistent
# with the new tenure/MonthlyCharges values
drift_pd["TotalCharges"] = np.where(
    drift_pd["tenure"] == 0,
    0.0,
    drift_pd["MonthlyCharges"] * drift_pd["tenure"] * np.random.uniform(0.85, 1.0, size=len(drift_pd))
)
drift_pd["avg_monthly_spend"] = np.where(
    drift_pd["tenure"] == 0,
    drift_pd["MonthlyCharges"],
    drift_pd["TotalCharges"] / drift_pd["tenure"]
)
drift_pd["charge_increase_ratio"] = (drift_pd["MonthlyCharges"] / drift_pd["avg_monthly_spend"]).round(4)

print(f"Drift batch: {drift_pd.shape}")
print(f"\nMonthlyCharges  — reference mean: {reference_pd['MonthlyCharges'].mean():.2f}  |  drift mean: {drift_pd['MonthlyCharges'].mean():.2f}")
print(f"tenure          — reference mean: {reference_pd['tenure'].mean():.2f}  |  drift mean: {drift_pd['tenure'].mean():.2f}")
print(f"contract_encoded mtm%  — reference: {(reference_pd['contract_encoded']==0).mean():.2%}  |  drift: {(drift_pd['contract_encoded']==0).mean():.2%}")

# COMMAND ----------

# MAGIC %md ## 3. PSI (Population Stability Index)
# MAGIC
# MAGIC PSI compares the binned distribution of a feature in the reference
# MAGIC vs. current population. We use 10 quantile bins derived from the
# MAGIC reference set, then measure how the current population's bin
# MAGIC proportions diverge.

# COMMAND ----------

def calculate_psi(reference, current, n_bins=10):
    """
    PSI = sum[ (current_pct - ref_pct) * ln(current_pct / ref_pct) ] over bins.
    Bin edges are quantiles of the reference distribution.
    """
    breakpoints = np.quantile(reference, np.linspace(0, 1, n_bins + 1))
    breakpoints = np.unique(breakpoints)  # handle duplicate edges from skewed data
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    ref_counts, _ = np.histogram(reference, bins=breakpoints)
    cur_counts, _ = np.histogram(current, bins=breakpoints)

    ref_pct = np.where(ref_counts == 0, 1e-4, ref_counts / len(reference))
    cur_pct = np.where(cur_counts == 0, 1e-4, cur_counts / len(current))

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return psi

# COMMAND ----------

numeric_for_drift = ["tenure", "MonthlyCharges", "TotalCharges", "contract_encoded",
                      "avg_monthly_spend", "charge_increase_ratio", "num_addon_services"]

psi_results = []
for col in numeric_for_drift:
    psi = calculate_psi(reference_pd[col].values, drift_pd[col].values)
    psi_results.append({"feature": col, "psi": psi})

psi_df = pd.DataFrame(psi_results).sort_values("psi", ascending=False)

def psi_flag(psi):
    if psi > 0.2:
        return "SEVERE"
    elif psi > 0.1:
        return "MODERATE"
    return "OK"

psi_df["flag"] = psi_df["psi"].apply(psi_flag)
print(psi_df.to_string(index=False))

# COMMAND ----------

# MAGIC %md ## 4. KS Test
# MAGIC
# MAGIC Complementary statistical test — confirms PSI findings with a p-value.

# COMMAND ----------

ks_results = []
for col in numeric_for_drift:
    stat, p_value = ks_2samp(reference_pd[col].values, drift_pd[col].values)
    ks_results.append({"feature": col, "ks_statistic": stat, "p_value": p_value,
                        "significant_drift": p_value < 0.05})

ks_df = pd.DataFrame(ks_results).sort_values("ks_statistic", ascending=False)
print(ks_df.to_string(index=False))

# COMMAND ----------

# MAGIC %md ## 5. Drift Report Visualization

# COMMAND ----------

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()

top_drifted = psi_df.head(6)["feature"].tolist() if len(psi_df) >= 6 else psi_df["feature"].tolist()

for i, col in enumerate(top_drifted):
    ax = axes[i]
    ax.hist(reference_pd[col], bins=30, alpha=0.5, label="Reference (test)", density=True, color="steelblue")
    ax.hist(drift_pd[col], bins=30, alpha=0.5, label="Current (drift batch)", density=True, color="indianred")
    psi_val = psi_df[psi_df["feature"] == col]["psi"].values[0]
    ax.set_title(f"{col}  (PSI={psi_val:.3f})", fontsize=10)
    ax.legend(fontsize=8)

plt.suptitle("Drift Report — Reference vs. Current Distributions", fontsize=13)
plt.tight_layout()
plt.savefig("/Workspace/Users/deshpande.ajay.us@gmail.com/churn-ops/assets/plots/drift/drift_report.png", dpi=110)
plt.show()

# COMMAND ----------

# MAGIC %md ## 6. Retrain Decision

# COMMAND ----------

severe_drift_features = psi_df[psi_df["flag"] == "SEVERE"]["feature"].tolist()
retrain_triggered = len(severe_drift_features) > 0

print(f"Features with severe drift (PSI > {PSI_RETRAIN_THRESHOLD}): {severe_drift_features}")
print(f"\nRetrain triggered: {retrain_triggered}")

# COMMAND ----------

# MAGIC %md ## 7. Score Drift Batch with Champion (v1) — Performance Check
# MAGIC
# MAGIC Before retraining, confirm the *business* impact of the drift: does
# MAGIC v1's predictive performance actually degrade on the drifted population?
# MAGIC (Note: drift_pd retains its original true Churn_flag labels — only the
# MAGIC feature values were perturbed, not the outcomes — so we can still score
# MAGIC accuracy here, which wouldn't be possible with truly unlabeled production data.)

# COMMAND ----------

X_drift = drift_pd[feature_cols]
y_drift = drift_pd[target_col]

v1_drift_proba = champion_pipeline.predict_proba(X_drift)[:, 1]
v1_roc_auc_on_drift = roc_auc_score(y_drift, v1_drift_proba)

print(f"Champion (v1) ROC-AUC on reference test set : {champion_info['test_roc_auc']:.4f}")
print(f"Champion (v1) ROC-AUC on drifted batch       : {v1_roc_auc_on_drift:.4f}")
print(f"Performance delta                            : {v1_roc_auc_on_drift - champion_info['test_roc_auc']:+.4f}")

# COMMAND ----------

# MAGIC %md ## 8. Retrain (v2) — Combined Original + Drifted Data
# MAGIC
# MAGIC Only runs if retrain was triggered. We retrain LightGBM (same family
# MAGIC as champion, for a fair v1-vs-v2 comparison) on the union of original
# MAGIC training data and the drifted batch — the realistic approach when new
# MAGIC production patterns emerge: incorporate them rather than discard
# MAGIC history entirely.

# COMMAND ----------

if retrain_triggered:

    combined_pd = pd.concat([train_pd, drift_pd], ignore_index=True)
    print(f"Combined retrain set: {combined_pd.shape}  (original {len(train_pd)} + drift {len(drift_pd)})")

    X_combined = combined_pd[feature_cols]
    y_combined = combined_pd[target_col]

    preprocessor_v2 = ColumnTransformer(
        transformers=[
            ("onehot", OneHotEncoder(handle_unknown="ignore"), categorical_for_onehot),
            ("scale", StandardScaler(), numeric_features),
            ("passthrough", "passthrough", binary_features),
        ]
    )

    param_dist_lgbm = {
        "classifier__n_estimators": randint(100, 500),
        "classifier__max_depth": randint(3, 12),
        "classifier__learning_rate": loguniform(0.01, 0.3),
        "classifier__num_leaves": randint(20, 150),
        "classifier__subsample": uniform(0.6, 0.4),
        "classifier__colsample_bytree": uniform(0.6, 0.4),
        "classifier__class_weight": [None, "balanced"],
    }

    pipeline_v2 = Pipeline([
        ("preprocess", preprocessor_v2),
        ("classifier", LGBMClassifier(random_state=RANDOM_STATE, verbose=-1)),
    ])

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    search = RandomizedSearchCV(
        pipeline_v2, param_distributions=param_dist_lgbm, n_iter=20,
        cv=cv, scoring="roc_auc", random_state=RANDOM_STATE, n_jobs=-1,
    )

    with mlflow.start_run(run_name="lightgbm_retrain_v2_post_drift") as retrain_run:
        search.fit(X_combined, y_combined)
        v2_model = search.best_estimator_

        best_params_v2 = {k.replace("classifier__", ""): v for k, v in search.best_params_.items()}
        mlflow.log_params(best_params_v2)
        mlflow.log_param("retrain_trigger", "synthetic_drift_PSI")
        mlflow.log_param("severe_drift_features", ",".join(severe_drift_features))
        mlflow.log_param("combined_train_rows", len(combined_pd))
        mlflow.log_metric("cv_best_roc_auc", search.best_score_)

        # Evaluate v2 on BOTH the original reference test set AND the drift batch
        v2_test_proba = v2_model.predict_proba(reference_pd[feature_cols])[:, 1]
        v2_test_roc_auc = roc_auc_score(reference_pd[target_col], v2_test_proba)

        v2_drift_proba = v2_model.predict_proba(X_drift)[:, 1]
        v2_drift_roc_auc = roc_auc_score(y_drift, v2_drift_proba)

        mlflow.log_metric("v2_roc_auc_on_reference_test", v2_test_roc_auc)
        mlflow.log_metric("v2_roc_auc_on_drift_batch", v2_drift_roc_auc)

        mlflow.log_artifact("/Workspace/Users/deshpande.ajay.us@gmail.com/churn-ops/assets/plots/shap/drift_report.png", artifact_path="drift_plots")

        mlflow.sklearn.log_model(
            sk_model=v2_model,
            artifact_path="model",
            input_example=X_combined.head(5),
        )

        mlflow.set_tag("notebook", "06_drift_detection")
        mlflow.set_tag("model_family", "lightgbm")
        mlflow.set_tag("registry_status", "challenger_v2")

        v2_run_id = retrain_run.info.run_id
        print(f"\n✓ v2 retrained and logged: {v2_run_id}")

    print(f"\n{'='*55}")
    print(f"  v1 vs v2 Comparison")
    print(f"{'='*55}")
    print(f"  v1 ROC-AUC on reference test : {champion_info['test_roc_auc']:.4f}")
    print(f"  v2 ROC-AUC on reference test : {v2_test_roc_auc:.4f}")
    print(f"  v1 ROC-AUC on drift batch    : {v1_roc_auc_on_drift:.4f}")
    print(f"  v2 ROC-AUC on drift batch    : {v2_drift_roc_auc:.4f}")

else:
    print("No retrain needed — drift below threshold.")
    v2_run_id = None

# COMMAND ----------

# MAGIC %md ## 9. Promote v2 (if it outperforms v1 on the drift batch)

# COMMAND ----------

if retrain_triggered and v2_run_id is not None:
    if v2_drift_roc_auc > v1_roc_auc_on_drift:
        from mlflow.tracking import MlflowClient
        client = MlflowClient()

        # Demote v1
        client.set_tag(champion_info["run_id"], "registry_status", "archived")

        # Promote v2
        client.set_tag(v2_run_id, "registry_status", "champion")
        client.set_tag(v2_run_id, "test_roc_auc", f"{v2_test_roc_auc:.4f}")

        new_champion_pointer = {
            "model_logical_name": "churn_classifier",
            "run_id": v2_run_id,
            "model_uri": f"runs:/{v2_run_id}/model",
            "model_family": "lightgbm",
            "test_roc_auc": v2_test_roc_auc,
            "drift_batch_roc_auc": v2_drift_roc_auc,
            "retrained_due_to_drift": True,
            "previous_champion_run_id": champion_info["run_id"],
        }

        with open(CHAMPION_POINTER_PATH, "w") as f:
            json.dump(new_champion_pointer, f, indent=2)

        print(f"✓ v2 promoted to champion — pointer updated")
        print(json.dumps(new_champion_pointer, indent=2))
    else:
        print(f"v2 ({v2_drift_roc_auc:.4f}) did not outperform v1 ({v1_roc_auc_on_drift:.4f}) "
              f"on the drift batch — keeping v1 as champion.")
else:
    print("No promotion needed — v1 remains champion.")

# COMMAND ----------

# MAGIC %md ## 10. Log Drift Detection Summary Run

# COMMAND ----------

with mlflow.start_run(run_name="drift_detection_summary") as run:
    mlflow.log_param("psi_retrain_threshold", PSI_RETRAIN_THRESHOLD)
    mlflow.log_metric("n_severe_drift_features", len(severe_drift_features))
    mlflow.log_metric("retrain_triggered", int(retrain_triggered))
    mlflow.log_metric("v1_roc_auc_on_drift_batch", v1_roc_auc_on_drift)

    if retrain_triggered and v2_run_id is not None:
        mlflow.log_metric("v2_roc_auc_on_drift_batch", v2_drift_roc_auc)
        mlflow.log_metric("v2_roc_auc_on_reference_test", v2_test_roc_auc)
        mlflow.set_tag("v2_run_id", v2_run_id)

    psi_df.to_csv("/tmp/psi_results.csv", index=False)
    ks_df.to_csv("/tmp/ks_results.csv", index=False)
    mlflow.log_artifact("/tmp/psi_results.csv", artifact_path="drift_report")
    mlflow.log_artifact("/tmp/ks_results.csv", artifact_path="drift_report")
    mlflow.log_artifact("/Workspace/Users/deshpande.ajay.us@gmail.com/churn-ops/assets/plots/drift/drift_report.png", artifact_path="drift_report")

    mlflow.set_tag("notebook", "06_drift_detection")

    print(f"✓ Drift detection summary logged: {run.info.run_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC | Step | Result |
# MAGIC |---|---|
# MAGIC | Features with severe drift | *(see PSI table above)* |
# MAGIC | Retrain triggered | *(True/False)* |
# MAGIC | v1 vs v2 on drift batch | *(see comparison above)* |
# MAGIC | Final champion | *(see champion_model.json)* |
# MAGIC
# MAGIC **Next:** `07_batch_scoring.py` — load the current champion, score a
# MAGIC new batch of customers, write predictions with risk tiers to Delta.
# MAGIC This notebook becomes the scheduled job task.
