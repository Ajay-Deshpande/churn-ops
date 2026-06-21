# Databricks notebook source
# MAGIC %md
# MAGIC # 05 — SHAP Explainability
# MAGIC
# MAGIC **Project:** Customer Churn — Production ML with MLflow
# MAGIC **Input:** Champion model (`champion_model.json` pointer from notebook 04), `churn.features_test`
# MAGIC **Output:** Global feature importance (beeswarm), 3 individual prediction explanations (waterfall)
# MAGIC
# MAGIC ### Why SHAP over built-in feature importance?
# MAGIC LightGBM's native `feature_importances_` only tells you *which* features
# MAGIC mattered on average across the training set — it can't tell you **why a
# MAGIC specific customer** was flagged as high-risk, and it doesn't show
# MAGIC direction (does high tenure push risk up or down?). SHAP gives both:
# MAGIC signed, per-prediction contributions that sum exactly to
# MAGIC `prediction - baseline`. This is what a retention team actually needs —
# MAGIC "why is customer X flagged?" not just "tenure matters."
# MAGIC
# MAGIC ### Explainer choice
# MAGIC `shap.TreeExplainer` — exact (not approximate) Shapley values for tree
# MAGIC ensembles in polynomial time, vs. `KernelExplainer`'s much slower
# MAGIC model-agnostic sampling approach. Since our champion is LightGBM,
# MAGIC TreeExplainer is both faster and exact.
# MAGIC
# MAGIC ### A note on the sklearn Pipeline wrapper
# MAGIC Our champion model is `Pipeline([("preprocess", ColumnTransformer), ("classifier", LGBMClassifier)])`.
# MAGIC SHAP's TreeExplainer needs the raw `LGBMClassifier` and the *already-transformed*
# MAGIC (one-hot encoded, scaled) feature matrix — it can't see through the
# MAGIC pipeline. So we split the pipeline into its two stages and explain
# MAGIC post-transformation feature names directly.

# COMMAND ----------

# MAGIC %md ## 0. Imports & Config

# COMMAND ----------

# DBTITLE 1,Cell 3
# MAGIC %pip install scikit-learn=1.9.0 lightgbm shap --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import json
import shap
import matplotlib.pyplot as plt

mlflow.set_experiment("/Workspace/Users/deshpande.ajay.us@gmail.com/churn-ops/05_shap_explainability")

DB_NAME = "churn"
CHAMPION_POINTER_PATH = "/Workspace/Users/deshpande.ajay.us@gmail.com/churn-ops/champion_model.json"

# COMMAND ----------

# MAGIC %md ## 1. Load the Champion Model
# MAGIC
# MAGIC Reads the pointer file written by notebook 04 — the same indirection a
# MAGIC `models:/name@champion` URI would give in a fully provisioned UC registry.

# COMMAND ----------

with open(CHAMPION_POINTER_PATH) as f:
    champion_info = json.load(f)

print(json.dumps(champion_info, indent=2))

champion_pipeline = mlflow.sklearn.load_model(champion_info["model_uri"])
print(f"\n✓ Loaded champion: {champion_info['model_family']} (run_id={champion_info['run_id']})")

# COMMAND ----------

# MAGIC %md ## 2. Load Test Data & Split the Pipeline

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
X_test = test_pd[feature_cols].reset_index(drop=True)
y_test = test_pd["Churn_flag"].reset_index(drop=True)
customer_ids = test_pd["customer_id"].reset_index(drop=True)

print(f"Test set: {X_test.shape}")

# COMMAND ----------

preprocessor = champion_pipeline.named_steps["preprocess"]
classifier = champion_pipeline.named_steps["classifier"]

# Transform test data through the fitted preprocessor (same transform used at training time)
X_test_transformed = preprocessor.transform(X_test)

# Recover human-readable feature names post-one-hot-encoding
onehot_feature_names = preprocessor.named_transformers_["onehot"].get_feature_names_out(categorical_for_onehot)
all_feature_names = list(onehot_feature_names) + numeric_features + binary_features

X_test_transformed_df = pd.DataFrame(
    X_test_transformed.toarray() if hasattr(X_test_transformed, "toarray") else X_test_transformed,
    columns=all_feature_names,
)

print(f"Transformed feature matrix: {X_test_transformed_df.shape}")
print(f"Feature names (first 10): {all_feature_names[:10]}")

# COMMAND ----------

# MAGIC %md ## 3. Compute SHAP Values

# COMMAND ----------

explainer = shap.TreeExplainer(classifier)
shap_values = explainer.shap_values(X_test_transformed_df)

# LightGBM binary classifier: shap_values is a single 2D array (contribution to the positive class)
# when using TreeExplainer in recent shap versions. Handle both possible return shapes defensively.
if isinstance(shap_values, list):
    shap_values_churn = shap_values[1]  # class 1 = churn
    expected_value = explainer.expected_value[1]
else:
    shap_values_churn = shap_values
    expected_value = explainer.expected_value

print(f"SHAP values shape: {shap_values_churn.shape}")
print(f"Expected value (baseline log-odds): {expected_value:.4f}")

# COMMAND ----------

# MAGIC %md ## 4. Global Explanation — Beeswarm Plot
# MAGIC
# MAGIC Shows, across all 1,057 test customers, which features matter most
# MAGIC (y-axis ranking) and which direction they push predictions (red = high
# MAGIC feature value, blue = low; right = pushes toward churn, left = pushes
# MAGIC away). This is the "what matters and why" view for a stakeholder
# MAGIC presentation.

# COMMAND ----------

fig = plt.figure(figsize=(10, 8))
shap.summary_plot(
    shap_values_churn, X_test_transformed_df,
    max_display=15, show=False
)
plt.title("Global Feature Importance — SHAP Beeswarm (Test Set)", fontsize=12)
plt.tight_layout()
plt.savefig("/Workspace/Users/deshpande.ajay.us@gmail.com/churn-ops/assets/plots/shap/shap_beeswarm_global.png", dpi=120, bbox_inches="tight")
plt.show()

# COMMAND ----------

# MAGIC %md ## 5. Global Explanation — Mean |SHAP| Bar Chart
# MAGIC
# MAGIC A simpler companion view — average magnitude of impact per feature,
# MAGIC without the directional detail of the beeswarm. Easier to read in a
# MAGIC README at a glance.

# COMMAND ----------

mean_abs_shap = pd.DataFrame({
    "feature": all_feature_names,
    "mean_abs_shap": np.abs(shap_values_churn).mean(axis=0),
}).sort_values("mean_abs_shap", ascending=False).head(15)

fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(mean_abs_shap["feature"][::-1], mean_abs_shap["mean_abs_shap"][::-1], color="steelblue")
ax.set_xlabel("Mean |SHAP value|")
ax.set_title("Top 15 Features by Mean Absolute SHAP Value")
plt.tight_layout()
plt.savefig("/Workspace/Users/deshpande.ajay.us@gmail.com/churn-ops/assets/plots/shap/shap_mean_abs_bar.png", dpi=120)
plt.show()

print(mean_abs_shap.to_string(index=False))

# COMMAND ----------

# MAGIC %md ## 6. Local Explanations — 3 Individual Customers
# MAGIC
# MAGIC We pick three representative cases from the test set:
# MAGIC - **High-risk**: highest predicted churn probability
# MAGIC - **Low-risk**: lowest predicted churn probability
# MAGIC - **Borderline**: probability closest to 0.5 — the cases a retention
# MAGIC   team would most want explained, since the model itself is uncertain

# COMMAND ----------

churn_proba = champion_pipeline.predict_proba(X_test)[:, 1]

high_risk_idx = np.argmax(churn_proba)
low_risk_idx = np.argmin(churn_proba)
borderline_idx = np.argmin(np.abs(churn_proba - 0.5))

selected_cases = {
    "high_risk": high_risk_idx,
    "low_risk": low_risk_idx,
    "borderline": borderline_idx,
}

for case_name, idx in selected_cases.items():
    print(f"{case_name:12s} | customer_id={customer_ids[idx]:12s} | "
          f"predicted_proba={churn_proba[idx]:.4f} | actual_churn={y_test[idx]}")

# COMMAND ----------

# MAGIC %md ### 6.1 Waterfall Plots

# COMMAND ----------

for case_name, idx in selected_cases.items():
    fig = plt.figure(figsize=(9, 6))

    explanation = shap.Explanation(
        values=shap_values_churn[idx],
        base_values=expected_value,
        data=X_test_transformed_df.iloc[idx].values,
        feature_names=all_feature_names,
    )

    shap.waterfall_plot(explanation, max_display=12, show=False)
    plt.title(
        f"{case_name.replace('_', ' ').title()} — customer_id={customer_ids[idx]}\n"
        f"Predicted P(churn)={churn_proba[idx]:.3f} | Actual churn={'Yes' if y_test[idx] == 1 else 'No'}",
        fontsize=10
    )
    plt.tight_layout()
    plt.savefig(f"/Workspace/Users/deshpande.ajay.us@gmail.com/churn-ops/assets/plots/shap/shap_waterfall_{case_name}.png", dpi=120, bbox_inches="tight")
    plt.show()

# COMMAND ----------

# MAGIC %md ## 7. Business Translation
# MAGIC What do the top SHAP features actually mean for a retention team?
# MAGIC (Based on this run's actual mean |SHAP| ranking.)
# MAGIC - **`contract_encoded` (mean |SHAP| = 0.776, by far the top feature)**:
# MAGIC   contract length is the single strongest churn driver — more than
# MAGIC   double the next feature. Month-to-month customers have no switching
# MAGIC   cost. Actionable: prioritize contract-upgrade incentives (discounted
# MAGIC   annual plans) for month-to-month customers, especially newer ones.
# MAGIC - **`tenure` (0.376)**: the second-strongest signal — new customers
# MAGIC   churn disproportionately. Actionable: a structured onboarding/early-
# MAGIC   tenure retention program (first 90-180 days) likely has outsized ROI.
# MAGIC - **`OnlineSecurity_No` (0.250) and `TechSupport_No` (0.170)**:
# MAGIC   customers without these add-ons churn more — likely a proxy for
# MAGIC   lower overall engagement/investment in the service, not necessarily
# MAGIC   that the add-on itself prevents churn. Actionable: low-cost bundled
# MAGIC   trials of these add-ons could be tested as a retention lever for
# MAGIC   at-risk segments.
# MAGIC - **`InternetService_Fiber optic` (0.177)**: fiber customers show
# MAGIC   higher churn risk than DSL — likely reflects price sensitivity or
# MAGIC   stronger competitive pressure in the fiber market segment specifically.
# MAGIC - **`PaymentMethod_Electronic check` (0.162)**: electronic check payers
# MAGIC   churn more than autopay/credit-card customers — a known pattern often
# MAGIC   linked to lower payment-method "stickiness" (no recurring autopay
# MAGIC   relationship with the company). Actionable: incentivize migration to
# MAGIC   autopay.
# MAGIC - **`MonthlyCharges` / `avg_monthly_spend` (0.132 / 0.124)**: higher
# MAGIC   bills correlate with higher churn risk, consistent with price
# MAGIC   sensitivity as a churn driver across the dataset.
# MAGIC **Headline insight for a retention team:** the top 2 features
# MAGIC (`contract_encoded`, `tenure`) together account for more SHAP impact
# MAGIC than the next 6 features combined. A retention strategy targeting
# MAGIC *new, month-to-month customers* specifically would address the two
# MAGIC dominant churn drivers simultaneously.

# COMMAND ----------

# MAGIC %md ## 8. Log SHAP Run to MLflow

# COMMAND ----------

with mlflow.start_run(run_name=f"shap_explainability_{champion_info['model_family']}") as run:

    mlflow.log_param("champion_run_id", champion_info["run_id"])
    mlflow.log_param("model_family", champion_info["model_family"])
    mlflow.log_param("explainer_type", "TreeExplainer")
    mlflow.log_param("n_test_samples_explained", len(X_test_transformed_df))

    mlflow.log_metric("expected_value_baseline", float(expected_value))
    for _, row in mean_abs_shap.head(5).iterrows():
        mlflow.log_metric(f"mean_abs_shap__{row['feature']}", row["mean_abs_shap"])

    mlflow.log_artifact("/Workspace/Users/deshpande.ajay.us@gmail.com/churn-ops/assets/plots/shap/shap_beeswarm_global.png", artifact_path="shap_plots")
    mlflow.log_artifact("/Workspace/Users/deshpande.ajay.us@gmail.com/churn-ops/assets/plots/shap/shap_mean_abs_bar.png", artifact_path="shap_plots")
    for case_name in selected_cases:
        mlflow.log_artifact(f"/Workspace/Users/deshpande.ajay.us@gmail.com/churn-ops/assets/plots/shap/shap_waterfall_{case_name}.png", artifact_path="shap_plots")

    mean_abs_shap.to_csv("/tmp/shap_feature_ranking.csv", index=False)
    mlflow.log_artifact("/tmp/shap_feature_ranking.csv", artifact_path="shap_plots")

    mlflow.set_tag("notebook", "05_shap_explainability")
    mlflow.set_tag("champion_run_id", champion_info["run_id"])

    print(f"✓ SHAP run logged: {run.info.run_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC | Output | Description |
# MAGIC |---|---|
# MAGIC | Global beeswarm | Top 15 features, ranked by impact, with direction |
# MAGIC | Mean \|SHAP\| bar chart | Simplified ranking for README/stakeholder view |
# MAGIC | 3 waterfall plots | High-risk, low-risk, borderline individual explanations |
# MAGIC
# MAGIC **Next:** `06_drift_detection.py` — inject synthetic drift into a copy of
# MAGIC the test set, compute PSI/KS statistics, and trigger a retrain comparison
# MAGIC if drift is detected.
