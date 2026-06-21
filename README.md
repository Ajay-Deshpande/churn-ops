# Customer Churn — Production ML with MLflow

End-to-end churn prediction system on Databricks, covering the full ML
lifecycle: ingestion, feature engineering, experiment tracking, model
registry, explainability, drift monitoring, and scheduled batch scoring.
This isn't just a churn model — it's a demonstration of how a production
ML team builds, validates, deploys, monitors, and maintains one.

## Architecture

```
01 Ingestion ──▶ 02 Feature Eng. ──▶ 03 Train/Track ──▶ 04 Registry
                                                              │
                                  ┌───────────────────────────┴───────┐
                                  ▼                                   ▼
                          05 SHAP Explain.                    06 Drift Detect.
                          (global + local)                    (PSI/KS, retrain)
                                  │                                   │
                                  └───────────────┬───────────────────┘
                                                   ▼
                                          07 Batch Scoring
                                          (scheduled job)
```

## Dataset

[IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
— 7,043 customers, 21 features, 26.5% churn rate.

**Why one dataset, not two:** a second real telecom dataset (Cell2Cell) was
considered for drift detection, but only ~7 features mapped cleanly across
both schemas — dropping the strongest predictors and cutting expected AUC
from ~0.84 to ~0.70. Instead, notebook 06 uses **synthetic drift
injection**, the same pattern Evidently AI and Databricks' own MLOps
reference implementations use to validate monitoring/retrain logic, since
real drift is rare and slow to occur naturally.

## Pipeline & Results

**01 — Ingestion.** Kaggle → Delta, schema/null validation, MLflow lineage logging.

**02 — Feature Engineering.** Five engineered features (`tenure_bucket`,
`contract_encoded`, `num_addon_services`, `avg_monthly_spend`,
`charge_increase_ratio`). Stratified 70/15/15 split.

**03 — Training & Tracking.** Four models via `RandomizedSearchCV` (3-fold,
20 iter), all logged to MLflow.

| Model | Val ROC-AUC | Val PR-AUC |
|---|---|---|
| **LightGBM** | **0.8510** | 0.6515 |
| XGBoost | 0.8480 | 0.6431 |
| Logistic Regression | 0.8470 | 0.6338 |
| Random Forest | 0.8448 | 0.6417 |

**04 — Registry.** Best run validated on the held-out test set
(`test_roc_auc >= 0.80` gate). LightGBM passed at **0.8373**.

> **Infra note:** this workspace's legacy registry is disabled, and Unity
> Catalog registration failed on S3 write permissions (free-tier
> limitation, not fixable from notebook code). UC registration code is
> included as reference; the working pipeline uses a `runs:/<run_id>/model`
> URI persisted to a JSON pointer file as the "champion" reference —
> functionally identical promotion logic, different storage mechanism.

**05 — SHAP Explainability.** `TreeExplainer` on the champion model — global
beeswarm, mean-\|SHAP\| ranking, and 3 local waterfall explanations
(high-risk, low-risk, borderline).

Top features: `contract_encoded` (0.776), `tenure` (0.376),
`OnlineSecurity_No` (0.250), `InternetService_Fiber optic` (0.177). The top
two alone outweigh the next six combined — a retention strategy targeting
*new, month-to-month customers* hits both dominant drivers at once.

**06 — Drift Detection.** Synthetic "6 months later" scenario (+15% pricing,
younger tenure mix, more month-to-month). PSI/KS correctly flagged severe
drift on `charge_increase_ratio` (PSI=4.33) and `MonthlyCharges` (PSI=1.01),
triggering an automatic retrain.

| Model | ROC-AUC on reference test | ROC-AUC on drift batch |
|---|---|---|
| v1 (pre-drift) | 0.8373 | 0.8469 |
| **v2 (retrained)** | **0.8375** | **0.9327** |

v2 gained +0.086 ROC-AUC on the drifted population with zero regression on
the original test set — promoted to champion automatically.

**07 — Batch Scoring.** Reads the champion pointer fresh every run (no code
change needed after a retrain), scores customers, assigns risk tiers
(High/Medium/Low), appends to Delta with full lineage. **This is the
scheduled job task** — see `job_config.json` (daily 6am, currently paused).

## Key Design Decisions

- **sklearn over Spark ML**: 7K rows fits comfortably in pandas; SHAP's
  mature tooling targets sklearn directly. Spark handles ingestion/feature
  engineering at scale instead.
- **Manual MLflow logging over pure autolog**: consistent metric names and
  custom artifacts (confusion matrices, ROC curves) across model families.
- **Synthetic drift over a second real dataset**: preserves the full,
  high-signal feature set while still exercising a genuine drift-detection
  and retrain pipeline.

## Tech Stack

Databricks (Delta Lake, MLflow, Jobs) · scikit-learn · XGBoost · LightGBM ·
SHAP · pandas · PySpark · kagglehub

## Future Extensions

Unity Catalog registry with a provisioned storage credential · Spark ML as
a dedicated comparison project · live scheduled job with failure alerting ·
feature store integration