# Databricks notebook source
# MAGIC %md
# MAGIC # 02 — Train + Register (Unity Catalog)
# MAGIC
# MAGIC Treina um classificador demo, empacota como **PyFunc** que lê JSON via `payload_uri`,
# MAGIC registra no UC e aponta o alias **Challenger**.
# MAGIC
# MAGIC Em produção:
# MAGIC - Substitua o gerador sintético pelos datasets reais (registre-os com `mlflow.log_input`).
# MAGIC - Rode via Job do bundle (`train_register`), não como notebook ad-hoc em prod.

# COMMAND ----------

dbutils.widgets.text("catalog", "ml_dev")
dbutils.widgets.text("schema", "mlops_large_payload")
dbutils.widgets.text("model_name", "large_payload_scorer")
dbutils.widgets.text("experiment_path", "/Shared/mlops-large-payload/dev")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
model_name = dbutils.widgets.get("model_name")
experiment_path = dbutils.widgets.get("experiment_path")
full_model_name = f"{catalog}.{schema}.{model_name}"

# COMMAND ----------

import sys
from pathlib import Path

import joblib
import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# Bundle source is synced next to notebooks when deployed via DABs / repo.
REPO_CANDIDATES = [
    Path.cwd(),
    Path.cwd().parent,
    Path.cwd().parent.parent,
    Path("/Workspace") / "Users" / dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get() / "mlops-large-payload",
]
root = Path.cwd()
for candidate in REPO_CANDIDATES:
    if (candidate / "src" / "model" / "features.py").exists():
        root = candidate
        break

# Import as top-level `model.*` so cloudpickle + Model Serving can reload
# the class from code_paths (MLflow puts this directory on sys.path).
sys.path.insert(0, str(root / "src"))
from model.features import make_synthetic_training_frame  # noqa: E402
from model.pyfunc_model import LargePayloadScorer, model_signature  # noqa: E402

mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(experiment_path)

print("Model:", full_model_name)
print("Experiment:", experiment_path)

# COMMAND ----------

X, y = make_synthetic_training_frame(n=600, seed=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

with mlflow.start_run(run_name="train-large-payload-scorer") as run:
    clf = LogisticRegression(max_iter=500)
    clf.fit(X_train, y_train)

    proba = clf.predict_proba(X_test)[:, 1]
    auc = float(roc_auc_score(y_test, proba))
    mlflow.log_metric("roc_auc", auc)
    mlflow.log_param("algorithm", "LogisticRegression")
    mlflow.log_param("n_train", len(X_train))
    mlflow.log_param("n_test", len(X_test))
    mlflow.set_tags(
        {
            "owner": "ml-platform",
            "domain": "demo-large-payload",
            "criticality": "medium",
            "payload_pattern": "volume_reference",
            "validation_status": "pending",
        }
    )

    # Persist sklearn artifact, then wrap with PyFunc that understands payload_uri.
    artifact_dir = Path("/tmp/mlops_lp_model")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifact_dir / "model.joblib"
    joblib.dump(clf, model_path)

    mlflow.pyfunc.log_model(
        name="model",
        python_model=LargePayloadScorer(),
        artifacts={"sklearn_model": str(model_path)},
        signature=model_signature(),
        registered_model_name=full_model_name,
        code_paths=[str(root / "src" / "model")],
        pip_requirements=[
            "mlflow>=2.14.0",
            "scikit-learn>=1.4.0",
            "pandas>=2.0.0",
            "numpy>=1.26.0",
            "joblib>=1.3.0",
            "cloudpickle>=3.0.0",
            "databricks-sdk>=0.36.0",
        ],
    )

    client = MlflowClient(registry_uri="databricks-uc")
    versions = client.search_model_versions(f"name='{full_model_name}'")
    latest = max(versions, key=lambda v: int(v.version))
    client.set_registered_model_alias(full_model_name, "Challenger", latest.version)
    client.set_model_version_tag(full_model_name, latest.version, "roc_auc", f"{auc:.4f}")
    client.set_model_version_tag(full_model_name, latest.version, "code_ref", run.info.run_id)

    print(f"Registered {full_model_name} v{latest.version}  roc_auc={auc:.4f}")
    print("Alias Challenger →", latest.version)
    dbutils.jobs.taskValues.set(key="model_version", value=str(latest.version))
    dbutils.jobs.taskValues.set(key="roc_auc", value=f"{auc:.4f}")
