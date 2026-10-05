# Databricks notebook source
# MAGIC %md
# MAGIC # 03 — Evaluate + Promote (gates)
# MAGIC
# MAGIC Valida a versão **Challenger** antes de virar **Champion**.
# MAGIC
# MAGIC Gates demonstrados:
# MAGIC 1. Métrica mínima (`min_roc_auc`)
# MAGIC 2. Signature / schema presente
# MAGIC 3. Smoke test de `predict` com payload de exemplo no Volume
# MAGIC
# MAGIC Para promoção **entre catálogos** (dev → staging → prod), use `mlflow.copy_model_version`
# MAGIC ou um Job dedicado com service principal de promoção (sem usuários individuais em prod).

# COMMAND ----------

dbutils.widgets.text("catalog", "ml_dev")
dbutils.widgets.text("schema", "mlops_large_payload")
dbutils.widgets.text("model_name", "large_payload_scorer")
dbutils.widgets.text("min_roc_auc", "0.70")
dbutils.widgets.dropdown("promote_to_champion", "false", ["false", "true"])

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
model_name = dbutils.widgets.get("model_name")
min_roc_auc = float(dbutils.widgets.get("min_roc_auc"))
promote = dbutils.widgets.get("promote_to_champion").lower() == "true"
full_model_name = f"{catalog}.{schema}.{model_name}"

# COMMAND ----------

import json
import sys
from pathlib import Path

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

REPO_CANDIDATES = [Path.cwd(), Path.cwd().parent, Path.cwd().parent.parent]
for root in REPO_CANDIDATES:
    if (root / "src" / "model" / "features.py").exists():
        sys.path.insert(0, str(root))
        break

from src.model.features import SCALAR_FEATURES  # noqa: E402

mlflow.set_registry_uri("databricks-uc")
client = MlflowClient(registry_uri="databricks-uc")

challenger = client.get_model_version_by_alias(full_model_name, "Challenger")
version = challenger.version
print(f"Evaluating {full_model_name} @Challenger → v{version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Gate 1 — métrica

# COMMAND ----------

mv = client.get_model_version(full_model_name, version)
tag_map = dict(mv.tags or {})
roc_auc = float(tag_map.get("roc_auc", "0"))
print("roc_auc tag:", roc_auc, "| threshold:", min_roc_auc)
assert roc_auc >= min_roc_auc, f"GATE FAIL: roc_auc {roc_auc} < {min_roc_auc}"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Gate 2 — signature

# COMMAND ----------

model_uri = f"models:/{full_model_name}/{version}"
model_info = mlflow.models.get_model_info(model_uri)
assert model_info.signature is not None, "GATE FAIL: model signature missing"
print("Signature OK:", model_info.signature)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Gate 3 — smoke test (Volume + PyFunc)

# COMMAND ----------

example_dir = Path(f"/Volumes/{catalog}/{schema}/request_payloads/examples")
example_dir.mkdir(parents=True, exist_ok=True)
example_uri = example_dir / "smoke_test.json"
example_payload = {
    "items": [
        {"amount": 10.5, "category": "A"},
        {"amount": 3.0, "category": "B"},
        {"amount": 7.2, "category": "A"},
    ],
    "meta": {"source": "evaluate_promote_smoke"},
}
example_uri.write_text(json.dumps(example_payload), encoding="utf-8")

record = {c: 0.1 for c in SCALAR_FEATURES}
record.update(
    {
        "payload_uri": str(example_uri),
        "payload_hash": "",
        "request_id": "smoke-test",
        "payload": "",
    }
)

pyfunc_model = mlflow.pyfunc.load_model(model_uri)
out = pyfunc_model.predict(pd.DataFrame([record]))
print(out)
assert "prediction" in out.columns and "probability" in out.columns
print("Smoke test OK")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Promoção Champion (opcional)

# COMMAND ----------

if promote:
    client.set_registered_model_alias(full_model_name, "Champion", version)
    client.set_model_version_tag(full_model_name, version, "validation_status", "valid")
    print(f"Promoted: alias Champion → v{version}")
else:
    client.set_model_version_tag(full_model_name, version, "validation_status", "valid")
    print("Gates passed. promote_to_champion=false → Champion não alterado.")

dbutils.jobs.taskValues.set(key="promoted_version", value=str(version))
dbutils.jobs.taskValues.set(key="promoted", value=str(promote).lower())
