# Databricks notebook source
# MAGIC %md
# MAGIC # 06 — Inferência assíncrona / batch (alternativa)
# MAGIC
# MAGIC Use este fluxo quando:
# MAGIC - o JSON de ~16 MB **não** precisa de resposta síncrona em poucos segundos; ou
# MAGIC - o Model Serving online fica no limite (envelope + 16 MB ≈ hard limit do endpoint).
# MAGIC
# MAGIC Padrão:
# MAGIC 1. Consumidor grava payload no Volume + enfileira linha em `async_inference_queue`
# MAGIC 2. Este Job processa `PENDING`, carrega o modelo `@Champion`, scoreia, atualiza a fila
# MAGIC 3. Observabilidade espelha o resultado

# COMMAND ----------

dbutils.widgets.text("catalog", "ml_dev")
dbutils.widgets.text("schema", "mlops_large_payload")
dbutils.widgets.text("model_name", "large_payload_scorer")
dbutils.widgets.text("batch_size", "50")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
model_name = dbutils.widgets.get("model_name")
batch_size = int(dbutils.widgets.get("batch_size"))
full_model_name = f"{catalog}.{schema}.{model_name}"
queue = f"{catalog}.{schema}.async_inference_queue"
obs = f"{catalog}.{schema}.inference_observability"

# COMMAND ----------

import json
import sys
import time
from pathlib import Path

import mlflow
import pandas as pd
from pyspark.sql import functions as F

REPO_CANDIDATES = [Path.cwd(), Path.cwd().parent, Path.cwd().parent.parent]
for root in REPO_CANDIDATES:
    if (root / "src" / "model" / "features.py").exists():
        sys.path.insert(0, str(root))
        break

from src.model.features import SCALAR_FEATURES  # noqa: E402

mlflow.set_registry_uri("databricks-uc")
model = mlflow.pyfunc.load_model(f"models:/{full_model_name}@Champion")

pending = (
    spark.table(queue)
    .filter("status = 'PENDING'")
    .orderBy("created_at")
    .limit(batch_size)
    .toPandas()
)

if pending.empty:
    print("Nenhum request PENDING")
    dbutils.notebook.exit("noop")

print(f"Processing {len(pending)} requests")

records = []
for _, row in pending.iterrows():
    scalars = json.loads(row["scalars_json"])
    rec = {c: float(scalars.get(c, 0.0)) for c in SCALAR_FEATURES}
    rec["payload_uri"] = row["payload_uri"]
    rec["payload_hash"] = row.get("payload_hash") or ""
    rec["request_id"] = row["request_id"]
    rec["payload"] = ""
    records.append(rec)

t0 = time.time()
preds = model.predict(pd.DataFrame(records))
latency_ms = (time.time() - t0) * 1000 / max(len(records), 1)

updates = pending.copy()
updates["status"] = "DONE"
updates["prediction"] = preds["prediction"].astype(int).values
updates["probability"] = preds["probability"].astype(float).values
updates["error_message"] = None
updates["completed_at"] = pd.Timestamp.utcnow()

# Merge back via temp view + MERGE
spark.createDataFrame(updates).createOrReplaceTempView("batch_updates")
spark.sql(
    f"""
    MERGE INTO {queue} t
    USING batch_updates s
    ON t.request_id = s.request_id
    WHEN MATCHED THEN UPDATE SET
      status = s.status,
      prediction = s.prediction,
      probability = s.probability,
      error_message = s.error_message,
      completed_at = s.completed_at
    """
)

obs_rows = []
for _, row in updates.iterrows():
    obs_rows.append(
        {
            "request_id": row["request_id"],
            "event_ts": pd.Timestamp.utcnow().isoformat(),
            "endpoint_name": "batch-job",
            "model_version": None,
            "model_alias": "Champion",
            "status": "SUCCESS",
            "latency_ms": latency_ms,
            "payload_uri": row["payload_uri"],
            "payload_hash": row.get("payload_hash"),
            "payload_bytes": None,
            "prediction": int(row["prediction"]),
            "probability": float(row["probability"]),
            "error_message": None,
            "env": catalog,
        }
    )

spark.createDataFrame(pd.DataFrame(obs_rows)).withColumn(
    "event_ts", F.to_timestamp("event_ts")
).write.format("delta").mode("append").option("mergeSchema", "true").saveAsTable(obs)

print(f"Done. avg_latency_ms≈{latency_ms:.1f}")
