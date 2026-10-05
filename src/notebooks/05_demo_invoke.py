# Databricks notebook source
# MAGIC %md
# MAGIC # 05 — Demo invoke (payload by reference)
# MAGIC
# MAGIC Simula o consumidor de produção:
# MAGIC 1. Gera `request_id`
# MAGIC 2. Grava JSON (tamanho configurável) no Volume UC
# MAGIC 3. Chama o endpoint **sem** o JSON no body
# MAGIC 4. Registra metadados em `inference_observability`
# MAGIC
# MAGIC > Default de demo: `payload_mb=15` (~15 MiB no Volume), abaixo do hard limit
# MAGIC > de 16 MB do Custom Model Serving e bem acima do limite de 1 MiB das inference tables.

# COMMAND ----------

dbutils.widgets.text("catalog", "ml_dev")
dbutils.widgets.text("schema", "mlops_large_payload")
dbutils.widgets.text("endpoint_name", "large-payload-scorer-dev")
dbutils.widgets.text("payload_mb", "15")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
endpoint_name = dbutils.widgets.get("endpoint_name")
payload_mb = float(dbutils.widgets.get("payload_mb"))

# COMMAND ----------

import json
import sys
import time
from pathlib import Path

import pandas as pd
from mlflow.deployments import get_deploy_client

REPO_CANDIDATES = [Path.cwd(), Path.cwd().parent, Path.cwd().parent.parent]
for root in REPO_CANDIDATES:
    if (root / "src" / "client" / "payload_ref.py").exists():
        sys.path.insert(0, str(root))
        break

from src.client.payload_ref import (  # noqa: E402
    build_payload_path,
    build_serving_record,
    observability_row,
    write_payload_to_volume,
)
from src.model.features import SCALAR_FEATURES  # noqa: E402

# COMMAND ----------

def make_large_json(target_mb: float) -> dict:
    """Build a JSON document of approximately target_mb (single padded field)."""
    target_bytes = int(target_mb * 1024 * 1024)
    envelope = '{"items":[{"amount":1.23,"category":"PAD"}],"meta":{"generated_for":"mlops-large-payload-demo","pad":""}}'
    pad_len = max(0, target_bytes - len(envelope))
    return {
        "items": [{"amount": 1.23, "category": "PAD"}],
        "meta": {
            "generated_for": "mlops-large-payload-demo",
            "pad": "x" * pad_len,
        },
    }


t_accept = time.time()
request_id, payload_uri = build_payload_path(catalog, schema)
payload = make_large_json(payload_mb)
meta = write_payload_to_volume(payload, payload_uri)
accept_latency_ms = (time.time() - t_accept) * 1000
print(
    f"Wrote {meta['payload_bytes'] / (1024 * 1024):.2f} MB → {meta['payload_uri']}\n"
    f"hash={meta['payload_hash']}\n"
    f"accept_latency_ms={accept_latency_ms:.1f} (Volume + request_id)"
)

scalars = {c: float(i) * 0.01 for i, c in enumerate(SCALAR_FEATURES, start=1)}
record = build_serving_record(
    scalars=scalars,
    payload_uri=meta["payload_uri"],
    payload_hash=meta["payload_hash"],
    request_id=request_id,
)

print("Serving record keys:", sorted(record.keys()))
print("Body size (approx):", len(json.dumps(record)), "bytes  ← deve ser << 1 MiB")

# COMMAND ----------

deploy = get_deploy_client("databricks")
t0 = time.time()
status = "SUCCESS"
prediction = None
probability = None
error_message = None
try:
    response = deploy.predict(endpoint=endpoint_name, inputs={"dataframe_records": [record]})
    latency_ms = (time.time() - t0) * 1000
    # Response shape varies slightly by runtime; normalize.
    preds = response.get("predictions") or response.get("dataframe_records") or response
    print("Raw response:", response)
    if isinstance(preds, list) and preds:
        first = preds[0]
        if isinstance(first, dict):
            prediction = first.get("prediction")
            probability = first.get("probability")
        else:
            prediction = first
    elif isinstance(preds, dict):
        prediction = preds.get("prediction")
        probability = preds.get("probability")
except Exception as ex:  # noqa: BLE001 — demo path logs failure into observability
    latency_ms = (time.time() - t0) * 1000
    status = "ERROR"
    error_message = str(ex)
    print("Invoke failed:", ex)

row = observability_row(
    request_id=request_id,
    endpoint_name=endpoint_name,
    model_version=None,
    payload_uri=meta["payload_uri"],
    payload_hash=meta["payload_hash"],
    payload_bytes=meta["payload_bytes"],
    status=status,
    latency_ms=accept_latency_ms + latency_ms,
    prediction=int(prediction) if prediction is not None else None,
    probability=float(probability) if probability is not None else None,
    error_message=error_message,
    accept_latency_ms=accept_latency_ms,
    total_latency_ms=accept_latency_ms + latency_ms,
)
row["env"] = catalog

from pyspark.sql import functions as F

obs_df = pd.DataFrame([row])
sdf = spark.createDataFrame(obs_df).withColumn("event_ts", F.to_timestamp("event_ts"))
sdf.write.format("delta").mode("append").option("mergeSchema", "true").saveAsTable(
    f"{catalog}.{schema}.inference_observability"
)

display(
    spark.table(f"{catalog}.{schema}.inference_observability")
    .orderBy(F.desc("event_ts"))
    .limit(5)
)
print(
    f"Logged observability row for request_id={request_id} "
    f"status={status} accept_latency_ms={accept_latency_ms:.1f} "
    f"total_latency_ms={accept_latency_ms + latency_ms:.1f} "
    f"(serving_invoke_ms={latency_ms:.1f})"
)
