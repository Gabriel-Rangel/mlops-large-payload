# Databricks notebook source
# MAGIC %md
# MAGIC # 04 — Deploy / update Model Serving endpoint
# MAGIC
# MAGIC Aponta o endpoint para a versão do alias **Champion**.
# MAGIC
# MAGIC Importante (payload grande):
# MAGIC - O endpoint recebe só `payload_uri` + scalars (~KB), não o JSON de 16 MB.
# MAGIC - Se o PyFunc ler o Volume em runtime, o **identity do serving** precisa de `READ VOLUME`.
# MAGIC - Alternativa mais comum em produção: extrair features do JSON **antes** do endpoint
# MAGIC   (App / Job / API gateway) e servir um modelo tabular clássico — veja `docs/ARCHITECTURE.md`.

# COMMAND ----------

dbutils.widgets.text("catalog", "ml_dev")
dbutils.widgets.text("schema", "mlops_large_payload")
dbutils.widgets.text("model_name", "large_payload_scorer")
dbutils.widgets.text("endpoint_name", "large-payload-scorer-dev")
dbutils.widgets.text("workload_size", "Small")
dbutils.widgets.dropdown("scale_to_zero", "true", ["true", "false"])
dbutils.widgets.text(
    "databricks_host",
    "https://fevm-gabriel-caballeria-demo.cloud.databricks.com",
)
dbutils.widgets.text(
    "volume_token_secret",
    "{{secrets/mlops-large-payload/databricks-token}}",
)

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
model_name = dbutils.widgets.get("model_name")
endpoint_name = dbutils.widgets.get("endpoint_name")
workload_size = dbutils.widgets.get("workload_size")
scale_to_zero = dbutils.widgets.get("scale_to_zero").lower() == "true"
databricks_host = dbutils.widgets.get("databricks_host")
# Jobs interpolate {{secrets/...}} in task parameters. Model Serving needs the
# literal reference on environment_vars so it injects the secret at container start.
volume_token_secret = "{{secrets/mlops-large-payload/databricks-token}}"
full_model_name = f"{catalog}.{schema}.{model_name}"

# COMMAND ----------

import time

from mlflow.deployments import get_deploy_client
from mlflow.tracking import MlflowClient

registry = MlflowClient(registry_uri="databricks-uc")
deploy = get_deploy_client("databricks")

champion = registry.get_model_version_by_alias(full_model_name, "Champion")
version = champion.version
entity_route_name = f"{model_name}-{version}"

print(f"Deploying {full_model_name} v{version} (@Champion) → endpoint `{endpoint_name}`")

config = {
    "served_entities": [
        {
            "entity_name": full_model_name,
            "entity_version": version,
            "workload_size": workload_size,
            "scale_to_zero_enabled": scale_to_zero,
            "name": entity_route_name,
            # Serving does not FUSE-mount /Volumes. Files API needs workspace creds.
            "environment_vars": {
                "DATABRICKS_HOST": databricks_host,
                "DATABRICKS_TOKEN": volume_token_secret,
            },
        }
    ],
    "traffic_config": {
        "routes": [
            {
                "served_model_name": entity_route_name,
                "traffic_percentage": 100,
            }
        ]
    },
}

existing = {e["name"] for e in deploy.list_endpoints()}
if endpoint_name in existing:
    deploy.update_endpoint(endpoint=endpoint_name, config=config)
    print("update_endpoint submitted")
else:
    deploy.create_endpoint(name=endpoint_name, config=config)
    print("create_endpoint submitted")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Poll readiness
# MAGIC
# MAGIC Em version swap, `state.ready` pode permanecer `READY` enquanto
# MAGIC `state.config_update` ainda é `IN_PROGRESS`. Espere os dois.

# COMMAND ----------

deadline = time.time() + 30 * 60
while time.time() < deadline:
    ep = deploy.get_endpoint(endpoint_name)
    ready = (ep.get("state") or {}).get("ready")
    cfg = (ep.get("state") or {}).get("config_update")
    print(f"ready={ready} config_update={cfg}")
    if cfg == "UPDATE_FAILED":
        raise RuntimeError(
            f"Endpoint {endpoint_name} config_update=UPDATE_FAILED. "
            "Check serving logs (often a missing Python module in the model artifact)."
        )
    if ready == "READY" and cfg in (None, "NOT_UPDATING"):
        break
    time.sleep(15)
else:
    raise TimeoutError(f"Endpoint {endpoint_name} não ficou READY a tempo")

print("Endpoint READY:", endpoint_name)
dbutils.jobs.taskValues.set(key="endpoint_name", value=endpoint_name)
dbutils.jobs.taskValues.set(key="served_version", value=str(version))
