# Databricks notebook source
# MAGIC %md
# MAGIC # 01 — Setup Unity Catalog (DDL + observabilidade)
# MAGIC
# MAGIC Cria:
# MAGIC - Volume de payloads (se o bundle ainda não criou)
# MAGIC - Tabela Delta de **observabilidade** (metadados + ponteiro; sem JSON completo)
# MAGIC - Tabela de **requests assíncronos** (opcional, para fluxo batch)
# MAGIC
# MAGIC > Inference tables do AI Gateway **não** são o repositório do payload de ~16 MB
# MAGIC > (limite ~1 MiB). Esta tabela customizada + Volume é o padrão recomendado.

# COMMAND ----------

dbutils.widgets.text("catalog", "ml_dev")
dbutils.widgets.text("schema", "mlops_large_payload")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")
spark.sql(
    f"""
    CREATE VOLUME IF NOT EXISTS {catalog}.{schema}.request_payloads
    COMMENT 'JSON payloads grandes referenciados por payload_uri'
    """
)
spark.sql(
    f"""
    CREATE VOLUME IF NOT EXISTS {catalog}.{schema}.model_artifacts_aux
    COMMENT 'Artefatos auxiliares do projeto'
    """
)

print(f"Ready: {catalog}.{schema}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Tabela de observabilidade (metadata-only)
# MAGIC
# MAGIC Guarda `request_id`, caminhos, hashes, latência, status e resultado resumido.
# MAGIC O JSON completo fica **somente** no Volume.

# COMMAND ----------

spark.sql(
    f"""
    CREATE TABLE IF NOT EXISTS {catalog}.{schema}.inference_observability (
      request_id       STRING       COMMENT 'Correlation id gerado pelo consumidor',
      event_ts         TIMESTAMP    COMMENT 'UTC do registro',
      endpoint_name    STRING,
      model_version    STRING,
      model_alias      STRING       COMMENT 'Champion / Challenger / null',
      status           STRING       COMMENT 'SUCCESS | ERROR | TIMEOUT',
      latency_ms       DOUBLE       COMMENT 'Alias de total_latency_ms (legado)',
      accept_latency_ms DOUBLE      COMMENT 'Envio dos dados + geração do request_id (202 no App; Volume+id no Serving)',
      total_latency_ms  DOUBLE      COMMENT 'Até o resultado (score). No App: accept + trabalho em background',
      payload_uri      STRING       COMMENT 'Caminho absoluto no Volume UC',
      payload_hash     STRING,
      payload_bytes    BIGINT,
      prediction       INT,
      probability      DOUBLE,
      error_message    STRING,
      env              STRING,
      ingested_at      TIMESTAMP,
      scalars_json     STRING       COMMENT 'JSON das 15 variáveis escalares; o payload completo fica só no Volume'
    )
    USING DELTA
    COMMENT 'Observabilidade customizada para payloads > 1 MiB'
    """
)
for col_sql in (
    "scalars_json STRING COMMENT 'JSON das 15 variáveis escalares; o payload completo fica só no Volume'",
    "accept_latency_ms DOUBLE COMMENT 'Envio dos dados + geração do request_id'",
    "total_latency_ms DOUBLE COMMENT 'Latência até o resultado (score)'",
):
    try:
        spark.sql(
            f"ALTER TABLE {catalog}.{schema}.inference_observability ADD COLUMNS ({col_sql})"
        )
    except Exception as exc:  # noqa: BLE001 — column may already exist
        print(f"ADD COLUMNS skipped ({col_sql.split()[0]}):", exc)

spark.sql(
    f"""
    CREATE TABLE IF NOT EXISTS {catalog}.{schema}.async_inference_queue (
      request_id       STRING,
      created_at       TIMESTAMP,
      payload_uri      STRING,
      payload_hash     STRING,
      scalars_json     STRING       COMMENT 'JSON das 15 variáveis escalares',
      status           STRING       COMMENT 'PENDING | RUNNING | DONE | FAILED',
      prediction       INT,
      probability      DOUBLE,
      error_message    STRING,
      completed_at     TIMESTAMP
    )
    USING DELTA
    COMMENT 'Fila opcional para inferência assíncrona/batch quando sync online não é necessário'
    """
)

display(
    spark.sql(
        f"""
        SHOW TABLES IN {catalog}.{schema}
        """
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Política de retenção (sugestão)
# MAGIC
# MAGIC - Metadados de observabilidade: 90–180 dias (ajuste por compliance).
# MAGIC - Payloads no Volume: retenção mais curta (ex.: 30 dias) ou só sob retenção legal.
# MAGIC - Em muitos casos **não** é necessário guardar o JSON completo — mascarar, amostrar ou guardar só hash.

# COMMAND ----------

# Serving identity precisa atravessar catálogo/schema e ler o Volume.
# Em FEVM/demo o criador do endpoint é o mesmo user do Job; em prod use o SP do serving.
_user = (
    dbutils.notebook.entry_point.getDbutils()
    .notebook()
    .getContext()
    .userName()
    .get()
)
volume_name = f"{catalog}.{schema}.request_payloads"
for stmt in [
    f"GRANT USE CATALOG ON CATALOG {catalog} TO `{_user}`",
    f"GRANT USE SCHEMA ON SCHEMA {catalog}.{schema} TO `{_user}`",
    f"GRANT READ VOLUME ON VOLUME {volume_name} TO `{_user}`",
]:
    try:
        spark.sql(stmt)
        print("OK:", stmt)
    except Exception as exc:  # noqa: BLE001
        print("SKIP/FAIL:", stmt, "→", exc)
