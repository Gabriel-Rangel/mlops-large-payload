# Databricks notebook source
# MAGIC %md
# MAGIC # 08 — Um exemplo de 15 MiB via **curl** (payload **junto** no POST)
# MAGIC
# MAGIC Uma única carga (~15 MiB). As **15 variáveis e o JSON vão no mesmo body** nas duas chamadas:
# MAGIC
# MAGIC 1. **Model Serving** — `curl` POST com `var_01..var_15` + `payload` (~15 MiB) no `dataframe_records`.
# MAGIC 2. **App API** — o mesmo: `curl` POST com as variáveis + `payload`; a API devolve **202** + `request_id`.
# MAGIC
# MAGIC Tokens não são impressos. `$TOKEN` nos exemplos é placeholder.

# COMMAND ----------

dbutils.widgets.text("catalog", "gabriel_caballeria_demo_catalog")
dbutils.widgets.text("schema", "mlops_large_payload")
dbutils.widgets.text("endpoint_name", "large-payload-scorer-dev")
dbutils.widgets.text(
    "app_url",
    "https://large-payload-api-7474644345582688.aws.databricksapps.com",
)
dbutils.widgets.text("payload_mb", "15")
dbutils.widgets.text("timeout_s", "180")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
endpoint_name = dbutils.widgets.get("endpoint_name")
app_url = dbutils.widgets.get("app_url").rstrip("/")
payload_mb = float(dbutils.widgets.get("payload_mb"))
timeout_s = float(dbutils.widgets.get("timeout_s"))

print(f"catalog={catalog} schema={schema} endpoint={endpoint_name}")
print(f"app_url={app_url} payload_mb={payload_mb}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Preparação: JSON de 15 MiB, autenticação e helper do `curl`

# COMMAND ----------

import json
import os
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

import pandas as pd
from databricks.sdk import WorkspaceClient
from pyspark.sql import functions as F

REPO_CANDIDATES = [Path.cwd(), Path.cwd().parent, Path.cwd().parent.parent]
for root in REPO_CANDIDATES:
    if (root / "src" / "client" / "payload_ref.py").exists():
        sys.path.insert(0, str(root))
        break

from src.client.payload_ref import (  # noqa: E402
    build_payload_path,
    observability_row,
    write_payload_to_volume,
)
from src.model.features import SCALAR_FEATURES, sha256_bytes  # noqa: E402

ws = WorkspaceClient()
workspace_host = (ws.config.host or "").rstrip("/")
work_dir = Path(tempfile.mkdtemp(prefix="curl-15mb-"))
payload_path = work_dir / "payload_15mb.json"
serving_body_path = work_dir / "serving_body.json"
app_body_path = work_dir / "app_body.json"
obs_table = f"{catalog}.{schema}.inference_observability"


def make_large_json(target_mb: float) -> dict[str, Any]:
    target_bytes = int(target_mb * 1024 * 1024)
    envelope = (
        '{"items":[{"amount":1.23,"category":"PAD"}],'
        '"meta":{"generated_for":"curl-15mb","pad":""}}'
    )
    pad_len = max(0, target_bytes - len(envelope))
    return {
        "items": [{"amount": 1.23, "category": "PAD"}],
        "meta": {"generated_for": "curl-15mb", "pad": "x" * pad_len},
    }


def default_scalars() -> dict[str, float]:
    return {c: float(i) * 0.01 for i, c in enumerate(SCALAR_FEATURES, start=1)}


def serving_token() -> str:
    authenticate = getattr(ws.config, "authenticate", None)
    if callable(authenticate):
        raw = authenticate()
        if isinstance(raw, dict):
            auth = raw.get("Authorization") or raw.get("authorization")
            if isinstance(auth, str) and auth.lower().startswith("bearer "):
                return auth.split(" ", 1)[1]
    token = getattr(ws.config, "token", None)
    if token:
        return str(token)
    ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
    return ctx.apiToken().get()


def mint_app_token() -> str:
    client_id = dbutils.secrets.get(scope="mlops-large-payload", key="validator-client-id")
    client_secret = dbutils.secrets.get(scope="mlops-large-payload", key="validator-client-secret")
    result = run_curl(
        f"{workspace_host}/oidc/v1/token",
        method="POST",
        form_data=["grant_type=client_credentials", "scope=all-apis"],
        basic_user=client_id,
        basic_password=client_secret,
        timeout=60,
    )
    if result["http_code"] != 200:
        raise RuntimeError(f"OAuth M2M falhou HTTP {result['http_code']}: {result['body'][:400]}")
    token = (json.loads(result["body"]) or {}).get("access_token")
    if not token:
        raise RuntimeError("OAuth M2M não devolveu access_token")
    return str(token)


def run_curl(
    url: str,
    *,
    method: str = "GET",
    token: str | None = None,
    data_file: str | None = None,
    form_data: list[str] | None = None,
    basic_user: str | None = None,
    basic_password: str | None = None,
    timeout: float = 60,
) -> dict[str, Any]:
    """Chama o binário curl. Credenciais só em env; o comando logado omite o token."""
    env = os.environ.copy()
    write_out = (
        "\nCURL_HTTP_CODE:%{http_code}"
        "\nCURL_TTFB:%{time_starttransfer}"
        "\nCURL_TOTAL:%{time_total}"
    )
    cmd = ["curl", "-sS", "-w", write_out, "-X", method]
    if token:
        env["TOKEN"] = token
        cmd += ["-H", "Authorization: Bearer ${TOKEN}"]
    if basic_user is not None:
        env["CURL_USER"] = basic_user
        env["CURL_PASSWORD"] = basic_password or ""
        cmd += ["-u", "${CURL_USER}:${CURL_PASSWORD}"]
    if form_data:
        for item in form_data:
            cmd += ["-d", item]
    else:
        cmd += ["-H", "Content-Type: application/json"]
    if data_file:
        cmd += ["--data-binary", f"@{data_file}"]
    cmd.append(url)

    quoted = []
    for a in cmd:
        if a in ("Authorization: Bearer ${TOKEN}", "${CURL_USER}:${CURL_PASSWORD}"):
            quoted.append(f'"{a}"')
        else:
            quoted.append("'" + a.replace("'", "'\\''") + "'")
    print("curl:", " ".join(cmd).replace("${TOKEN}", "$TOKEN"))

    proc = subprocess.run(
        ["bash", "-lc", " ".join(quoted)],
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    stdout = proc.stdout or ""
    meta: dict[str, str] = {}
    body = stdout
    for key in ("CURL_HTTP_CODE", "CURL_TTFB", "CURL_TOTAL"):
        marker = f"{key}:"
        if marker in stdout:
            parts = stdout.rsplit(marker, 1)
            if key == "CURL_HTTP_CODE":
                body = parts[0]
            meta[key] = parts[1].splitlines()[0].strip()
    http_code = int(meta.get("CURL_HTTP_CODE") or 0)
    if proc.returncode != 0 and not http_code:
        raise RuntimeError(
            f"curl exit={proc.returncode} stderr={(proc.stderr or '')[:800]} body={body[:400]}"
        )
    return {
        "http_code": http_code,
        "ttfb_s": float(meta.get("CURL_TTFB") or 0.0),
        "total_s": float(meta.get("CURL_TOTAL") or 0.0),
        "body": body.strip(),
        "returncode": proc.returncode,
        "stderr": (proc.stderr or "")[:800],
    }


payload = make_large_json(payload_mb)
payload_path.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
payload_bytes = payload_path.stat().st_size
scalars = default_scalars()
print(f"work_dir={work_dir}")
print(f"payload_file={payload_path} bytes={payload_bytes} ({payload_bytes / (1024 * 1024):.2f} MiB)")

serving_tok = serving_token()
app_tok = mint_app_token()
print("Tokens prontos (não impressos). curl disponível:")
print(subprocess.check_output(["curl", "--version"], text=True).splitlines()[0])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1) Model Serving via curl — variáveis **e** payload no mesmo POST
# MAGIC
# MAGIC O body do `curl` tem ~15 MiB: as 15 variáveis **e** o JSON em `payload`. Não há gravação no Volume **antes** da chamada.
# MAGIC
# MAGIC ```bash
# MAGIC curl -sS -X POST "$HOST/serving-endpoints/$ENDPOINT/invocations" \
# MAGIC   -H "Authorization: Bearer $TOKEN" \
# MAGIC   -H "Content-Type: application/json" \
# MAGIC   --data-binary @serving_body.json
# MAGIC ```
# MAGIC
# MAGIC `serving_body.json` é `{"dataframe_records":[{ "var_01":..., "payload": { ... 15 MiB ... } }]}`.

# COMMAND ----------

request_id = str(uuid.uuid4())
payload_raw = payload_path.read_bytes()
serving_record = {
    **scalars,
    "payload": json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
    "payload_uri": "",
    "payload_hash": "",
    "request_id": request_id,
}
serving_payload = {"dataframe_records": [serving_record]}
serving_body_path.write_text(
    json.dumps(serving_payload, ensure_ascii=False, separators=(",", ":")),
    encoding="utf-8",
)
serving_body_bytes = serving_body_path.stat().st_size
print(
    f"request_id={request_id}\n"
    f"serving_curl_body_bytes={serving_body_bytes} "
    f"({serving_body_bytes / (1024 * 1024):.2f} MiB) "
    f"— payload vai no HTTP junto com as 15 variáveis"
)
if serving_body_bytes < 10 * 1024 * 1024:
    raise RuntimeError("Body do Serving ficou pequeno demais; o JSON de 15 MiB não foi embutido.")

serving_url = f"{workspace_host}/serving-endpoints/{endpoint_name}/invocations"
serving_curl = run_curl(
    serving_url,
    method="POST",
    token=serving_tok,
    data_file=str(serving_body_path),
    timeout=max(180.0, timeout_s),
)
accept_ms = serving_curl["ttfb_s"] * 1000.0
total_ms = serving_curl["total_s"] * 1000.0
print(
    f"curl Serving HTTP {serving_curl['http_code']} "
    f"ttfb={accept_ms:.0f} ms "
    f"total={total_ms:.0f} ms"
)
print("Resposta:", serving_curl["body"][:1500])

serving_json: dict[str, Any] | list[Any] | None
try:
    serving_json = json.loads(serving_curl["body"])
except json.JSONDecodeError:
    serving_json = None

prediction = probability = None
preds = serving_json
if isinstance(serving_json, dict):
    preds = serving_json.get("predictions") or serving_json
if isinstance(preds, list) and preds and isinstance(preds[0], dict):
    prediction = preds[0].get("prediction")
    probability = preds[0].get("probability")

# Persistência **depois** do curl, só para a tabela de observabilidade.
_, payload_uri = build_payload_path(catalog, schema, request_id=request_id)
meta = write_payload_to_volume(payload, payload_uri)
row = observability_row(
    request_id=request_id,
    endpoint_name=endpoint_name,
    model_version=None,
    payload_uri=meta["payload_uri"],
    payload_hash=sha256_bytes(payload_raw),
    payload_bytes=len(payload_raw),
    status="SUCCESS" if serving_curl["http_code"] == 200 and prediction is not None else "ERROR",
    latency_ms=total_ms,
    prediction=int(prediction) if prediction is not None else None,
    probability=float(probability) if probability is not None else None,
    error_message=None if serving_curl["http_code"] == 200 else serving_curl["body"][:1000],
    accept_latency_ms=accept_ms,
    total_latency_ms=total_ms,
)
row["env"] = catalog
row["scalars_json"] = json.dumps(scalars, separators=(",", ":"))
spark.createDataFrame(pd.DataFrame([row])).withColumn(
    "event_ts", F.to_timestamp("event_ts")
).write.format("delta").mode("append").option("mergeSchema", "true").saveAsTable(obs_table)

if serving_curl["http_code"] != 200 or prediction is None:
    raise RuntimeError(
        f"Serving via curl falhou: HTTP {serving_curl['http_code']} body={serving_curl['body'][:500]}"
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2) App API via curl — o mesmo contrato (variáveis + payload no POST)
# MAGIC
# MAGIC O `curl` envia os 15 MiB **junto** com as 15 variáveis. O App responde 202; o segundo `curl` espera o score.
# MAGIC
# MAGIC ```bash
# MAGIC curl -sS -X POST "$APP_URL/invocations" \
# MAGIC   -H "Authorization: Bearer $TOKEN" \
# MAGIC   -H "Content-Type: application/json" \
# MAGIC   --data-binary @app_body.json
# MAGIC
# MAGIC curl -sS "$APP_URL/invocations/$REQUEST_ID" \
# MAGIC   -H "Authorization: Bearer $TOKEN"
# MAGIC ```
# MAGIC
# MAGIC Auth do App: OAuth M2M (PAT cai na tela de login).

# COMMAND ----------

app_request = {**scalars, "payload": payload}
app_body_path.write_text(json.dumps(app_request, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
print(f"app_curl_body_bytes={app_body_path.stat().st_size} ({app_body_path.stat().st_size / (1024 * 1024):.2f} MiB)")

t_app = time.time()
app_post = run_curl(
    f"{app_url}/invocations",
    method="POST",
    token=app_tok,
    data_file=str(app_body_path),
    timeout=max(180.0, timeout_s),
)
accept_app_ms = app_post["ttfb_s"] * 1000.0
print(
    f"curl App POST HTTP {app_post['http_code']} "
    f"ttfb={accept_app_ms:.0f} ms total={app_post['total_s'] * 1000:.0f} ms"
)
print("Aceite 202:", app_post["body"][:1500])

if app_post["http_code"] != 202:
    raise RuntimeError(f"App POST via curl falhou: HTTP {app_post['http_code']} body={app_post['body'][:500]}")

accepted = json.loads(app_post["body"])
app_request_id = accepted["request_id"]
status_url = f"{app_url}/invocations/{app_request_id}"
deadline = time.time() + timeout_s
last = {}
while time.time() < deadline:
    app_get = run_curl(
        status_url,
        token=app_tok,
        timeout=30,
    )
    last = json.loads(app_get["body"]) if app_get["body"] else {}
    print(f"curl App GET HTTP {app_get['http_code']} status={last.get('status')}")
    if last.get("status") in ("done", "failed"):
        break
    time.sleep(2)

total_app_ms = (time.time() - t_app) * 1000.0
print("Resultado:", json.dumps({k: last.get(k) for k in (
    "request_id", "status", "prediction", "probability",
    "accept_latency_ms", "total_latency_ms", "payload_uri", "payload_bytes",
) if k in last}, indent=2))

if last.get("status") != "done" or last.get("prediction") is None:
    raise RuntimeError(f"App não concluiu via curl: {last}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Resumo dos dois `curl`

# COMMAND ----------

summary = pd.DataFrame(
    [
        {
            "via": "curl → Model Serving",
            "http": serving_curl["http_code"],
            "payload_on_wire_mib": round(serving_body_bytes / (1024 * 1024), 2),
            "payload_on_volume_mib": round(meta["payload_bytes"] / (1024 * 1024), 2),
            "accept_latency_ms": round(accept_ms, 1),
            "total_latency_ms": round(total_ms, 1),
            "prediction": prediction,
            "request_id": request_id,
        },
        {
            "via": "curl → App API",
            "http": app_post["http_code"],
            "payload_on_wire_mib": round(app_body_path.stat().st_size / (1024 * 1024), 2),
            "payload_on_volume_mib": round(float(last.get("payload_bytes") or 0) / (1024 * 1024), 2),
            "accept_latency_ms": round(float(last.get("accept_latency_ms") or accept_app_ms), 1),
            "total_latency_ms": round(float(last.get("total_latency_ms") or total_app_ms), 1),
            "prediction": last.get("prediction"),
            "request_id": app_request_id,
        },
    ]
)
display(summary)
print(summary.to_string(index=False))
print("OK: um payload de 15 MiB, duas chamadas curl.")
