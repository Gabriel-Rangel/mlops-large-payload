# Databricks notebook source
# MAGIC %md
# MAGIC # 07 — Validação ponta a ponta das duas abordagens de payload grande
# MAGIC
# MAGIC Este notebook gera um relatório **pass/fail** para:
# MAGIC
# MAGIC - **Abordagem A** — Model Serving com payload por referência (Volume UC + body pequeno).
# MAGIC - **Abordagem B** — API assíncrona em Databricks App (JSON no POST; score local; 202 + poll).
# MAGIC
# MAGIC Comentários e markdown em português. Identificadores de código em inglês.

# COMMAND ----------

dbutils.widgets.text("catalog", "gabriel_caballeria_demo_catalog")
dbutils.widgets.text("schema", "mlops_large_payload")
dbutils.widgets.text("endpoint_name", "large-payload-scorer-dev")
dbutils.widgets.text(
    "app_url",
    "https://large-payload-api-7474644345582688.aws.databricksapps.com",
)
dbutils.widgets.text("payload_sizes_mb", "0.01,2,15")
dbutils.widgets.text("timeout_s", "600")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
endpoint_name = dbutils.widgets.get("endpoint_name")
app_url = dbutils.widgets.get("app_url").rstrip("/")
payload_sizes_mb = [
    float(x.strip()) for x in dbutils.widgets.get("payload_sizes_mb").split(",") if x.strip()
]
timeout_s = float(dbutils.widgets.get("timeout_s"))
obs_table = f"{catalog}.{schema}.inference_observability"

for col_sql in (
    "accept_latency_ms DOUBLE COMMENT 'Envio dos dados + geração do request_id'",
    "total_latency_ms DOUBLE COMMENT 'Latência até o resultado (score)'",
):
    try:
        spark.sql(f"ALTER TABLE {obs_table} ADD COLUMNS ({col_sql})")
        print("Added column:", col_sql.split()[0])
    except Exception as exc:  # noqa: BLE001
        print(f"ADD COLUMNS skipped ({col_sql.split()[0]}):", type(exc).__name__)

print(
    f"catalog={catalog} schema={schema} endpoint={endpoint_name}\n"
    f"app_url={app_url} sizes={payload_sizes_mb} timeout_s={timeout_s}"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Preparação: helpers, autenticação e gerador de JSON sintético

# COMMAND ----------

import json
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from databricks.sdk import WorkspaceClient
from pyspark.sql import functions as F

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

ws = WorkspaceClient()
workspace_host = (ws.config.host or "").rstrip("/")
results: list[dict[str, Any]] = []
latency_rows: list[dict[str, Any]] = []


def record_check(
    approach: str,
    size_mb: float | None,
    check: str,
    passed: bool,
    detail: str,
    latency: float | None = None,
    skipped: bool = False,
) -> None:
    results.append(
        {
            "approach": approach,
            "size_mb": size_mb,
            "check": check,
            "passed": "SKIP" if skipped else ("PASS" if passed else "FAIL"),
            "detail": detail[:2000],
            "latency_ms": None if latency is None else round(latency, 1),
        }
    )


def obs_float(row: pd.Series, col: str) -> float | None:
    if col not in row.index:
        return None
    val = row[col]
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def latency_split_ok(accept_ms: float | None, total_ms: float | None) -> tuple[bool, str]:
    if accept_ms is None or total_ms is None:
        return False, f"accept={accept_ms} total={total_ms} (esperado ambos preenchidos)"
    if accept_ms < 0 or total_ms < 0:
        return False, f"valores negativos accept={accept_ms:.1f} total={total_ms:.1f}"
    if total_ms + 1.0 < accept_ms:
        return False, f"total ({total_ms:.1f}) < accept ({accept_ms:.1f})"
    return True, f"accept_latency_ms={accept_ms:.1f} total_latency_ms={total_ms:.1f}"


def make_large_json(target_mb: float) -> dict[str, Any]:
    """JSON sintético ~N MiB: um campo `pad` + array `items` pequeno (features derivadas)."""
    target_bytes = int(target_mb * 1024 * 1024)
    envelope = (
        '{"items":[{"amount":1.23,"category":"PAD"}],'
        '"meta":{"generated_for":"validate-approaches","pad":""}}'
    )
    pad_len = max(0, target_bytes - len(envelope))
    return {
        "items": [{"amount": 1.23, "category": "PAD"}],
        "meta": {"generated_for": "validate-approaches", "pad": "x" * pad_len},
    }


def default_scalars() -> dict[str, float]:
    return {c: float(i) * 0.01 for i, c in enumerate(SCALAR_FEATURES, start=1)}


def parse_prediction(payload: Any) -> tuple[Any, Any]:
    preds = payload
    if isinstance(payload, dict):
        preds = payload.get("predictions") or payload.get("dataframe_records") or payload
    prediction = None
    probability = None
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
    return prediction, probability


def looks_like_login_html(resp: requests.Response) -> bool:
    ctype = (resp.headers.get("content-type") or "").lower()
    text = (resp.text or "")[:2000]
    if "text/html" in ctype:
        return True
    lowered = text.lower()
    return "databricks - sign in" in lowered or "<html" in lowered and "sign in" in lowered


def app_json(resp: requests.Response) -> dict[str, Any] | None:
    if looks_like_login_html(resp):
        return None
    try:
        data = resp.json()
    except Exception:  # noqa: BLE001
        return None
    return data if isinstance(data, dict) else None


def headers_from_token(token: str | None) -> dict[str, str] | None:
    if not token:
        return None
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


def workspace_auth_headers() -> dict[str, str]:
    """Monta headers de auth sem imprimir o token."""
    headers: dict[str, str] = {"Content-Type": "application/json"}
    authenticate = getattr(ws.config, "authenticate", None)
    if callable(authenticate):
        raw = authenticate()
        if isinstance(raw, dict):
            headers.update({str(k): str(v) for k, v in raw.items()})
    if "Authorization" not in headers:
        oauth_token = getattr(ws.config, "oauth_token", None)
        token = None
        if callable(oauth_token):
            tok = oauth_token()
            token = getattr(tok, "access_token", None) or (tok if isinstance(tok, str) else None)
        if not token:
            token = getattr(ws.config, "token", None)
        if token:
            headers["Authorization"] = f"Bearer {token}"
    return headers


def oauth_only_headers() -> dict[str, str] | None:
    oauth_token = getattr(ws.config, "oauth_token", None)
    if not callable(oauth_token):
        return None
    try:
        tok = oauth_token()
    except Exception as exc:  # noqa: BLE001
        print("oauth_token() falhou:", type(exc).__name__)
        return None
    token = getattr(tok, "access_token", None) or (tok if isinstance(tok, str) else None)
    return headers_from_token(token)


def notebook_context_headers() -> dict[str, str] | None:
    try:
        ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
        token = ctx.apiToken().get()
    except Exception as exc:  # noqa: BLE001
        print("Token do contexto do notebook indisponível:", type(exc).__name__)
        return None
    return headers_from_token(token)


def secret_pat_headers() -> dict[str, str] | None:
    """Último recurso: PAT do secret scope (Apps costumam rejeitar PAT)."""
    try:
        token = dbutils.secrets.get(scope="mlops-large-payload", key="databricks-token")
    except Exception as exc:  # noqa: BLE001
        print("Secret PAT indisponível:", type(exc).__name__)
        return None
    return headers_from_token(token)


def app_base_candidates() -> list[str]:
    bases = [app_url]
    ws_app = f"{workspace_host}/apps/large-payload-api"
    if ws_app not in bases:
        bases.append(ws_app)
    return bases


def probe_app_health(base: str, headers: dict[str, str]) -> tuple[int | None, str, bool]:
    """Retorna (status, detalhe_sem_segredo, json_ok)."""
    try:
        resp = requests.get(
            f"{base}/health",
            headers=headers,
            timeout=30,
            allow_redirects=False,
        )
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}", False
    if resp.status_code in (301, 302, 303, 307, 308):
        loc = resp.headers.get("Location", "")
        return resp.status_code, f"redirect Location={loc[:120]}", False
    if looks_like_login_html(resp):
        return resp.status_code, "HTML Databricks Sign In (proxy recusou o token)", False
    payload = app_json(resp)
    ctype = (resp.headers.get("content-type") or "")[:80]
    if payload is not None:
        return resp.status_code, f"JSON content-type={ctype} keys={sorted(payload.keys())}", True
    return resp.status_code, f"não-JSON content-type={ctype} body={(resp.text or '')[:180]}", False


def mint_m2m_headers() -> tuple[dict[str, str] | None, str, str]:
    """OAuth M2M (client_credentials). SKIP só se os secrets não existirem."""
    try:
        client_id = dbutils.secrets.get(scope="mlops-large-payload", key="validator-client-id")
        client_secret = dbutils.secrets.get(scope="mlops-large-payload", key="validator-client-secret")
    except Exception as exc:  # noqa: BLE001
        return None, f"secrets validator-client-id/validator-client-secret ausentes ({type(exc).__name__})", "skip"
    if not client_id or not client_secret:
        return None, "secrets validator-client-id/validator-client-secret vazios", "skip"
    try:
        resp = requests.post(
            f"{workspace_host}/oidc/v1/token",
            auth=(client_id, client_secret),
            data={"grant_type": "client_credentials", "scope": "all-apis"},
            timeout=30,
        )
    except Exception as exc:  # noqa: BLE001
        return None, f"POST /oidc/v1/token falhou: {type(exc).__name__}: {exc}", "fail"
    if resp.status_code != 200:
        return (
            None,
            f"POST /oidc/v1/token HTTP {resp.status_code} body={(resp.text or '')[:240]}",
            "fail",
        )
    try:
        token = (resp.json() or {}).get("access_token")
    except Exception as exc:  # noqa: BLE001
        return None, f"oidc/v1/token JSON inválido: {type(exc).__name__}", "fail"
    headers = headers_from_token(token)
    if not headers:
        return None, "oidc/v1/token sem access_token", "fail"
    return headers, "oauth m2m client_credentials (SP mlops-large-payload-validator)", "ok"


def resolve_app_client() -> tuple[str | None, dict[str, str] | None, str, str]:
    """Retorna (base_url, headers, detalhe, mode) com mode em ok|skip|fail."""
    headers, mint_detail, mode = mint_m2m_headers()
    print("M2M mint:", mint_detail if mode != "ok" else "ok (token não impresso)")
    if mode != "ok" or not headers:
        return None, None, mint_detail, mode

    last_detail = mint_detail
    for base in app_base_candidates():
        status, detail, json_ok = probe_app_health(base, headers)
        print(f"App /health via M2M @ {base}: HTTP {status} {detail}")
        last_detail = f"{mint_detail} @ {base}: HTTP {status} {detail}"
        if json_ok:
            return base, headers, last_detail, "ok"
    return None, None, last_detail, "fail"


# COMMAND ----------

# MAGIC %md
# MAGIC ## Abordagem A — Serving com payload por referência
# MAGIC
# MAGIC Fluxo: gravar JSON no Volume → POST pequeno (`dataframe_records`) → HTTP 200 + predição
# MAGIC → linha em `inference_observability`. Body do request deve ser **muito menor** que 1 MiB.
# MAGIC Teste negativo: `payload_hash` errado deve retornar **400**.
# MAGIC
# MAGIC **Latência (Serving é síncrono; o “id assíncrono” aqui é o `request_id` + JSON no Volume):**
# MAGIC
# MAGIC | coluna | significado nesta abordagem |
# MAGIC |---|---|
# MAGIC | `accept_latency_ms` | gravar o JSON no Volume + gerar `request_id` |
# MAGIC | `total_latency_ms` | accept + chamada ao endpoint até a predição |

# COMMAND ----------

serving_url = f"{workspace_host}/serving-endpoints/{endpoint_name}/invocations"
serving_headers = {**workspace_auth_headers(), "Content-Type": "application/json"}
if "Authorization" not in serving_headers:
    raise RuntimeError("Não foi possível autenticar contra o Model Serving (sem Authorization).")


def invoke_serving(record: dict[str, Any], read_timeout: float) -> tuple[int, dict | str, float]:
    t0 = time.time()
    resp = requests.post(
        serving_url,
        headers=serving_headers,
        json={"dataframe_records": [record]},
        timeout=(30, read_timeout),
    )
    latency_ms = (time.time() - t0) * 1000.0
    try:
        body: dict | str = resp.json()
    except Exception:  # noqa: BLE001
        body = resp.text or ""
    return resp.status_code, body, latency_ms


def insert_obs_row(row: dict[str, Any]) -> None:
    sdf = spark.createDataFrame(pd.DataFrame([row])).withColumn(
        "event_ts", F.to_timestamp("event_ts")
    )
    sdf.write.format("delta").mode("append").option("mergeSchema", "true").saveAsTable(obs_table)


def fetch_obs(request_id: str, retries: int = 8, sleep_s: float = 3.0) -> pd.DataFrame | None:
    for _ in range(retries):
        df = (
            spark.table(obs_table)
            .filter(F.col("request_id") == request_id)
            .toPandas()
        )
        if len(df):
            return df
        time.sleep(sleep_s)
    return None


def max_string_len(df: pd.DataFrame) -> int:
    max_len = 0
    for col in df.columns:
        series = df[col].astype(str)
        for val in series.tolist():
            if val in ("None", "nan", "NaT"):
                continue
            max_len = max(max_len, len(val))
    return max_len


# Teste negativo uma vez (menor tamanho), para não multiplicar 400s.
neg_size = payload_sizes_mb[0]
neg_payload = make_large_json(neg_size)
neg_rid, neg_uri = build_payload_path(catalog, schema)
neg_meta = write_payload_to_volume(neg_payload, neg_uri)
neg_record = build_serving_record(
    scalars=default_scalars(),
    payload_uri=neg_meta["payload_uri"],
    payload_hash="sha256:" + ("0" * 64),
    request_id=neg_rid,
)
neg_status, neg_body, neg_latency = invoke_serving(neg_record, read_timeout=120)
neg_ok = neg_status == 400
record_check(
    "A_serving_ref",
    neg_size,
    "A_wrong_hash_http_400",
    neg_ok,
    f"HTTP {neg_status}; esperado 400 por payload_hash divergente. corpo={str(neg_body)[:400]}",
    latency=neg_latency,
)
print(f"Negativo A (hash errado): HTTP {neg_status} em {neg_latency:.0f} ms")

for size_mb in payload_sizes_mb:
    print(f"\n=== A serving_ref size_mb={size_mb} ===")
    payload = make_large_json(size_mb)
    t_accept = time.time()
    request_id, payload_uri = build_payload_path(catalog, schema)
    meta = write_payload_to_volume(payload, payload_uri)
    accept_ms = (time.time() - t_accept) * 1000.0
    scalars = default_scalars()
    record = build_serving_record(
        scalars=scalars,
        payload_uri=meta["payload_uri"],
        payload_hash=meta["payload_hash"],
        request_id=request_id,
    )
    body_bytes = len(json.dumps({"dataframe_records": [record]}).encode("utf-8"))
    body_ok = body_bytes < 1024 * 1024
    record_check(
        "A_serving_ref",
        size_mb,
        "A_request_body_lt_1mib",
        body_ok,
        f"body_bytes={body_bytes} (deve ser << 1 MiB); payload_volume_bytes={meta['payload_bytes']}",
    )

    read_timeout = max(120.0, timeout_s)
    http_status, resp_body, invoke_ms = invoke_serving(record, read_timeout=read_timeout)
    total_ms = accept_ms + invoke_ms
    http_ok = http_status == 200
    record_check(
        "A_serving_ref",
        size_mb,
        "A_http_200",
        http_ok,
        f"HTTP {http_status} body={str(resp_body)[:400]}",
        latency=invoke_ms,
    )

    prediction, probability = parse_prediction(resp_body) if http_ok else (None, None)
    pred_ok = prediction is not None
    record_check(
        "A_serving_ref",
        size_mb,
        "A_prediction_present",
        pred_ok,
        f"prediction={prediction} probability={probability}",
        latency=invoke_ms,
    )

    row = observability_row(
        request_id=request_id,
        endpoint_name=endpoint_name,
        model_version="4",
        payload_uri=meta["payload_uri"],
        payload_hash=meta["payload_hash"],
        payload_bytes=meta["payload_bytes"],
        status="SUCCESS" if http_ok and pred_ok else "ERROR",
        latency_ms=total_ms,
        prediction=int(prediction) if prediction is not None else None,
        probability=float(probability) if probability is not None else None,
        error_message=None if http_ok else str(resp_body)[:1000],
        accept_latency_ms=accept_ms,
        total_latency_ms=total_ms,
    )
    row["env"] = catalog
    row["scalars_json"] = json.dumps(scalars, separators=(",", ":"))
    try:
        insert_obs_row(row)
    except Exception as exc:  # noqa: BLE001
        record_check(
            "A_serving_ref",
            size_mb,
            "A_observability_row",
            False,
            f"falha ao inserir observabilidade: {type(exc).__name__}: {exc}",
        )
        continue

    obs = fetch_obs(request_id)
    obs_ok = obs is not None and len(obs) > 0
    record_check(
        "A_serving_ref",
        size_mb,
        "A_observability_row",
        obs_ok,
        "linha encontrada" if obs_ok else "linha não encontrada após insert",
    )
    if not obs_ok:
        continue

    accept_tbl = obs_float(obs.iloc[0], "accept_latency_ms")
    total_tbl = obs_float(obs.iloc[0], "total_latency_ms")
    split_ok, split_detail = latency_split_ok(accept_tbl, total_tbl)
    record_check(
        "A_serving_ref",
        size_mb,
        "A_accept_latency_in_table",
        accept_tbl is not None and accept_tbl >= 0,
        f"tabela accept_latency_ms={accept_tbl} (cliente Volume+id={accept_ms:.1f} ms)",
        latency=accept_tbl,
    )
    record_check(
        "A_serving_ref",
        size_mb,
        "A_total_latency_in_table",
        split_ok,
        (
            f"{split_detail}; cliente invoke={invoke_ms:.1f} ms "
            f"total={total_ms:.1f} ms (Serving é síncrono: accept=Volume+id, total=accept+invoke)"
        ),
        latency=total_tbl,
    )
    latency_rows.append(
        {
            "approach": "A — Model Serving (payload_uri)",
            "size_mb": size_mb,
            "accept_latency_ms": None if accept_tbl is None else round(accept_tbl, 1),
            "total_latency_ms": None if total_tbl is None else round(total_tbl, 1),
            "accept_means": "gravar JSON no Volume + gerar request_id",
            "total_means": "accept + HTTP do endpoint até a predição",
        }
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Abordagem B — Databricks App assíncrona
# MAGIC
# MAGIC POST `/invocations` com os 15 escalares + `payload` → **202** rápido; GET até `done`/`failed`.
# MAGIC Auth: OAuth M2M (`client_credentials`) do SP `mlops-large-payload-validator`.
# MAGIC SKIP apenas se os secrets `validator-client-id` / `validator-client-secret` faltarem.
# MAGIC
# MAGIC **Latência (App é assíncrono de verdade):**
# MAGIC
# MAGIC | coluna | significado nesta abordagem |
# MAGIC |---|---|
# MAGIC | `accept_latency_ms` | receber o body + gerar `request_id` e devolver **202** |
# MAGIC | `total_latency_ms` | do início do POST até score + linha de observabilidade |

# COMMAND ----------

B_CHECKS = (
    "B_http_202_fast",
    "B_done_with_prediction",
    "B_observability_row",
    "B_payload_bytes_approx",
    "B_scalars_json_present",
    "B_no_full_json_in_table",
    "B_volume_file_exists",
    "B_accept_latency_in_table",
    "B_total_latency_in_table",
)


def skip_b_sizes(sizes: list[float], reason: str) -> None:
    for size_mb in sizes:
        for check in B_CHECKS:
            record_check("B_async_app", size_mb, check, True, reason, skipped=True)


def fail_b_remaining(size_mb: float, reason: str, latency: float | None = None) -> None:
    recorded = {
        r["check"]
        for r in results
        if r["approach"] == "B_async_app" and r["size_mb"] == size_mb
    }
    for check in B_CHECKS:
        if check in recorded:
            continue
        record_check("B_async_app", size_mb, check, False, reason, latency=latency)


app_base, app_headers, app_auth_detail, app_auth_mode = resolve_app_client()
print("App auth:", app_auth_detail)

if app_auth_mode == "skip":
    skip_b_sizes(payload_sizes_mb, "SKIP: " + app_auth_detail)
elif app_auth_mode != "ok" or not app_headers or not app_base:
    for size_mb in payload_sizes_mb:
        fail_b_remaining(size_mb, "FAIL auth App: " + app_auth_detail)
else:
    remaining = list(payload_sizes_mb)
    while remaining:
        size_mb = remaining.pop(0)
        print(f"\n=== B async_app size_mb={size_mb} ===")
        payload = make_large_json(size_mb)
        body = {**default_scalars(), "payload": payload}
        post_timeout = max(180.0, timeout_s)
        t_wall = time.time()
        try:
            post_resp = requests.post(
                f"{app_base}/invocations",
                headers=app_headers,
                json=body,
                timeout=(30, post_timeout),
                allow_redirects=False,
            )
        except Exception as exc:  # noqa: BLE001
            latency_ms = (time.time() - t_wall) * 1000.0
            msg = f"POST falhou: {type(exc).__name__}: {exc}"
            print(msg)
            record_check(
                "B_async_app",
                size_mb,
                "B_http_202_fast",
                False,
                msg,
                latency=latency_ms,
            )
            fail_b_remaining(size_mb, msg, latency=latency_ms)
            continue

        ttfb_ms = (time.time() - t_wall) * 1000.0
        body_preview = (post_resp.text or "")[:400]
        if post_resp.status_code in (401, 403) or looks_like_login_html(post_resp):
            reason = (
                f"FAIL: POST /invocations HTTP {post_resp.status_code} "
                f"(auth/proxy). body={body_preview}"
            )
            record_check(
                "B_async_app",
                size_mb,
                "B_http_202_fast",
                False,
                reason,
                latency=ttfb_ms,
            )
            fail_b_remaining(size_mb, reason, latency=ttfb_ms)
            continue

        accepted = app_json(post_resp) or {"raw": body_preview}
        request_id = accepted.get("request_id") if isinstance(accepted, dict) else None
        payload_uri = accepted.get("payload_uri") if isinstance(accepted, dict) else None
        fast_ok = post_resp.status_code == 202 and ttfb_ms < 60_000 and bool(request_id)
        record_check(
            "B_async_app",
            size_mb,
            "B_http_202_fast",
            fast_ok,
            f"HTTP {post_resp.status_code} time_to_202_ms={ttfb_ms:.0f} accepted={accepted}",
            latency=ttfb_ms,
        )
        if not request_id:
            fail_b_remaining(
                size_mb,
                f"HTTP {post_resp.status_code} sem request_id; poll impossível. body={body_preview}",
                latency=ttfb_ms,
            )
            continue

        deadline = time.time() + timeout_s
        last_job: dict[str, Any] = {}
        poll_error = None
        while time.time() < deadline:
            try:
                get_resp = requests.get(
                    f"{app_base}/invocations/{request_id}",
                    headers=app_headers,
                    timeout=30,
                    allow_redirects=False,
                )
                last_job = get_resp.json() if get_resp.ok else {"http": get_resp.status_code, "body": get_resp.text[:300]}
            except Exception as exc:  # noqa: BLE001
                poll_error = f"{type(exc).__name__}: {exc}"
                time.sleep(2)
                continue
            status = str(last_job.get("status") or "")
            if status in ("done", "failed"):
                break
            time.sleep(2)
        else:
            last_job["status"] = last_job.get("status") or "timeout"

        job_status = str(last_job.get("status") or "")
        prediction = last_job.get("prediction")
        done_ok = job_status == "done" and prediction is not None
        record_check(
            "B_async_app",
            size_mb,
            "B_done_with_prediction",
            done_ok,
            f"status={job_status} prediction={prediction} poll_error={poll_error} job={last_job}",
            latency=last_job.get("latency_ms"),
        )

        payload_uri = payload_uri or last_job.get("payload_uri")
        obs = fetch_obs(request_id, retries=12, sleep_s=5.0)
        obs_ok = obs is not None and len(obs) > 0
        record_check(
            "B_async_app",
            size_mb,
            "B_observability_row",
            obs_ok,
            "linha encontrada" if obs_ok else "linha não encontrada (warehouse/insert da App?)",
        )
        if not obs_ok:
            continue

        target_bytes = int(size_mb * 1024 * 1024)
        payload_bytes = int(obs.iloc[0]["payload_bytes"]) if obs.iloc[0]["payload_bytes"] is not None else -1
        slack = max(int(0.15 * target_bytes), 2048)
        bytes_ok = abs(payload_bytes - target_bytes) <= slack
        record_check(
            "B_async_app",
            size_mb,
            "B_payload_bytes_approx",
            bytes_ok,
            f"payload_bytes={payload_bytes} target≈{target_bytes} slack={slack}",
        )

        scalars_val = obs.iloc[0].get("scalars_json")
        scalars_ok = scalars_val is not None and str(scalars_val) not in ("", "None", "nan")
        record_check(
            "B_async_app",
            size_mb,
            "B_scalars_json_present",
            scalars_ok,
            f"scalars_json_len={len(str(scalars_val)) if scalars_ok else 0}",
        )

        max_len = max_string_len(obs)
        no_blob = max_len < min(payload_bytes * 0.25, 20_000) if payload_bytes > 0 else False
        record_check(
            "B_async_app",
            size_mb,
            "B_no_full_json_in_table",
            no_blob,
            f"max_string_col_len={max_len} vs payload_bytes={payload_bytes} (JSON completo não deve estar na tabela)",
        )

        vol_ok = False
        vol_detail = "payload_uri ausente"
        if payload_uri:
            try:
                p = Path(str(payload_uri))
                if p.exists():
                    vol_ok = abs(p.stat().st_size - payload_bytes) <= slack if payload_bytes > 0 else p.stat().st_size > 0
                    vol_detail = f"POSIX size={p.stat().st_size} uri={payload_uri}"
                else:
                    listing = dbutils.fs.ls(str(payload_uri))
                    file_size = listing[0].size if listing else -1
                    vol_ok = file_size > 0 and (
                        abs(file_size - payload_bytes) <= slack if payload_bytes > 0 else True
                    )
                    vol_detail = f"dbutils.fs.ls size={file_size} uri={payload_uri}"
            except Exception as exc:  # noqa: BLE001
                vol_detail = f"falha ao listar Volume: {type(exc).__name__}: {exc}; uri={payload_uri}"
        record_check(
            "B_async_app",
            size_mb,
            "B_volume_file_exists",
            vol_ok,
            vol_detail,
        )

        accept_tbl = obs_float(obs.iloc[0], "accept_latency_ms")
        total_tbl = obs_float(obs.iloc[0], "total_latency_ms")
        split_ok, split_detail = latency_split_ok(accept_tbl, total_tbl)
        total_client_ms = (time.time() - t_wall) * 1000.0
        record_check(
            "B_async_app",
            size_mb,
            "B_accept_latency_in_table",
            accept_tbl is not None and accept_tbl >= 0,
            (
                f"tabela accept_latency_ms={accept_tbl}; "
                f"cliente HTTP 202 em {ttfb_ms:.1f} ms (envio do JSON + request_id)"
            ),
            latency=accept_tbl,
        )
        record_check(
            "B_async_app",
            size_mb,
            "B_total_latency_in_table",
            split_ok,
            (
                f"{split_detail}; cliente wall-clock até done≈{total_client_ms:.1f} ms "
                f"(total deve ser > accept: 202 rápido, score em background)"
            ),
            latency=total_tbl,
        )
        latency_rows.append(
            {
                "approach": "B — App assíncrono",
                "size_mb": size_mb,
                "accept_latency_ms": None if accept_tbl is None else round(accept_tbl, 1),
                "total_latency_ms": None if total_tbl is None else round(total_tbl, 1),
                "accept_means": "receber o body + gerar request_id + HTTP 202",
                "total_means": "do POST até score + linha na tabela",
            }
        )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Relatório de latência (accept vs total)
# MAGIC
# MAGIC Colunas da tabela `inference_observability`:
# MAGIC
# MAGIC - **`accept_latency_ms`**: envio dos dados e geração do `request_id` (202 no App; Volume+id no Serving).
# MAGIC - **`total_latency_ms`**: até o resultado. No App inclui o trabalho em background.
# MAGIC - **`latency_ms`**: alias de `total_latency_ms` (legado).

# COMMAND ----------

latency_report = pd.DataFrame(latency_rows)
display(latency_report)
print(latency_report.to_string(index=False) if len(latency_report) else "sem linhas de latência")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Relatório final
# MAGIC
# MAGIC Qualquer check **FAIL** (não SKIP) falha o notebook (`raise`). SKIP é reservado para
# MAGIC bloqueio de plataforma (ex.: App recusando PAT/OAuth do job).

# COMMAND ----------

summary = pd.DataFrame(results)
display(summary)

failed = summary[summary["passed"] == "FAIL"]
skipped = summary[summary["passed"] == "SKIP"]
print(summary.to_string(index=False))
print(f"PASS={int((summary['passed']=='PASS').sum())} FAIL={len(failed)} SKIP={len(skipped)}")
if len(skipped):
    print("Checks SKIP:\n", skipped[["approach", "size_mb", "check", "detail"]].to_string(index=False))
if len(failed):
    display(failed)
    raise RuntimeError(
        "Validação falhou em "
        + "; ".join(
            f"{r.approach}/{r.size_mb}/{r.check}" for r in failed.itertuples()
        )
    )

print("Todos os checks não-SKIP passaram.")
