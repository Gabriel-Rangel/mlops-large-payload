"""Async Databricks App: accept large JSON + 15 scalars, score locally, persist metadata."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import uuid
from typing import Any

import uvicorn
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

import job_store
from scoring import LocalScorer, extract_scalars
from storage import build_payload_path, insert_observability_row, write_payload_bytes

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("large-payload-api")

# Databricks Apps inject hostname without scheme.
_host = os.environ.get("DATABRICKS_HOST", "")
if _host and not _host.startswith("http"):
    os.environ["DATABRICKS_HOST"] = f"https://{_host}"

CATALOG = os.environ.get("DATABRICKS_CATALOG", "gabriel_caballeria_demo_catalog")
SCHEMA = os.environ.get("DATABRICKS_SCHEMA", "mlops_large_payload")
VOLUME_NAME = os.environ.get("VOLUME_NAME", "request_payloads")
MODEL_URI = os.environ.get(
    "MODEL_URI",
    "models:/gabriel_caballeria_demo_catalog.mlops_large_payload.large_payload_scorer@Champion",
)
APP_NAME = os.environ.get("APP_NAME", os.environ.get("DATABRICKS_APP_NAME", "large-payload-api"))
MAX_BODY_BYTES = int(os.environ.get("MAX_BODY_BYTES", str(20 * 1024 * 1024)))

scorer: LocalScorer | None = None
scorer_load_error: str | None = None

app = FastAPI(
    title="Large payload inference API",
    description=(
        "Accepts 15 scalar features plus a large JSON payload. "
        "Writes the JSON to a UC Volume, scores locally (no Model Serving), "
        "and logs metadata to inference_observability."
    ),
    version="1.0.0",
)


@app.on_event("startup")
def _startup() -> None:
    global scorer, scorer_load_error
    try:
        scorer = LocalScorer(MODEL_URI)
        scorer_load_error = None
        logger.info("Scorer ready model_version=%s", scorer.model_version)
    except Exception as exc:  # noqa: BLE001
        scorer = None
        scorer_load_error = str(exc)
        logger.exception("Failed to load UC model at startup; scoring will fail until retry")


def sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _serialize_payload(payload: Any) -> tuple[dict[str, Any], bytes]:
    if payload is None:
        raise HTTPException(status_code=400, detail="payload is required")
    if isinstance(payload, (bytes, bytearray)):
        raw = bytes(payload)
        try:
            obj = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail=f"payload is not valid JSON: {exc}") from exc
        return obj, raw
    if isinstance(payload, str):
        try:
            obj = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail=f"payload is not valid JSON: {exc}") from exc
        raw = payload.encode("utf-8")
        return obj, raw
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="payload must be a JSON object")
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return payload, raw


async def _parse_json_body(request: Request) -> dict[str, Any]:
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > MAX_BODY_BYTES:
        raise HTTPException(status_code=413, detail=f"Body exceeds {MAX_BODY_BYTES} bytes")
    raw = await request.body()
    if len(raw) > MAX_BODY_BYTES:
        raise HTTPException(status_code=413, detail=f"Body exceeds {MAX_BODY_BYTES} bytes")
    try:
        body = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON body: {exc}") from exc
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="JSON body must be an object")
    return body


def enqueue_and_schedule(
    background_tasks: BackgroundTasks,
    scalars: dict[str, float],
    payload_obj: dict[str, Any],
    payload_raw: bytes,
    received_at: float,
) -> dict[str, Any]:
    request_id = str(uuid.uuid4())
    payload_uri = build_payload_path(CATALOG, SCHEMA, VOLUME_NAME, request_id)
    payload_hash = sha256_bytes(payload_raw)
    accept_latency_ms = (time.time() - received_at) * 1000.0
    job_store.create_job(
        request_id,
        {
            "status": "queued",
            "payload_uri": payload_uri,
            "payload_hash": payload_hash,
            "payload_bytes": len(payload_raw),
            "accept_latency_ms": accept_latency_ms,
        },
    )
    logger.info(
        "Accepted request_id=%s payload_bytes=%s accept_latency_ms=%.1f (JSON body is not logged)",
        request_id,
        len(payload_raw),
        accept_latency_ms,
    )
    background_tasks.add_task(
        process_inference,
        request_id,
        scalars,
        payload_obj,
        payload_raw,
        payload_uri,
        payload_hash,
        received_at,
        accept_latency_ms,
    )
    return {
        "request_id": request_id,
        "status": "queued",
        "payload_uri": payload_uri,
        "payload_bytes": len(payload_raw),
        "accept_latency_ms": accept_latency_ms,
        "status_url": f"/invocations/{request_id}",
    }


def process_inference(
    request_id: str,
    scalars: dict[str, float],
    payload_obj: dict[str, Any],
    payload_raw: bytes,
    payload_uri: str,
    payload_hash: str,
    received_at: float,
    accept_latency_ms: float,
) -> None:
    t_score = time.time()
    job_store.update_job(request_id, status="running")
    prediction = None
    probability = None
    error_message = None
    status = "SUCCESS"
    try:
        write_payload_bytes(payload_uri, payload_raw)
        active = scorer
        if active is None or not active.ready:
            raise RuntimeError(scorer_load_error or "Model is not loaded in the app process")
        prediction, probability = active.score(scalars, payload_obj)
    except Exception as exc:  # noqa: BLE001
        status = "ERROR"
        error_message = str(exc)[:4000]
        logger.exception("Inference failed request_id=%s", request_id)

    score_latency_ms = (time.time() - t_score) * 1000.0
    total_latency_ms = (time.time() - received_at) * 1000.0
    model_version = scorer.model_version if scorer else None
    model_alias = scorer.model_alias if scorer else "Champion"
    obs_error = None
    try:
        insert_observability_row(
            {
                "request_id": request_id,
                "endpoint_name": APP_NAME,
                "model_version": model_version,
                "model_alias": model_alias,
                "status": status,
                "latency_ms": total_latency_ms,
                "accept_latency_ms": accept_latency_ms,
                "total_latency_ms": total_latency_ms,
                "payload_uri": payload_uri,
                "payload_hash": payload_hash,
                "payload_bytes": len(payload_raw),
                "prediction": prediction,
                "probability": probability,
                "error_message": error_message,
                "env": CATALOG,
                "scalars_json": json.dumps(scalars, separators=(",", ":")),
            }
        )
    except Exception as exc:  # noqa: BLE001
        obs_error = str(exc)[:1000]
        logger.exception("Observability insert failed request_id=%s", request_id)
        if error_message:
            error_message = f"{error_message}; observability: {obs_error}"
        else:
            error_message = f"observability insert failed: {obs_error}"
    job_store.update_job(
        request_id,
        status="done" if status == "SUCCESS" and obs_error is None else "failed" if status == "ERROR" else "done",
        prediction=prediction,
        probability=probability,
        latency_ms=total_latency_ms,
        accept_latency_ms=accept_latency_ms,
        total_latency_ms=total_latency_ms,
        score_latency_ms=score_latency_ms,
        error_message=error_message,
        model_version=model_version,
        payload_uri=payload_uri,
        payload_hash=payload_hash,
        payload_bytes=len(payload_raw),
    )


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok" if scorer and scorer.ready else "degraded",
        "model_ready": bool(scorer and scorer.ready),
        "model_uri": MODEL_URI,
        "model_version": scorer.model_version if scorer else None,
        "model_load_error": scorer_load_error,
        "catalog": CATALOG,
        "schema": SCHEMA,
        "volume": VOLUME_NAME,
        "app": APP_NAME,
    }


@app.post("/invocations", status_code=202)
@app.post("/score", status_code=202)
async def invocations(request: Request, background_tasks: BackgroundTasks) -> JSONResponse:
    content_type = (request.headers.get("content-type") or "").lower()
    if content_type.startswith("multipart/form-data"):
        raise HTTPException(
            status_code=415,
            detail="Use POST /invocations/multipart for multipart/form-data",
        )
    received_at = time.time()
    body = await _parse_json_body(request)
    try:
        scalars = extract_scalars(body)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    payload_obj, payload_raw = _serialize_payload(body.get("payload"))
    result = enqueue_and_schedule(
        background_tasks, scalars, payload_obj, payload_raw, received_at
    )
    return JSONResponse(status_code=202, content=result)


@app.post("/invocations/multipart", status_code=202)
async def invocations_multipart(
    background_tasks: BackgroundTasks,
    payload: UploadFile | None = File(default=None),
    payload_json: str | None = Form(default=None),
    var_01: float = Form(...),
    var_02: float = Form(...),
    var_03: float = Form(...),
    var_04: float = Form(...),
    var_05: float = Form(...),
    var_06: float = Form(...),
    var_07: float = Form(...),
    var_08: float = Form(...),
    var_09: float = Form(...),
    var_10: float = Form(...),
    var_11: float = Form(...),
    var_12: float = Form(...),
    var_13: float = Form(...),
    var_14: float = Form(...),
    var_15: float = Form(...),
) -> JSONResponse:
    received_at = time.time()
    scalars = {
        "var_01": var_01,
        "var_02": var_02,
        "var_03": var_03,
        "var_04": var_04,
        "var_05": var_05,
        "var_06": var_06,
        "var_07": var_07,
        "var_08": var_08,
        "var_09": var_09,
        "var_10": var_10,
        "var_11": var_11,
        "var_12": var_12,
        "var_13": var_13,
        "var_14": var_14,
        "var_15": var_15,
    }
    if payload is not None:
        raw = await payload.read()
        payload_obj, payload_raw = _serialize_payload(raw)
    elif payload_json is not None:
        payload_obj, payload_raw = _serialize_payload(payload_json)
    else:
        raise HTTPException(status_code=400, detail="Provide payload file or payload_json")
    if len(payload_raw) > MAX_BODY_BYTES:
        raise HTTPException(status_code=413, detail=f"Body exceeds {MAX_BODY_BYTES} bytes")
    result = enqueue_and_schedule(
        background_tasks, scalars, payload_obj, payload_raw, received_at
    )
    return JSONResponse(status_code=202, content=result)


@app.get("/invocations/{request_id}")
@app.get("/score/{request_id}")
def get_invocation(request_id: str) -> dict[str, Any]:
    job = job_store.get_job(request_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown request_id")
    return job


if __name__ == "__main__":
    port = int(os.environ.get("DATABRICKS_APP_PORT", os.environ.get("PORT", "8000")))
    uvicorn.run(app, host="0.0.0.0", port=port)
