"""Local (in-app) scoring — no Model Serving round-trip.

Preferred path: derive JSON features from the in-memory payload, then
score with the sklearn artifact wrapped by the UC-registered pyfunc.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import pandas as pd

logger = logging.getLogger("large-payload-api.scoring")

SCALAR_FEATURES = [
    "var_01",
    "var_02",
    "var_03",
    "var_04",
    "var_05",
    "var_06",
    "var_07",
    "var_08",
    "var_09",
    "var_10",
    "var_11",
    "var_12",
    "var_13",
    "var_14",
    "var_15",
]

JSON_DERIVED_FEATURES = [
    "json_n_items",
    "json_total_amount",
    "json_avg_amount",
    "json_max_amount",
    "json_unique_categories",
]


def derive_features_from_json(payload: dict[str, Any]) -> dict[str, float]:
    items = payload.get("items") or []
    amounts = [float(i.get("amount", 0.0)) for i in items if isinstance(i, dict)]
    categories = {
        str(i.get("category"))
        for i in items
        if isinstance(i, dict) and i.get("category") is not None
    }
    n = len(amounts)
    total = float(sum(amounts)) if amounts else 0.0
    avg = total / n if n else 0.0
    mx = float(max(amounts)) if amounts else 0.0
    return {
        "json_n_items": float(n),
        "json_total_amount": total,
        "json_avg_amount": avg,
        "json_max_amount": mx,
        "json_unique_categories": float(len(categories)),
    }


def extract_scalars(body: dict[str, Any]) -> dict[str, float]:
    missing = [c for c in SCALAR_FEATURES if c not in body]
    if missing:
        raise ValueError(f"Missing scalar features: {missing}")
    return {c: float(body[c]) for c in SCALAR_FEATURES}


def build_feature_frame_from_payload(
    scalars: dict[str, float], payload: dict[str, Any]
) -> pd.DataFrame:
    derived = derive_features_from_json(payload)
    row = {**scalars, **derived}
    return pd.DataFrame([row], columns=SCALAR_FEATURES + JSON_DERIVED_FEATURES)


class LocalScorer:
    """Loads Champion pyfunc once and scores with the inner sklearn model."""

    def __init__(self, model_uri: str):
        self.model_uri = model_uri
        self.sklearn_model = None
        self.model_version: str | None = None
        self.model_alias = "Champion"
        self.error: str | None = None
        self._load()

    def _load(self) -> None:
        import mlflow
        from mlflow.tracking import MlflowClient

        os.environ.setdefault("MLFLOW_TRACKING_URI", "databricks")
        os.environ.setdefault("MLFLOW_REGISTRY_URI", "databricks-uc")
        mlflow.set_tracking_uri("databricks")
        mlflow.set_registry_uri("databricks-uc")

        logger.info("Loading model %s (this does not call Model Serving)", self.model_uri)
        pyfunc_model = mlflow.pyfunc.load_model(self.model_uri)
        python_model = pyfunc_model.unwrap_python_model()
        inner = getattr(python_model, "model", None)
        if inner is None:
            raise RuntimeError("UC pyfunc did not expose an inner sklearn model")
        self.sklearn_model = inner

        try:
            name = self.model_uri.split("models:/", 1)[-1].split("@", 1)[0]
            alias = self.model_uri.split("@", 1)[-1] if "@" in self.model_uri else "Champion"
            self.model_alias = alias
            mv = MlflowClient(registry_uri="databricks-uc").get_model_version_by_alias(name, alias)
            self.model_version = str(mv.version)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not resolve model version: %s", exc)
            self.model_version = None

        logger.info("Loaded sklearn model version=%s", self.model_version)

    @property
    def ready(self) -> bool:
        return self.sklearn_model is not None

    def score(self, scalars: dict[str, float], payload: dict[str, Any]) -> tuple[int, float]:
        if self.sklearn_model is None:
            raise RuntimeError(self.error or "Model is not loaded")
        features = build_feature_frame_from_payload(scalars, payload)
        proba = float(self.sklearn_model.predict_proba(features)[:, 1][0])
        prediction = int(proba >= 0.5)
        return prediction, proba
