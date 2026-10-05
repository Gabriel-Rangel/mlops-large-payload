"""Offline unit tests — no Databricks workspace required."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.model.features import (
    SCALAR_FEATURES,
    build_feature_frame,
    derive_features_from_json,
    load_json_payload,
    make_synthetic_training_frame,
    sha256_bytes,
)
from src.client.payload_ref import build_payload_path, build_serving_record, write_payload_to_volume


class FeatureTests(unittest.TestCase):
    def test_derive_features(self):
        payload = {
            "items": [
                {"amount": 10, "category": "A"},
                {"amount": 5, "category": "B"},
                {"amount": 5, "category": "A"},
            ]
        }
        feats = derive_features_from_json(payload)
        self.assertEqual(feats["json_n_items"], 3.0)
        self.assertEqual(feats["json_total_amount"], 20.0)
        self.assertEqual(feats["json_unique_categories"], 2.0)

    def test_synthetic_shapes(self):
        X, y = make_synthetic_training_frame(n=50)
        self.assertEqual(len(X), 50)
        self.assertEqual(len(y), 50)
        self.assertTrue(set(SCALAR_FEATURES).issubset(X.columns))

    def test_payload_write_and_serving_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            # Simulate Volume path layout under a temp root
            request_id, uri = build_payload_path("ml_dev", "mlops_large_payload")
            # Remap /Volumes/... into temp for offline test
            local_uri = str(Path(tmp) / Path(*Path(uri).parts[1:]))
            meta = write_payload_to_volume({"items": [{"amount": 1, "category": "X"}]}, local_uri)
            self.assertTrue(Path(local_uri).exists())
            self.assertTrue(meta["payload_hash"].startswith("sha256:"))

            scalars = {c: 0.0 for c in SCALAR_FEATURES}
            record = build_serving_record(scalars, local_uri, meta["payload_hash"], request_id)
            body = json.dumps(record).encode()
            self.assertLess(len(body), 8_000)  # tiny vs multi-MB payload on Volume
            self.assertEqual(sha256_bytes(Path(local_uri).read_bytes()), meta["payload_hash"])
            self.assertIn("payload_uri", record)
            self.assertEqual(record["request_id"], request_id)
            loaded = load_json_payload(local_uri)
            self.assertEqual(loaded["items"][0]["amount"], 1)

    def test_inline_payload_features(self):
        payload = {"items": [{"amount": 10, "category": "A"}]}
        row = {c: 0.0 for c in SCALAR_FEATURES}
        row["payload"] = json.dumps(payload)
        row["payload_uri"] = ""
        frame = build_feature_frame(__import__("pandas").DataFrame([row]))
        self.assertEqual(frame["json_n_items"].iloc[0], 1.0)
        self.assertEqual(frame["json_total_amount"].iloc[0], 10.0)


if __name__ == "__main__":
    unittest.main()
