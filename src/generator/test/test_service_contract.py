from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock


GENERATION_DIR = Path(__file__).resolve().parents[1] / "generation"
sys.path.insert(0, str(GENERATION_DIR))

from service_client import GeneratorServiceClient
from service_contract import (
    mima_mapping_to_vreq,
    mima_vector_to_vreq,
    normalize_generation_request,
    validate_generation_response,
)


class RequirementMappingTests(unittest.TestCase):
    def test_mima_vector_drops_hs_and_reorders_dimensions(self):
        first = mima_vector_to_vreq([0.6, 0.5, 0.8, 0.1, 1, 0, 0])
        second = mima_vector_to_vreq([0.6, 0.5, 0.8, 9.9, 1, 0, 0])
        self.assertEqual(first, [0.8, 0.6, 0.5, 1.0, 0.0, 0.0])
        self.assertEqual(first, second)

    def test_named_mima_vector_uses_same_mapping(self):
        result = mima_mapping_to_vreq(
            {
                "wp_m": 0.6,
                "hp_m": 0.5,
                "dp_m": 0.8,
                "hs_m": 0.2,
                "fl": 0,
                "fi": 1,
                "fp": 0,
            }
        )
        self.assertEqual(result, [0.8, 0.6, 0.5, 0.0, 1.0, 0.0])

    def test_request_defaults_are_stable(self):
        result = normalize_generation_request({"vreq": [0.8, 0.6, 0.5, 0, 0, 0]})
        self.assertEqual(result["samples_per_bar"], 64)
        self.assertEqual(result["top_k"], 10)
        self.assertEqual(result["seed"], 7)


class ResponseValidationTests(unittest.TestCase):
    @staticmethod
    def valid_response():
        return {
            "summary": {"generated": 1, "valid": 1, "returned": 1},
            "candidates": [
                {
                    "bar_type": "4-bar",
                    "structure": {
                        "nodes": [[0.0] * 7 for _ in range(8)],
                        "edges": [[0.0] * 8 for _ in range(8)],
                        "global": {},
                    },
                }
            ],
        }

    def test_valid_response_is_accepted(self):
        response = self.valid_response()
        self.assertIs(validate_generation_response(response), response)

    def test_client_posts_native_six_dimensional_request(self):
        client = GeneratorServiceClient("http://generator.test")
        response = self.valid_response()
        with mock.patch.object(client, "_request_json", return_value=response) as post:
            result = client.generate([0.8, 0.6, 0.5, 0, 0, 0], top_k=1)
        self.assertIs(result, response)
        payload = post.call_args.args[2]
        self.assertEqual(payload["vreq"], [0.8, 0.6, 0.5, 0.0, 0.0, 0.0])
        self.assertEqual(payload["top_k"], 1)


if __name__ == "__main__":
    unittest.main()
