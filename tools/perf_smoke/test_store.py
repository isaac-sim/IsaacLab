# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Baseline storage tests with a simulated Azure client."""

import io
import json
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from azure.core.exceptions import AzureError, ResourceExistsError

from . import store
from .contract import Contract
from .metrics import PerfSmokeError


class TestBaselineStore(unittest.TestCase):
    def setUp(self):
        contract = Contract(workload={"task": "task"}, runtime={"gpu_model": "l40s"})
        self.row = store.BaselineRow(
            contract.as_dict(),
            contract.hash,
            {"total_fps": 100.0},
            "abcdef123456",
            "2026-09-01T00:00:00Z",
            "run",
        )
        self.client = MagicMock()
        self.enterContext(
            patch.dict("os.environ", {store.BLOB_URL_ENV: "https://example.invalid/container?sig=secret"})
        )
        self.enterContext(patch.object(store, "make_container_client", return_value=self.client))
        self.enterContext(patch("sys.stderr", new=io.StringIO()))
        self.now = datetime(2026, 9, 1, tzinfo=timezone.utc)

    def test_reads_only_newest_rows_in_chronological_order(self):
        self.client.list_blobs.return_value = (
            SimpleNamespace(name=str(index), last_modified=index) for index in range(1000)
        )
        self.client.download_blob.return_value.readall.return_value = json.dumps(self.row.as_dict())
        rows = store.read(self.row.contract_hash, 3, self.now)
        self.assertEqual(rows, [self.row] * 3)
        self.assertEqual([call.args[0] for call in self.client.download_blob.call_args_list], ["997", "998", "999"])
        self.assertEqual(self.client.list_blobs.call_count, 1)
        self.client.close.assert_called_once()

    def test_corrupt_rows_do_not_abort_the_read(self):
        valid = self.row.as_dict()
        contents = [
            UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid byte"),
            "not json",
            json.dumps({**valid, "contract": {}}),
            json.dumps({**valid, "contract_hash": "wrong"}),
            json.dumps({**valid, "metrics": {"total_fps": -1}}),
            json.dumps(valid),
        ]
        self.client.list_blobs.return_value = [
            SimpleNamespace(name=str(index), last_modified=index) for index in range(len(contents))
        ]

        def download(name, **kwargs):
            result = MagicMock()
            content = contents[int(name)]
            if isinstance(content, Exception):
                result.readall.side_effect = content
            else:
                result.readall.return_value = content
            return result

        self.client.download_blob.side_effect = download
        self.assertEqual(store.read(self.row.contract_hash, len(contents), self.now), [self.row])
        self.client.close.assert_called_once()

    def test_bad_advisory_metric_does_not_discard_throughput(self):
        payload = self.row.as_dict()
        payload["metrics"]["ram_peak_gb"] = -1
        self.assertEqual(store.parse_row(payload, "row"), self.row)

    def test_errors_close_client_and_redact_credentials(self):
        self.client.list_blobs.side_effect = AzureError("https://example.invalid/container?sig=secret")
        with self.assertRaises(PerfSmokeError) as error:
            store.read(self.row.contract_hash, 1, self.now)
        self.assertNotIn("secret", str(error.exception))
        self.client.close.assert_called_once()

    def test_writes_are_create_only_and_close_client(self):
        self.client.upload_blob.side_effect = [None, ResourceExistsError()]
        self.assertTrue(store.write(self.row))
        self.assertFalse(store.write(self.row))
        first, second = self.client.upload_blob.call_args_list
        self.assertEqual(first.kwargs["name"], second.kwargs["name"])
        self.assertFalse(first.kwargs["overwrite"])
        self.assertEqual(self.client.close.call_count, 2)


if __name__ == "__main__":
    unittest.main()
