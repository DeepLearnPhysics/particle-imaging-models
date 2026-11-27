import json
import os
from pathlib import Path
import sys
from typing import Callable

import pytest

from tests.conftest import ProcessResult


pytestmark = [
    pytest.mark.gpu,
    pytest.mark.external_data,
    pytest.mark.external_model,
]

ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = Path(__file__).parent / "fixtures" / "metric_baselines.json"

TRANSIENT_HUB_ERRORS = (
    "ConnectError",
    "ConnectionError",
    "Connection reset by peer",
    "Connection aborted",
    "Max retries exceeded",
    "ReadTimeout",
    "Temporary failure in name resolution",
)


def _is_transient_hub_error(output):
    return any(signature in output for signature in TRANSIENT_HUB_ERRORS)


def _option_value(value):
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (list, tuple)):
        return json.dumps(value, separators=(",", ":"))
    return str(value)


@pytest.mark.parametrize("baseline_name", ("semantic", "particle"))
def test_reviewed_metric_baseline(
    baseline_name,
    pilarnet_mini_root,
    tmp_path,
    run_process: Callable[..., ProcessResult],
):
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    baseline = manifest["baselines"][baseline_name]
    output_dir = tmp_path / "eval"

    options = {
        "weight": baseline["model_uri"],
        "save_path": str(tmp_path / "run"),
        "eval_save_path": str(output_dir),
        "batch_size_test": 1,
        "data.test.data_root": str(pilarnet_mini_root),
        "data.test.revision": "v3",
        "data.test.split": "test",
        "data.test.max_len": manifest["dataset"]["event_count"],
        "data.test.min_points": 0,
        "test._delete_": True,
        **baseline.get("model_overrides", {}),
    }
    options.update(
        {f"test.{key}": value for key, value in baseline["evaluator"].items()}
    )

    command = [
        "uv",
        "run",
        "--no-sync",
        "python",
        "-m",
        "pimm.test",
        "--config-file",
        str(ROOT / baseline["config"]),
        "--options",
        *(f"{key}={_option_value(value)}" for key, value in options.items()),
    ]
    env = os.environ.copy()
    env.update(
        PIMM_TEST_DATA_ROOT=str(pilarnet_mini_root),
        HF_HUB_DISABLE_XET="1",
        UV_PROJECT_ENVIRONMENT=os.environ.get(
            "UV_PROJECT_ENVIRONMENT",
            sys.prefix,
        ),
    )
    # the run resolves the hf:// weight over the network; retry only on transient
    # hub connection errors so a genuine regression still fails on the first run.
    attempts = 3
    for attempt in range(1, attempts + 1):
        result = run_process(command, env=env, timeout=1800, capture=True)
        if result.output:
            print(result.output)
        if result.returncode == 0:
            break
        transient = result.output is not None and _is_transient_hub_error(result.output)
        if not transient or attempt == attempts:
            raise AssertionError(
                f"metric run failed (returncode={result.returncode}, "
                f"transient={transient}, attempt {attempt}/{attempts})"
            )

    payload = json.loads(
        (output_dir / "test_metrics.json").read_text(encoding="utf-8")
    )
    assert payload["weight"] == baseline["model_uri"]
    summary = payload["metrics"]["summary"]
    for name, expectation in baseline["expected"].items():
        assert summary[name] == pytest.approx(
            expectation["value"],
            abs=expectation["abs_tolerance"],
            rel=0,
        )
