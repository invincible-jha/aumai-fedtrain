"""Tests for aumai-fedtrain CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from aumai_fedtrain.cli import main
from aumai_fedtrain.models import LocalUpdate, TrainingConfig


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


def _make_update_json(
    tmp_path: Path,
    node_id: str = "node-001",
    gradients: dict | None = None,  # type: ignore[type-arg]
) -> Path:
    """Write a LocalUpdate JSON file to disk and return its path."""
    if gradients is None:
        gradients = {"weight": [0.1, 0.2, 0.3], "bias": [0.01]}
    update = {
        "node_id": node_id,
        "round_id": 1,
        "gradients": gradients,
        "loss": 0.5,
        "samples_used": 64,
    }
    path = tmp_path / "update.json"
    path.write_text(json.dumps(update), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# --version
# ---------------------------------------------------------------------------


def test_cli_version(runner: CliRunner) -> None:
    result = runner.invoke(main, ["--version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.output


# ---------------------------------------------------------------------------
# init command
# ---------------------------------------------------------------------------


def test_init_creates_session(
    runner: CliRunner, config_json_file: Path
) -> None:
    result = runner.invoke(
        main, ["init", "--config", str(config_json_file), "--nodes", "2"]
    )
    assert result.exit_code == 0
    assert "test-model-v1" in result.output
    assert "node-000" in result.output
    assert "node-001" in result.output


def test_init_shows_config_details(
    runner: CliRunner, config_json_file: Path
) -> None:
    result = runner.invoke(
        main, ["init", "--config", str(config_json_file)]
    )
    assert result.exit_code == 0
    assert "Global rounds" in result.output
    assert "Learning rate" in result.output or "learning_rate" in result.output


def test_init_invalid_json(runner: CliRunner, tmp_path: Path) -> None:
    bad_file = tmp_path / "bad.json"
    bad_file.write_text("{not valid}", encoding="utf-8")
    result = runner.invoke(main, ["init", "--config", str(bad_file)])
    assert result.exit_code != 0


def test_init_non_object_json(runner: CliRunner, tmp_path: Path) -> None:
    arr_file = tmp_path / "arr.json"
    arr_file.write_text("[1, 2, 3]", encoding="utf-8")
    result = runner.invoke(main, ["init", "--config", str(arr_file)])
    assert result.exit_code != 0


def test_init_missing_required_field(runner: CliRunner, tmp_path: Path) -> None:
    # Missing learning_rate, etc.
    config_file = tmp_path / "incomplete.json"
    config_file.write_text('{"model_name": "x"}', encoding="utf-8")
    result = runner.invoke(main, ["init", "--config", str(config_file)])
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# run command
# ---------------------------------------------------------------------------


def test_run_completes(runner: CliRunner, config_json_file: Path) -> None:
    result = runner.invoke(
        main,
        [
            "run",
            "--config", str(config_json_file),
            "--nodes", "2",
            "--param-names", "weight,bias",
            "--param-size", "4",
        ],
    )
    assert result.exit_code == 0
    assert "Training complete" in result.output


def test_run_shows_rounds_and_loss(
    runner: CliRunner, config_json_file: Path
) -> None:
    result = runner.invoke(
        main,
        [
            "run",
            "--config", str(config_json_file),
            "--nodes", "2",
        ],
    )
    assert result.exit_code == 0
    assert "Rounds completed" in result.output
    assert "Final avg loss" in result.output


def test_run_shows_node_credits(
    runner: CliRunner, config_json_file: Path
) -> None:
    result = runner.invoke(
        main,
        [
            "run",
            "--config", str(config_json_file),
            "--nodes", "2",
        ],
    )
    assert result.exit_code == 0
    assert "node-000" in result.output or "samples" in result.output


def test_run_invalid_config(runner: CliRunner, tmp_path: Path) -> None:
    bad_file = tmp_path / "bad.json"
    bad_file.write_text("{bad}", encoding="utf-8")
    result = runner.invoke(main, ["run", "--config", str(bad_file)])
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# status command
# ---------------------------------------------------------------------------


def test_status_exits_cleanly(runner: CliRunner) -> None:
    """Status command must exit 0 regardless of whether a session is active."""
    result = runner.invoke(main, ["status", "--round-id", "1"])
    assert result.exit_code == 0
    # Output varies depending on whether a session was previously initialised
    assert "round" in result.output.lower() or "session" in result.output.lower()


def test_status_after_explicit_init(
    runner: CliRunner, config_json_file: Path
) -> None:
    """Initialise a session then check status — no crash expected."""
    runner.invoke(
        main, ["init", "--config", str(config_json_file), "--nodes", "2"]
    )
    result = runner.invoke(main, ["status", "--round-id", "0"])
    assert result.exit_code == 0
    assert "round" in result.output.lower() or "session" in result.output.lower()


def test_status_missing_round_id(runner: CliRunner) -> None:
    result = runner.invoke(main, ["status"])
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# verify command
# ---------------------------------------------------------------------------


def test_verify_correct_hash(runner: CliRunner, tmp_path: Path) -> None:
    import hashlib
    import json as json_module

    gradients = {"weight": [0.1, 0.2, 0.3], "bias": [0.01]}
    # Compute the expected hash the same way HashVerifier does
    serialised = json_module.dumps(gradients, sort_keys=True, separators=(",", ":"))
    expected_hash = hashlib.sha256(serialised.encode("utf-8")).hexdigest()

    update_path = _make_update_json(tmp_path, gradients=gradients)
    result = runner.invoke(
        main,
        ["verify", "--update", str(update_path), "--hash", expected_hash],
    )
    assert result.exit_code == 0
    assert "PASSED" in result.output


def test_verify_wrong_hash(runner: CliRunner, tmp_path: Path) -> None:
    update_path = _make_update_json(tmp_path)
    result = runner.invoke(
        main,
        ["verify", "--update", str(update_path), "--hash", "a" * 64],
    )
    assert result.exit_code != 0
    assert "FAILED" in result.output


def test_verify_invalid_json(runner: CliRunner, tmp_path: Path) -> None:
    bad_file = tmp_path / "bad.json"
    bad_file.write_text("{bad}", encoding="utf-8")
    result = runner.invoke(
        main,
        ["verify", "--update", str(bad_file), "--hash", "a" * 64],
    )
    assert result.exit_code != 0


def test_verify_non_object_json(runner: CliRunner, tmp_path: Path) -> None:
    arr_file = tmp_path / "arr.json"
    arr_file.write_text("[1, 2, 3]", encoding="utf-8")
    result = runner.invoke(
        main,
        ["verify", "--update", str(arr_file), "--hash", "a" * 64],
    )
    assert result.exit_code != 0


def test_verify_invalid_update_schema(runner: CliRunner, tmp_path: Path) -> None:
    bad_update = tmp_path / "bad_update.json"
    bad_update.write_text('{"node_id": "x"}', encoding="utf-8")  # missing fields
    result = runner.invoke(
        main,
        ["verify", "--update", str(bad_update), "--hash", "a" * 64],
    )
    assert result.exit_code != 0


def test_verify_missing_hash_option(runner: CliRunner, tmp_path: Path) -> None:
    update_path = _make_update_json(tmp_path)
    result = runner.invoke(main, ["verify", "--update", str(update_path)])
    assert result.exit_code != 0
