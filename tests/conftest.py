"""Shared test fixtures for aumai-fedtrain."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aumai_fedtrain.core import (
    CreditTracker,
    DiLoCoCoordinator,
    GradientAverager,
    HashVerifier,
    UpdateProvider,
)
from aumai_fedtrain.models import (
    FederatedNode,
    GlobalState,
    LocalUpdate,
    TrainingConfig,
)


@pytest.fixture()
def training_config() -> TrainingConfig:
    return TrainingConfig(
        model_name="test-model-v1",
        learning_rate=0.01,
        batch_size=32,
        local_steps=5,
        global_rounds=3,
        num_nodes=2,
    )


@pytest.fixture()
def two_nodes() -> list[FederatedNode]:
    return [
        FederatedNode(
            node_id="node-001",
            address="127.0.0.1:5000",
            capabilities={"gpu": True},
        ),
        FederatedNode(
            node_id="node-002",
            address="127.0.0.2:5000",
            capabilities={"gpu": False},
        ),
    ]


@pytest.fixture()
def sample_update_node1() -> LocalUpdate:
    return LocalUpdate(
        node_id="node-001",
        round_id=1,
        gradients={"weight": [0.1, 0.2, 0.3], "bias": [0.01]},
        loss=0.5,
        samples_used=64,
    )


@pytest.fixture()
def sample_update_node2() -> LocalUpdate:
    return LocalUpdate(
        node_id="node-002",
        round_id=1,
        gradients={"weight": [0.3, 0.4, 0.5], "bias": [0.03]},
        loss=0.7,
        samples_used=128,
    )


@pytest.fixture()
def coordinator() -> DiLoCoCoordinator:
    return DiLoCoCoordinator()


@pytest.fixture()
def config_json_file(tmp_path: Path, training_config: TrainingConfig) -> Path:
    path = tmp_path / "train_config.json"
    path.write_text(training_config.model_dump_json(), encoding="utf-8")
    return path
