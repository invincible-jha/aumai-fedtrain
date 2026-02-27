"""Tests for aumai-fedtrain core module (standard DiLoCo only)."""

from __future__ import annotations

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
    TrainingResult,
)


# ---------------------------------------------------------------------------
# Model validation tests
# ---------------------------------------------------------------------------


class TestModels:
    def test_training_config_valid(self, training_config: TrainingConfig) -> None:
        assert training_config.model_name == "test-model-v1"
        assert training_config.learning_rate == 0.01
        assert training_config.global_rounds == 3

    def test_training_config_zero_lr_raises(self) -> None:
        with pytest.raises(Exception):
            TrainingConfig(
                model_name="x",
                learning_rate=0.0,
                batch_size=32,
                local_steps=5,
                global_rounds=3,
                num_nodes=2,
            )

    def test_training_config_zero_batch_size_raises(self) -> None:
        with pytest.raises(Exception):
            TrainingConfig(
                model_name="x",
                learning_rate=0.01,
                batch_size=0,
                local_steps=5,
                global_rounds=3,
                num_nodes=2,
            )

    def test_training_config_zero_rounds_raises(self) -> None:
        with pytest.raises(Exception):
            TrainingConfig(
                model_name="x",
                learning_rate=0.01,
                batch_size=32,
                local_steps=5,
                global_rounds=0,
                num_nodes=2,
            )

    def test_federated_node_default_status(self) -> None:
        node = FederatedNode(node_id="n1", address="localhost:5000")
        assert node.status == "idle"

    def test_local_update_valid(
        self, sample_update_node1: LocalUpdate
    ) -> None:
        assert sample_update_node1.node_id == "node-001"
        assert sample_update_node1.loss == 0.5
        assert sample_update_node1.samples_used == 64

    def test_local_update_negative_samples_raises(self) -> None:
        with pytest.raises(Exception):
            LocalUpdate(
                node_id="n",
                round_id=1,
                gradients={"w": [0.1]},
                loss=0.5,
                samples_used=-1,
            )

    def test_global_state_valid(self) -> None:
        state = GlobalState(
            round_id=0,
            global_weights={},
            participating_nodes=[],
            avg_loss=0.0,
        )
        assert state.round_id == 0


# ---------------------------------------------------------------------------
# GradientAverager tests
# ---------------------------------------------------------------------------


class TestGradientAverager:
    def test_average_empty_raises(self) -> None:
        averager = GradientAverager()
        with pytest.raises(ValueError, match="empty"):
            averager.average([])

    def test_average_single_update(
        self, sample_update_node1: LocalUpdate
    ) -> None:
        averager = GradientAverager()
        result = averager.average([sample_update_node1])
        assert "weight" in result
        assert "bias" in result
        assert result["weight"] == pytest.approx([0.1, 0.2, 0.3])

    def test_average_two_updates_correct_mean(
        self,
        sample_update_node1: LocalUpdate,
        sample_update_node2: LocalUpdate,
    ) -> None:
        averager = GradientAverager()
        result = averager.average([sample_update_node1, sample_update_node2])
        # weight: mean([0.1,0.2,0.3], [0.3,0.4,0.5]) = [0.2, 0.3, 0.4]
        assert result["weight"] == pytest.approx([0.2, 0.3, 0.4])
        # bias: mean([0.01], [0.03]) = [0.02]
        assert result["bias"] == pytest.approx([0.02])

    def test_average_keys_sorted(
        self,
        sample_update_node1: LocalUpdate,
        sample_update_node2: LocalUpdate,
    ) -> None:
        averager = GradientAverager()
        result = averager.average([sample_update_node1, sample_update_node2])
        keys = list(result.keys())
        assert keys == sorted(keys)

    def test_average_missing_key_treated_as_zeros(self) -> None:
        averager = GradientAverager()
        u1 = LocalUpdate(
            node_id="n1", round_id=1,
            gradients={"weight": [1.0, 2.0], "bias": [0.5]},
            loss=0.5, samples_used=64,
        )
        u2 = LocalUpdate(
            node_id="n2", round_id=1,
            gradients={"weight": [3.0, 4.0]},  # no bias key
            loss=0.5, samples_used=64,
        )
        result = averager.average([u1, u2])
        # bias: [0.5 + 0.0] / 2 = [0.25]
        assert result["bias"] == pytest.approx([0.25])

    def test_average_empty_gradient_vector(self) -> None:
        averager = GradientAverager()
        u1 = LocalUpdate(
            node_id="n1", round_id=1,
            gradients={"empty_param": []},
            loss=0.5, samples_used=32,
        )
        result = averager.average([u1])
        assert result["empty_param"] == []

    def test_average_preserves_all_parameter_keys(
        self,
        sample_update_node1: LocalUpdate,
        sample_update_node2: LocalUpdate,
    ) -> None:
        averager = GradientAverager()
        result = averager.average([sample_update_node1, sample_update_node2])
        assert "weight" in result
        assert "bias" in result


# ---------------------------------------------------------------------------
# HashVerifier tests
# ---------------------------------------------------------------------------


class TestHashVerifier:
    def test_compute_hash_returns_hex_string(
        self, sample_update_node1: LocalUpdate
    ) -> None:
        verifier = HashVerifier()
        digest = verifier.compute_hash(sample_update_node1.gradients)
        assert isinstance(digest, str)
        assert len(digest) == 64
        assert all(c in "0123456789abcdef" for c in digest)

    def test_compute_hash_deterministic(
        self, sample_update_node1: LocalUpdate
    ) -> None:
        verifier = HashVerifier()
        h1 = verifier.compute_hash(sample_update_node1.gradients)
        h2 = verifier.compute_hash(sample_update_node1.gradients)
        assert h1 == h2

    def test_compute_hash_different_gradients_differ(
        self,
        sample_update_node1: LocalUpdate,
        sample_update_node2: LocalUpdate,
    ) -> None:
        verifier = HashVerifier()
        h1 = verifier.compute_hash(sample_update_node1.gradients)
        h2 = verifier.compute_hash(sample_update_node2.gradients)
        assert h1 != h2

    def test_compute_hash_empty_gradients(self) -> None:
        verifier = HashVerifier()
        digest = verifier.compute_hash({})
        assert len(digest) == 64

    def test_verify_correct_hash(
        self, sample_update_node1: LocalUpdate
    ) -> None:
        verifier = HashVerifier()
        expected = verifier.compute_hash(sample_update_node1.gradients)
        assert verifier.verify(sample_update_node1, expected) is True

    def test_verify_wrong_hash(
        self, sample_update_node1: LocalUpdate
    ) -> None:
        verifier = HashVerifier()
        assert verifier.verify(sample_update_node1, "a" * 64) is False

    def test_verify_tampered_gradient(
        self, sample_update_node1: LocalUpdate
    ) -> None:
        verifier = HashVerifier()
        original_hash = verifier.compute_hash(sample_update_node1.gradients)
        # Tamper with gradients
        tampered = LocalUpdate(
            node_id=sample_update_node1.node_id,
            round_id=sample_update_node1.round_id,
            gradients={"weight": [9.9, 9.9, 9.9], "bias": [9.9]},
            loss=sample_update_node1.loss,
            samples_used=sample_update_node1.samples_used,
        )
        assert verifier.verify(tampered, original_hash) is False


# ---------------------------------------------------------------------------
# CreditTracker tests
# ---------------------------------------------------------------------------


class TestCreditTracker:
    def test_initial_credits_zero(self) -> None:
        tracker = CreditTracker()
        assert tracker.get_credits("unknown-node") == 0

    def test_record_contribution_accumulates(self) -> None:
        tracker = CreditTracker()
        tracker.record_contribution("node-1", 100, round_id=1)
        tracker.record_contribution("node-1", 200, round_id=2)
        assert tracker.get_credits("node-1") == 300

    def test_record_contribution_multiple_nodes(self) -> None:
        tracker = CreditTracker()
        tracker.record_contribution("node-a", 100, round_id=1)
        tracker.record_contribution("node-b", 50, round_id=1)
        assert tracker.get_credits("node-a") == 100
        assert tracker.get_credits("node-b") == 50

    def test_all_credits_returns_all_nodes(self) -> None:
        tracker = CreditTracker()
        tracker.record_contribution("n1", 64, round_id=1)
        tracker.record_contribution("n2", 128, round_id=1)
        credits = tracker.all_credits()
        assert credits["n1"] == 64
        assert credits["n2"] == 128

    def test_all_credits_empty(self) -> None:
        tracker = CreditTracker()
        assert tracker.all_credits() == {}

    def test_all_credits_returns_copy(self) -> None:
        tracker = CreditTracker()
        tracker.record_contribution("n1", 64, round_id=1)
        c1 = tracker.all_credits()
        c2 = tracker.all_credits()
        c1["n1"] = 9999
        assert tracker.get_credits("n1") == 64  # original not mutated


# ---------------------------------------------------------------------------
# DiLoCoCoordinator tests
# ---------------------------------------------------------------------------


class TestDiLoCoCoordinator:
    def test_initialize_creates_round_zero(
        self,
        coordinator: DiLoCoCoordinator,
        training_config: TrainingConfig,
        two_nodes: list[FederatedNode],
    ) -> None:
        state = coordinator.initialize(training_config, two_nodes)
        assert state.round_id == 0
        assert state.avg_loss == 0.0

    def test_initialize_sets_participating_nodes(
        self,
        coordinator: DiLoCoCoordinator,
        training_config: TrainingConfig,
        two_nodes: list[FederatedNode],
    ) -> None:
        state = coordinator.initialize(training_config, two_nodes)
        assert "node-001" in state.participating_nodes
        assert "node-002" in state.participating_nodes

    def test_collect_updates_empty_raises(
        self, coordinator: DiLoCoCoordinator
    ) -> None:
        with pytest.raises(ValueError, match="No updates"):
            coordinator.collect_updates(round_id=1, updates=[])

    def test_collect_updates_averages_gradients(
        self,
        coordinator: DiLoCoCoordinator,
        sample_update_node1: LocalUpdate,
        sample_update_node2: LocalUpdate,
    ) -> None:
        state = coordinator.collect_updates(
            round_id=1, updates=[sample_update_node1, sample_update_node2]
        )
        assert state.round_id == 1
        assert "weight" in state.global_weights
        assert state.global_weights["weight"] == pytest.approx([0.2, 0.3, 0.4])

    def test_collect_updates_computes_avg_loss(
        self,
        coordinator: DiLoCoCoordinator,
        sample_update_node1: LocalUpdate,
        sample_update_node2: LocalUpdate,
    ) -> None:
        state = coordinator.collect_updates(
            round_id=1, updates=[sample_update_node1, sample_update_node2]
        )
        # avg_loss = (0.5 + 0.7) / 2 = 0.6
        assert state.avg_loss == pytest.approx(0.6)

    def test_collect_updates_records_credits(
        self,
        coordinator: DiLoCoCoordinator,
        sample_update_node1: LocalUpdate,
        sample_update_node2: LocalUpdate,
    ) -> None:
        coordinator.collect_updates(
            round_id=1, updates=[sample_update_node1, sample_update_node2]
        )
        assert coordinator.credit_tracker.get_credits("node-001") == 64
        assert coordinator.credit_tracker.get_credits("node-002") == 128

    def test_run_round_increments_round_id(
        self,
        coordinator: DiLoCoCoordinator,
        training_config: TrainingConfig,
        two_nodes: list[FederatedNode],
        sample_update_node1: LocalUpdate,
        sample_update_node2: LocalUpdate,
    ) -> None:
        state = coordinator.initialize(training_config, two_nodes)
        assert state.round_id == 0
        new_state = coordinator.run_round(
            state, [sample_update_node1, sample_update_node2]
        )
        assert new_state.round_id == 1

    def test_run_round_empty_updates_raises(
        self,
        coordinator: DiLoCoCoordinator,
        training_config: TrainingConfig,
        two_nodes: list[FederatedNode],
    ) -> None:
        state = coordinator.initialize(training_config, two_nodes)
        with pytest.raises(ValueError):
            coordinator.run_round(state, [])

    def test_distribute_state_no_op(
        self,
        coordinator: DiLoCoCoordinator,
        training_config: TrainingConfig,
        two_nodes: list[FederatedNode],
    ) -> None:
        state = coordinator.initialize(training_config, two_nodes)
        # Should not raise
        coordinator.distribute_state(state, two_nodes)

    def test_run_training_completes(
        self,
        training_config: TrainingConfig,
        two_nodes: list[FederatedNode],
    ) -> None:
        """Run a full training loop with a simple simulated update provider."""

        class ConstantProvider(UpdateProvider):
            def get_updates(
                self,
                round_id: int,
                state: GlobalState,
                nodes: list[FederatedNode],
            ) -> list[LocalUpdate]:
                return [
                    LocalUpdate(
                        node_id=node.node_id,
                        round_id=round_id,
                        gradients={"weight": [0.1, 0.2], "bias": [0.01]},
                        loss=1.0 / (round_id + 1),
                        samples_used=64,
                    )
                    for node in nodes
                ]

        coordinator = DiLoCoCoordinator()
        result = coordinator.run_training(training_config, two_nodes, ConstantProvider())

        assert isinstance(result, TrainingResult)
        assert result.rounds_completed == training_config.global_rounds
        assert result.total_samples > 0
        assert result.duration_seconds >= 0.0

    def test_run_training_empty_provider_stops_early(
        self,
        training_config: TrainingConfig,
        two_nodes: list[FederatedNode],
    ) -> None:
        """Provider returning no updates should stop the loop early."""

        class EmptyProvider(UpdateProvider):
            def get_updates(
                self,
                round_id: int,
                state: GlobalState,
                nodes: list[FederatedNode],
            ) -> list[LocalUpdate]:
                return []

        coordinator = DiLoCoCoordinator()
        result = coordinator.run_training(training_config, two_nodes, EmptyProvider())
        assert result.rounds_completed == 0

    def test_run_training_credit_tracker_accumulates(
        self,
        training_config: TrainingConfig,
        two_nodes: list[FederatedNode],
    ) -> None:
        class FixedProvider(UpdateProvider):
            def get_updates(
                self,
                round_id: int,
                state: GlobalState,
                nodes: list[FederatedNode],
            ) -> list[LocalUpdate]:
                return [
                    LocalUpdate(
                        node_id=node.node_id,
                        round_id=round_id,
                        gradients={"w": [0.1]},
                        loss=0.5,
                        samples_used=100,
                    )
                    for node in nodes
                ]

        coordinator = DiLoCoCoordinator()
        coordinator.run_training(training_config, two_nodes, FixedProvider())

        # 2 nodes * 3 rounds * 100 samples each
        for node in two_nodes:
            assert coordinator.credit_tracker.get_credits(node.node_id) == 300

    def test_hash_verifier_property(
        self, coordinator: DiLoCoCoordinator
    ) -> None:
        verifier = coordinator.hash_verifier
        assert isinstance(verifier, HashVerifier)

    def test_credit_tracker_property(
        self, coordinator: DiLoCoCoordinator
    ) -> None:
        tracker = coordinator.credit_tracker
        assert isinstance(tracker, CreditTracker)
