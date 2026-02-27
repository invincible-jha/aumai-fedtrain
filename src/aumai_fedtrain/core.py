"""Core logic for aumai-fedtrain.

Standard DiLoCo federated training coordinator with:
  - Simple gradient averaging (no trust-weighting)
  - SHA-256 hash verification
  - Cumulative sample-count credit tracking
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import defaultdict

from aumai_fedtrain.models import (
    FederatedNode,
    GlobalState,
    LocalUpdate,
    TrainingConfig,
    TrainingResult,
)


class GradientAverager:
    """Averages gradients element-wise across all participating nodes."""

    def average(self, updates: list[LocalUpdate]) -> dict[str, list[float]]:
        """Compute the element-wise mean of gradients from all updates.

        All nodes must provide the same set of parameter keys.  If a
        parameter is missing from some nodes, those positions are treated
        as zero.

        Args:
            updates: List of ``LocalUpdate`` objects from all nodes.

        Returns:
            Dictionary mapping parameter name to averaged gradient vector.

        Raises:
            ValueError: If ``updates`` is empty.
        """
        if not updates:
            raise ValueError("Cannot average an empty list of updates.")

        # Collect all parameter keys.
        all_keys: set[str] = set()
        for update in updates:
            all_keys.update(update.gradients.keys())

        averaged: dict[str, list[float]] = {}
        n = len(updates)

        for key in sorted(all_keys):
            # Determine the vector length from the first node that has this key.
            vector_length = 0
            for update in updates:
                if key in update.gradients and update.gradients[key]:
                    vector_length = len(update.gradients[key])
                    break

            if vector_length == 0:
                averaged[key] = []
                continue

            summed = [0.0] * vector_length
            for update in updates:
                node_grad = update.gradients.get(key, [0.0] * vector_length)
                for index in range(min(len(node_grad), vector_length)):
                    summed[index] += node_grad[index]

            averaged[key] = [value / n for value in summed]

        return averaged


class HashVerifier:
    """Produces and verifies SHA-256 hashes of gradient dictionaries."""

    def compute_hash(self, gradients: dict[str, list[float]]) -> str:
        """Compute a deterministic SHA-256 hash of the gradients.

        The gradients dict is serialised as JSON with sorted keys to
        ensure determinism across Python versions.

        Args:
            gradients: Mapping of parameter names to gradient vectors.

        Returns:
            Lowercase hex string of the SHA-256 digest.
        """
        serialised = json.dumps(gradients, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialised.encode("utf-8")).hexdigest()

    def verify(self, update: LocalUpdate, expected_hash: str) -> bool:
        """Verify that the gradient payload matches an expected hash.

        Args:
            update: The ``LocalUpdate`` to verify.
            expected_hash: SHA-256 hex digest to compare against.

        Returns:
            True if the computed hash matches ``expected_hash``.
        """
        return self.compute_hash(update.gradients) == expected_hash


class CreditTracker:
    """Tracks cumulative sample contributions from each node."""

    def __init__(self) -> None:
        self._credits: dict[str, int] = defaultdict(int)
        self._history: dict[str, list[tuple[int, int]]] = defaultdict(list)

    def record_contribution(
        self, node_id: str, samples: int, round_id: int
    ) -> None:
        """Add a node's sample count for a given round.

        Args:
            node_id: The contributing node.
            samples: Number of samples processed in this round.
            round_id: The global round number.
        """
        self._credits[node_id] += samples
        self._history[node_id].append((round_id, samples))

    def get_credits(self, node_id: str) -> int:
        """Return the cumulative sample count for a node.

        Args:
            node_id: The node to query.

        Returns:
            Total samples processed by the node across all rounds.
        """
        return self._credits[node_id]

    def all_credits(self) -> dict[str, int]:
        """Return cumulative credits for all nodes.

        Returns:
            Mapping of node_id to cumulative sample count.
        """
        return dict(self._credits)


class DiLoCoCoordinator:
    """Orchestrates federated training using the DiLoCo protocol.

    DiLoCo = Distributed Low-Communication training.

    Each global round:
      1. All nodes perform ``local_steps`` gradient steps locally.
      2. Nodes submit ``LocalUpdate`` objects to the coordinator.
      3. The coordinator averages gradients and updates global weights.
      4. Updated global weights are distributed back to all nodes.
    """

    # This implementation uses simple gradient averaging only.

    def __init__(self) -> None:
        self._averager = GradientAverager()
        self._verifier = HashVerifier()
        self._credit_tracker = CreditTracker()

    def initialize(
        self, config: TrainingConfig, nodes: list[FederatedNode]
    ) -> GlobalState:
        """Create the initial global state for a training run.

        Initialises all weights to zero vectors of length 1 per parameter.
        The model's real parameter initialisation is expected to happen on
        the nodes themselves; the coordinator only tracks gradient deltas.

        Args:
            config: Training configuration.
            nodes: Participating nodes.

        Returns:
            Initial ``GlobalState`` at round 0.
        """
        node_ids = [node.node_id for node in nodes]
        # Initialise with an empty weight map; nodes supply the parameter keys
        # via their first update.
        return GlobalState(
            round_id=0,
            global_weights={},
            participating_nodes=node_ids,
            avg_loss=0.0,
        )

    def collect_updates(
        self, round_id: int, updates: list[LocalUpdate]
    ) -> GlobalState:
        """Aggregate node updates into a new global state.

        Gradients from all nodes are averaged element-wise.  Each node's
        sample contribution is recorded for credit tracking.

        Args:
            round_id: The current global round number.
            updates: ``LocalUpdate`` objects from all participating nodes.

        Returns:
            Updated ``GlobalState`` with averaged gradients as new weights.

        Raises:
            ValueError: If ``updates`` is empty.
        """
        if not updates:
            raise ValueError("No updates provided for aggregation.")

        averaged_gradients = self._averager.average(updates)

        avg_loss = sum(u.loss for u in updates) / len(updates)

        for update in updates:
            self._credit_tracker.record_contribution(
                node_id=update.node_id,
                samples=update.samples_used,
                round_id=round_id,
            )

        return GlobalState(
            round_id=round_id,
            global_weights=averaged_gradients,
            participating_nodes=[u.node_id for u in updates],
            avg_loss=avg_loss,
        )

    def distribute_state(
        self, state: GlobalState, nodes: list[FederatedNode]
    ) -> None:
        """Distribute the updated global state to all nodes.

        In this reference implementation, distribution is a no-op because
        there is no real network layer.  In a production system this would
        serialise ``state.global_weights`` and send it to each node's
        endpoint.

        Args:
            state: The global state to distribute.
            nodes: Target nodes.
        """
        # No-op reference implementation.
        # Production: serialise state and push to each node.node_id endpoint.
        _ = state
        _ = nodes

    def run_round(
        self, state: GlobalState, updates: list[LocalUpdate]
    ) -> GlobalState:
        """Execute a single full DiLoCo round.

        Collects updates, aggregates, and returns the new global state.
        Distribution is handled separately via ``distribute_state``.

        Args:
            state: Current global state.
            updates: Node updates for this round.

        Returns:
            New ``GlobalState`` after aggregation.
        """
        next_round_id = state.round_id + 1
        return self.collect_updates(round_id=next_round_id, updates=updates)

    def run_training(
        self,
        config: TrainingConfig,
        nodes: list[FederatedNode],
        update_provider: UpdateProvider,
    ) -> TrainingResult:
        """Run a full federated training loop.

        ``update_provider`` is a callable conforming to::

            def get_updates(round_id: int, state: GlobalState, nodes: list[FederatedNode]) -> list[LocalUpdate]: ...

        This allows callers to inject simulated or real node behaviour.

        Args:
            config: Training configuration.
            nodes: Participating nodes.
            update_provider: Callable that returns updates for each round.

        Returns:
            ``TrainingResult`` summarising the completed run.
        """
        start_time = time.monotonic()
        state = self.initialize(config, nodes)

        total_samples = 0
        rounds_completed = 0
        final_loss = 0.0

        for round_index in range(config.global_rounds):
            updates = update_provider.get_updates(
                round_id=round_index + 1, state=state, nodes=nodes
            )
            if not updates:
                break

            state = self.run_round(state, updates)
            self.distribute_state(state, nodes)

            total_samples += sum(u.samples_used for u in updates)
            rounds_completed += 1
            final_loss = state.avg_loss

        duration = time.monotonic() - start_time

        return TrainingResult(
            config=config,
            final_loss=final_loss,
            rounds_completed=rounds_completed,
            total_samples=total_samples,
            duration_seconds=round(duration, 4),
        )

    @property
    def credit_tracker(self) -> CreditTracker:
        """Expose the credit tracker for external querying."""
        return self._credit_tracker

    @property
    def hash_verifier(self) -> HashVerifier:
        """Expose the hash verifier for external use."""
        return self._verifier


class UpdateProvider:
    """Protocol-style base class for update providers.

    Subclass and implement ``get_updates`` to inject custom node
    behaviour into ``DiLoCoCoordinator.run_training``.
    """

    def get_updates(
        self,
        round_id: int,
        state: GlobalState,
        nodes: list[FederatedNode],
    ) -> list[LocalUpdate]:
        """Return simulated or real updates for all nodes.

        Args:
            round_id: Current global round number.
            state: Current global state.
            nodes: Participating nodes.

        Returns:
            List of ``LocalUpdate`` objects (one per active node).
        """
        raise NotImplementedError


__all__ = [
    "CreditTracker",
    "DiLoCoCoordinator",
    "GradientAverager",
    "HashVerifier",
    "UpdateProvider",
]
