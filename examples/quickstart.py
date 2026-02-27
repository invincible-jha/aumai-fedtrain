"""aumai-fedtrain quickstart example.

Demonstrates four common usage patterns for the aumai-fedtrain library.
Run directly to verify your installation:

    python examples/quickstart.py

All demos are self-contained and use only in-process objects — no network
connections or real model weights are required.
"""

from __future__ import annotations

import random

from aumai_fedtrain import (
    CreditTracker,
    DiLoCoCoordinator,
    FederatedNode,
    GlobalState,
    GradientAverager,
    HashVerifier,
    LocalUpdate,
    TrainingConfig,
    UpdateProvider,
)


# ---------------------------------------------------------------------------
# Demo 1: Gradient averaging
# ---------------------------------------------------------------------------


def demo_gradient_averaging() -> None:
    """Show how GradientAverager combines updates from multiple nodes.

    This is the lowest-level primitive in the DiLoCo protocol. In practice,
    DiLoCoCoordinator calls this automatically during collect_updates().
    """
    print("=== Demo 1: Gradient Averaging ===")

    averager = GradientAverager()

    # Simulate two nodes returning gradients for the same parameters.
    updates = [
        LocalUpdate(
            node_id="node-001",
            round_id=1,
            gradients={"weight": [1.0, 2.0, 3.0], "bias": [0.5]},
            loss=0.42,
            samples_used=128,
        ),
        LocalUpdate(
            node_id="node-002",
            round_id=1,
            gradients={"weight": [3.0, 4.0, 5.0], "bias": [1.5]},
            loss=0.38,
            samples_used=96,
        ),
    ]

    averaged = averager.average(updates)

    print("Node 001 weight gradient: [1.0, 2.0, 3.0]")
    print("Node 002 weight gradient: [3.0, 4.0, 5.0]")
    print(f"Averaged weight gradient: {averaged['weight']}")  # [2.0, 3.0, 4.0]
    print(f"Averaged bias gradient:   {averaged['bias']}")    # [1.0]
    print()


# ---------------------------------------------------------------------------
# Demo 2: Hash verification of gradient payloads
# ---------------------------------------------------------------------------


def demo_hash_verification() -> None:
    """Show how HashVerifier detects tampered gradient payloads.

    SHA-256 hashes are computed over a deterministically serialised JSON
    representation, so the same gradients always produce the same digest.
    """
    print("=== Demo 2: Hash Verification ===")

    verifier = HashVerifier()

    update = LocalUpdate(
        node_id="node-001",
        round_id=2,
        gradients={"weight": [0.01, -0.02, 0.005], "bias": [0.001]},
        loss=0.31,
        samples_used=200,
    )

    # Compute the canonical hash of this update's gradients.
    digest = verifier.compute_hash(update.gradients)
    print(f"SHA-256 digest: {digest}")

    # Verify the unmodified update — should pass.
    passed = verifier.verify(update, digest)
    print(f"Verification with correct hash:  {passed}")   # True

    # Verify against a wrong hash — should fail.
    failed = verifier.verify(update, "a" * 64)
    print(f"Verification with tampered hash: {failed}")   # False
    print()


# ---------------------------------------------------------------------------
# Demo 3: Credit tracking across rounds
# ---------------------------------------------------------------------------


def demo_credit_tracking() -> None:
    """Show how CreditTracker accumulates per-node sample contributions.

    Credits are expressed in total samples processed, enabling basic
    participation auditing and incentive calculations.
    """
    print("=== Demo 3: Credit Tracking ===")

    tracker = CreditTracker()

    # Simulate three rounds with varying node participation.
    contributions = [
        ("node-001", 128, 1),
        ("node-002", 256, 1),
        ("node-001", 64,  2),
        ("node-002", 256, 2),
        ("node-001", 192, 3),
        # node-002 absent in round 3
    ]

    for node_id, samples, round_id in contributions:
        tracker.record_contribution(node_id=node_id, samples=samples, round_id=round_id)

    print("Cumulative credits after 3 rounds:")
    for node_id, total in sorted(tracker.all_credits().items()):
        print(f"  {node_id}: {total} samples")
    # node-001: 384 samples
    # node-002: 512 samples
    print()


# ---------------------------------------------------------------------------
# Demo 4: Full federated training run
# ---------------------------------------------------------------------------


class SyntheticUpdateProvider(UpdateProvider):
    """Generates synthetic Gaussian gradient updates for demonstration.

    In production, replace this with a provider that calls real training
    workers or distributed compute nodes.
    """

    def __init__(self, param_names: list[str], param_size: int, seed: int = 42) -> None:
        self._param_names = param_names
        self._param_size = param_size
        self._rng = random.Random(seed)

    def get_updates(
        self,
        round_id: int,
        state: GlobalState,
        nodes: list[FederatedNode],
    ) -> list[LocalUpdate]:
        """Return one synthetic LocalUpdate per node.

        Loss decreases exponentially with round number to simulate convergence.
        """
        import math
        updates: list[LocalUpdate] = []
        for node in nodes:
            gradients = {
                name: [
                    self._rng.gauss(0.0, 0.1 / (round_id + 1))
                    for _ in range(self._param_size)
                ]
                for name in self._param_names
            }
            simulated_loss = math.exp(-round_id * 0.15) + self._rng.uniform(0.0, 0.05)
            updates.append(
                LocalUpdate(
                    node_id=node.node_id,
                    round_id=round_id,
                    gradients=gradients,
                    loss=round(simulated_loss, 6),
                    samples_used=self._rng.randint(64, 256),
                )
            )
        return updates


def demo_full_training_run() -> None:
    """Run a complete DiLoCo federated training loop.

    This demo exercises the complete DiLoCoCoordinator API:
      - TrainingConfig  defines hyperparameters
      - FederatedNode   represents each training participant
      - UpdateProvider  injects gradient updates each round
      - TrainingResult  summarises the completed run
    """
    print("=== Demo 4: Full Federated Training Run ===")

    # Define training hyperparameters.
    config = TrainingConfig(
        model_name="demo-transformer",
        learning_rate=1e-4,
        batch_size=32,
        local_steps=5,
        global_rounds=8,
        num_nodes=3,
    )

    # Create three simulated nodes.
    nodes = [
        FederatedNode(
            node_id=f"node-{i:03d}",
            address=f"10.0.0.{i + 1}:5000",
            capabilities={"gpu": True, "vram_gb": 16},
            status="idle",
        )
        for i in range(config.num_nodes)
    ]

    # The coordinator orchestrates all rounds.
    coordinator = DiLoCoCoordinator()

    # Wire up the update provider (replace with your real implementation).
    provider = SyntheticUpdateProvider(
        param_names=["embed.weight", "mlp.weight", "mlp.bias"],
        param_size=16,
        seed=0,
    )

    # Run training.
    result = coordinator.run_training(config, nodes, provider)

    print(f"Model:            {config.model_name}")
    print(f"Rounds completed: {result.rounds_completed}/{config.global_rounds}")
    print(f"Final avg loss:   {result.final_loss:.6f}")
    print(f"Total samples:    {result.total_samples:,}")
    print(f"Duration:         {result.duration_seconds:.4f}s")
    print()

    # Inspect per-node sample credit after training.
    print("Per-node sample credits:")
    for node_id, samples in sorted(coordinator.credit_tracker.all_credits().items()):
        print(f"  {node_id}: {samples:,} samples")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run all aumai-fedtrain quickstart demos."""
    print("aumai-fedtrain quickstart\n")
    demo_gradient_averaging()
    demo_hash_verification()
    demo_credit_tracking()
    demo_full_training_run()
    print("All demos complete.")


if __name__ == "__main__":
    main()
