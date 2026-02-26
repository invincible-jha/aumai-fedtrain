"""CLI entry point for aumai-fedtrain."""

from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

import click

from aumai_fedtrain.core import (
    DiLoCoCoordinator,
    HashVerifier,
    UpdateProvider,
)
from aumai_fedtrain.models import (
    FederatedNode,
    GlobalState,
    LocalUpdate,
    TrainingConfig,
)

# Module-level state shared across CLI commands within a single invocation.
_coordinator = DiLoCoCoordinator()
_current_config: TrainingConfig | None = None
_current_nodes: list[FederatedNode] = []
_current_state: GlobalState | None = None


class _SimulatedUpdateProvider(UpdateProvider):
    """Generates synthetic gradient updates for demonstration purposes."""

    def __init__(self, param_names: list[str], param_size: int) -> None:
        self._param_names = param_names
        self._param_size = param_size

    def get_updates(
        self,
        round_id: int,
        state: GlobalState,
        nodes: list[FederatedNode],
    ) -> list[LocalUpdate]:
        updates: list[LocalUpdate] = []
        for node in nodes:
            gradients: dict[str, list[float]] = {
                name: [
                    random.gauss(0.0, 0.1 / (round_id + 1))
                    for _ in range(self._param_size)
                ]
                for name in self._param_names
            }
            simulated_loss = math.exp(-round_id * 0.15) + random.uniform(0.0, 0.05)
            updates.append(
                LocalUpdate(
                    node_id=node.node_id,
                    round_id=round_id,
                    gradients=gradients,
                    loss=round(simulated_loss, 6),
                    samples_used=random.randint(64, 256),
                )
            )
        return updates


def _load_config(config_path: str) -> TrainingConfig:
    raw_text = Path(config_path).read_text(encoding="utf-8")
    try:
        data: object = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        click.echo(f"Invalid JSON in config: {exc}", err=True)
        sys.exit(1)
    if not isinstance(data, dict):
        click.echo("Config file must be a JSON object.", err=True)
        sys.exit(1)
    try:
        return TrainingConfig.model_validate(data)
    except Exception as exc:
        click.echo(f"Config validation error: {exc}", err=True)
        sys.exit(1)


@click.group()
@click.version_option()
def main() -> None:
    """AumAI FedTrain — generic federated training with the DiLoCo protocol."""


@main.command("init")
@click.option(
    "--config",
    "config_path",
    required=True,
    type=click.Path(exists=True),
    help="Path to training config JSON.",
)
@click.option(
    "--nodes",
    "num_nodes",
    default=4,
    show_default=True,
    type=int,
    help="Number of simulated nodes to create.",
)
def init_command(config_path: str, num_nodes: int) -> None:
    """Initialise a federated training session.

    Creates simulated nodes and the initial global state.

    Example: aumai-fedtrain init --config train.json --nodes 4
    """
    global _current_config, _current_nodes, _current_state

    _current_config = _load_config(config_path)
    _current_nodes = [
        FederatedNode(
            node_id=f"node-{i:03d}",
            address=f"127.0.0.{i + 1}:5000",
            capabilities={"gpu": True, "ram_gb": 16},
            status="idle",
        )
        for i in range(num_nodes)
    ]
    _current_state = _coordinator.initialize(_current_config, _current_nodes)

    click.echo(
        f"Initialised training run for model '{_current_config.model_name}' "
        f"with {num_nodes} node(s)."
    )
    click.echo(f"Global rounds planned: {_current_config.global_rounds}")
    click.echo(f"Local steps per round: {_current_config.local_steps}")
    click.echo(f"Learning rate:         {_current_config.learning_rate}")
    click.echo()
    for node in _current_nodes:
        click.echo(f"  Node: {node.node_id}  Address: {node.address}")


@main.command("run")
@click.option(
    "--config",
    "config_path",
    required=True,
    type=click.Path(exists=True),
    help="Path to training config JSON.",
)
@click.option(
    "--nodes",
    "num_nodes",
    default=4,
    show_default=True,
    type=int,
    help="Number of simulated nodes.",
)
@click.option(
    "--param-names",
    default="weight,bias",
    show_default=True,
    help="Comma-separated list of parameter names for simulation.",
)
@click.option(
    "--param-size",
    default=8,
    show_default=True,
    type=int,
    help="Vector length per parameter for simulation.",
)
def run_command(
    config_path: str, num_nodes: int, param_names: str, param_size: int
) -> None:
    """Run a full federated training simulation.

    Example: aumai-fedtrain run --config train.json
    """
    config = _load_config(config_path)
    nodes = [
        FederatedNode(
            node_id=f"node-{i:03d}",
            address=f"127.0.0.{i + 1}:5000",
            capabilities={"gpu": True, "ram_gb": 16},
            status="idle",
        )
        for i in range(num_nodes)
    ]

    names = [name.strip() for name in param_names.split(",") if name.strip()]
    provider = _SimulatedUpdateProvider(param_names=names, param_size=param_size)

    click.echo(
        f"Starting federated training: {config.model_name}"
        f"  nodes={num_nodes}  rounds={config.global_rounds}"
    )
    click.echo()

    coordinator = DiLoCoCoordinator()
    result = coordinator.run_training(config, nodes, provider)

    click.echo(f"Training complete.")
    click.echo(f"  Rounds completed:  {result.rounds_completed}/{config.global_rounds}")
    click.echo(f"  Total samples:     {result.total_samples}")
    click.echo(f"  Final avg loss:    {result.final_loss:.6f}")
    click.echo(f"  Duration:          {result.duration_seconds:.3f}s")
    click.echo()

    credits = coordinator.credit_tracker.all_credits()
    if credits:
        click.echo("Node credits (samples processed):")
        for node_id_key, sample_count in sorted(credits.items()):
            click.echo(f"  {node_id_key}: {sample_count} samples")


@main.command("status")
@click.option(
    "--round-id",
    "round_id",
    required=True,
    type=int,
    help="Round ID to query.",
)
@click.option(
    "--config",
    "config_path",
    default=None,
    type=click.Path(exists=False),
    help="Optional training config to show context.",
)
def status_command(round_id: int, config_path: str | None) -> None:
    """Show status information for a given round.

    Example: aumai-fedtrain status --round-id 5
    """
    if _current_state is not None:
        current_round = _current_state.round_id
        if round_id <= current_round:
            click.echo(f"Round {round_id}: completed.")
        else:
            click.echo(f"Round {round_id}: pending (current round is {current_round}).")

        click.echo(f"Current global round:    {_current_state.round_id}")
        click.echo(f"Participating nodes:     {', '.join(_current_state.participating_nodes)}")
        click.echo(f"Average loss this round: {_current_state.avg_loss:.6f}")

        credits = _coordinator.credit_tracker.all_credits()
        if credits:
            click.echo()
            click.echo("Cumulative node credits:")
            for nid, samples in sorted(credits.items()):
                click.echo(f"  {nid}: {samples} samples")
    else:
        click.echo(
            f"No active training session. Use 'init' to create one.",
            err=True,
        )

        if config_path:
            config = _load_config(config_path)
            click.echo(
                f"Config loaded: {config.model_name}"
                f" ({config.global_rounds} rounds planned)"
            )


@main.command("verify")
@click.option(
    "--update",
    "update_json_path",
    required=True,
    type=click.Path(exists=True),
    help="Path to a LocalUpdate JSON file to verify.",
)
@click.option("--hash", "expected_hash", required=True, help="Expected SHA-256 hex digest.")
def verify_command(update_json_path: str, expected_hash: str) -> None:
    """Verify the integrity of a LocalUpdate's gradients.

    Example: aumai-fedtrain verify --update update.json --hash <sha256>
    """
    raw_text = Path(update_json_path).read_text(encoding="utf-8")
    try:
        data: object = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        click.echo(f"Invalid JSON: {exc}", err=True)
        sys.exit(1)

    if not isinstance(data, dict):
        click.echo("Update file must be a JSON object.", err=True)
        sys.exit(1)

    try:
        update = LocalUpdate.model_validate(data)
    except Exception as exc:
        click.echo(f"Validation error: {exc}", err=True)
        sys.exit(1)

    verifier = HashVerifier()
    computed = verifier.compute_hash(update.gradients)

    if verifier.verify(update, expected_hash):
        click.echo(f"Verification PASSED.  Hash: {computed}")
    else:
        click.echo(f"Verification FAILED.")
        click.echo(f"  Expected: {expected_hash}")
        click.echo(f"  Computed: {computed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
