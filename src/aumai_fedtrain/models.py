"""Pydantic models for aumai-fedtrain."""

from __future__ import annotations

from pydantic import BaseModel, Field


class FederatedNode(BaseModel):
    """A participating node in a federated training run."""

    node_id: str = Field(..., description="Unique identifier for the node.")
    address: str = Field(..., description="Network address or identifier for the node.")
    capabilities: dict[str, object] = Field(
        default_factory=dict,
        description="Hardware / software capability descriptors.",
    )
    status: str = Field(
        default="idle",
        description="Current node status: idle, training, done, failed.",
    )


class TrainingConfig(BaseModel):
    """Configuration for a federated training run."""

    model_name: str = Field(..., description="Identifier for the model being trained.")
    learning_rate: float = Field(..., gt=0.0, description="Learning rate for local optimisation.")
    batch_size: int = Field(..., gt=0, description="Batch size for local training steps.")
    local_steps: int = Field(..., gt=0, description="Number of local gradient steps per round.")
    global_rounds: int = Field(..., gt=0, description="Total number of global aggregation rounds.")
    num_nodes: int = Field(..., gt=0, description="Expected number of participating nodes.")


class LocalUpdate(BaseModel):
    """Gradient update produced by a single node for one global round."""

    node_id: str = Field(..., description="ID of the node that produced this update.")
    round_id: int = Field(..., ge=0, description="Global round number.")
    gradients: dict[str, list[float]] = Field(
        ..., description="Parameter-name → gradient vector mapping."
    )
    loss: float = Field(..., description="Local training loss for this round.")
    samples_used: int = Field(..., ge=0, description="Number of training samples processed.")


class GlobalState(BaseModel):
    """Aggregated global model state after a round of federation."""

    round_id: int = Field(..., ge=0, description="Current global round number.")
    global_weights: dict[str, list[float]] = Field(
        default_factory=dict,
        description="Parameter-name → weight vector mapping.",
    )
    participating_nodes: list[str] = Field(
        default_factory=list,
        description="IDs of nodes that contributed to this round.",
    )
    avg_loss: float = Field(default=0.0, description="Average loss across all participating nodes.")


class TrainingResult(BaseModel):
    """Summary of a completed federated training run."""

    config: TrainingConfig = Field(..., description="The training configuration used.")
    final_loss: float = Field(..., description="Final average loss after all rounds.")
    rounds_completed: int = Field(..., ge=0, description="Total rounds actually completed.")
    total_samples: int = Field(..., ge=0, description="Cumulative samples processed across all nodes and rounds.")
    duration_seconds: float = Field(..., ge=0.0, description="Wall-clock duration of the training run.")


__all__ = [
    "FederatedNode",
    "GlobalState",
    "LocalUpdate",
    "TrainingConfig",
    "TrainingResult",
]
