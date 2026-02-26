"""AumAI FedTrain — generic federated training with the DiLoCo protocol."""

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

__version__ = "0.1.0"

__all__ = [
    "CreditTracker",
    "DiLoCoCoordinator",
    "FederatedNode",
    "GlobalState",
    "GradientAverager",
    "HashVerifier",
    "LocalUpdate",
    "TrainingConfig",
    "TrainingResult",
    "UpdateProvider",
]
