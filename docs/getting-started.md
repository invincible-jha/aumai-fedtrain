# Getting Started with aumai-fedtrain

This guide takes you from installation to a complete simulated federated training run in about ten minutes.

---

## Prerequisites

- Python 3.11 or newer
- `pip`
- Basic familiarity with gradient-based machine learning (you do not need to know federated learning specifically)

No GPU, no ML framework, and no distributed infrastructure are required to follow this guide. The library ships a built-in simulation harness for local testing.

---

## Installation

### From PyPI

```bash
pip install aumai-fedtrain
```

### From source

```bash
git clone https://github.com/aumai/aumai-fedtrain
cd aumai-fedtrain
pip install -e ".[dev]"
```

### Verify

```bash
aumai-fedtrain --version
python -c "import aumai_fedtrain; print(aumai_fedtrain.__version__)"
```

---

## Core Concepts

Before running anything, it helps to understand four concepts:

**Node** — A `FederatedNode` is a participant in the training run. It has an ID, a network address, and capability metadata. In a real deployment, each node has its own private dataset and runs its own training loop.

**LocalUpdate** — After a node completes its local training steps, it sends back a `LocalUpdate` containing its gradient dictionary, the loss it achieved, and how many samples it processed.

**GlobalState** — The coordinator's view of the world after each round: the aggregated (averaged) gradients, which nodes participated, and the average loss.

**UpdateProvider** — The injection point between the coordinator and the nodes. Implement `get_updates(round_id, state, nodes)` to plug in real nodes or a simulator.

---

## Tutorial — Step by Step

### Step 1 — Define your training configuration

Create a `train.json` file:

```json
{
  "model_name": "my-model-v1",
  "learning_rate": 0.001,
  "batch_size": 32,
  "local_steps": 10,
  "global_rounds": 5,
  "num_nodes": 3
}
```

Field meanings:
- `local_steps` — how many gradient steps each node takes before sending its update. Higher values reduce communication but can diverge from the global optimum.
- `global_rounds` — how many times all nodes synchronise with the coordinator.
- `num_nodes` — for the simulation, how many virtual nodes to create.

### Step 2 — Run the CLI simulation

```bash
aumai-fedtrain run --config train.json --nodes 3
```

Expected output:

```
Starting federated training: my-model-v1  nodes=3  rounds=5

Training complete.
  Rounds completed:  5/5
  Total samples:     3412
  Final avg loss:    0.112843
  Duration:          0.002s

Node credits (samples processed):
  node-000: 1204 samples
  node-001: 1089 samples
  node-002: 1119 samples
```

The `Final avg loss` should be significantly lower than 1.0, showing that the simulated loss decays across rounds as designed.

### Step 3 — Verify gradient integrity

When nodes send gradient updates over a network, you may want to verify they have not been corrupted or tampered with. The `verify` command checks a `LocalUpdate` JSON file against a known SHA-256 hash.

First, create an `update.json` file:

```json
{
  "node_id": "node-000",
  "round_id": 1,
  "gradients": {
    "weight": [0.1, -0.2, 0.05, 0.03],
    "bias":   [0.01]
  },
  "loss": 0.852341,
  "samples_used": 128
}
```

Compute the expected hash programmatically:

```python
from aumai_fedtrain import HashVerifier, LocalUpdate

update = LocalUpdate.model_validate({
    "node_id": "node-000",
    "round_id": 1,
    "gradients": {"weight": [0.1, -0.2, 0.05, 0.03], "bias": [0.01]},
    "loss": 0.852341,
    "samples_used": 128,
})
verifier = HashVerifier()
print(verifier.compute_hash(update.gradients))
```

Then verify:

```bash
aumai-fedtrain verify --update update.json --hash <paste the hash here>
# Verification PASSED.  Hash: <hash>
```

### Step 4 — Run programmatically with a custom provider

```python
import math
import random

from aumai_fedtrain import (
    DiLoCoCoordinator,
    FederatedNode,
    GlobalState,
    LocalUpdate,
    TrainingConfig,
    UpdateProvider,
)


class DecayingLossProvider(UpdateProvider):
    """Produces updates with exponentially decaying loss."""

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
                gradients={
                    "w": [random.gauss(0, 0.05 / round_id) for _ in range(8)],
                    "b": [random.gauss(0, 0.01 / round_id)],
                },
                loss=math.exp(-round_id * 0.3),
                samples_used=random.randint(100, 300),
            )
            for node in nodes
        ]


config = TrainingConfig(
    model_name="demo",
    learning_rate=0.001,
    batch_size=32,
    local_steps=10,
    global_rounds=8,
    num_nodes=4,
)

nodes = [
    FederatedNode(node_id=f"node-{i}", address=f"192.168.1.{i}:5000")
    for i in range(4)
]

coordinator = DiLoCoCoordinator()
result = coordinator.run_training(config, nodes, DecayingLossProvider())

print(f"Final loss:    {result.final_loss:.6f}")
print(f"Total samples: {result.total_samples}")
print(f"Duration:      {result.duration_seconds:.4f}s")
```

### Step 5 — Inspect credit tracking

After training, see how many samples each node contributed:

```python
credits = coordinator.credit_tracker.all_credits()
for node_id, samples in sorted(credits.items()):
    print(f"  {node_id}: {samples} samples")

# Query a single node
node_0_credits = coordinator.credit_tracker.get_credits("node-0")
print(f"node-0 contributed {node_0_credits} samples total")
```

---

## Common Patterns and Recipes

### Pattern 1 — Run a single round manually

```python
from aumai_fedtrain import DiLoCoCoordinator, FederatedNode, TrainingConfig, LocalUpdate

coordinator = DiLoCoCoordinator()

config = TrainingConfig(
    model_name="manual-run",
    learning_rate=0.01,
    batch_size=64,
    local_steps=5,
    global_rounds=10,
    num_nodes=2,
)
nodes = [
    FederatedNode(node_id="node-a", address="10.0.0.1:5000"),
    FederatedNode(node_id="node-b", address="10.0.0.2:5000"),
]

# Start
state = coordinator.initialize(config, nodes)
print(f"Initialised at round {state.round_id}")

# Simulate one round
updates = [
    LocalUpdate(node_id="node-a", round_id=1, gradients={"w": [0.1, 0.2]}, loss=0.5, samples_used=64),
    LocalUpdate(node_id="node-b", round_id=1, gradients={"w": [0.2, 0.1]}, loss=0.4, samples_used=64),
]
new_state = coordinator.run_round(state, updates)
print(f"After round {new_state.round_id}: avg_loss={new_state.avg_loss}")
print(f"Averaged weights: {new_state.global_weights}")
```

### Pattern 2 — Verify updates in a pipeline

```python
from aumai_fedtrain import HashVerifier, LocalUpdate

verifier = HashVerifier()

# On the sending node: compute and attach the hash
update = LocalUpdate(
    node_id="node-0",
    round_id=3,
    gradients={"w": [0.01, -0.02, 0.005], "b": [0.001]},
    loss=0.321,
    samples_used=200,
)
hash_digest = verifier.compute_hash(update.gradients)

# On the receiving coordinator: verify before aggregating
if not verifier.verify(update, hash_digest):
    raise ValueError(f"Gradient tampering detected from {update.node_id}")
```

### Pattern 3 — Implement an HTTP-based provider

```python
from aumai_fedtrain import UpdateProvider, LocalUpdate, GlobalState, FederatedNode

class HttpUpdateProvider(UpdateProvider):
    """Collects real gradient updates from live nodes over HTTP."""

    def get_updates(
        self,
        round_id: int,
        state: GlobalState,
        nodes: list[FederatedNode],
    ) -> list[LocalUpdate]:
        import httpx
        updates: list[LocalUpdate] = []
        for node in nodes:
            response = httpx.post(
                f"http://{node.address}/train",
                json={
                    "round_id": round_id,
                    "global_weights": state.global_weights,
                },
                timeout=60.0,
            )
            response.raise_for_status()
            updates.append(LocalUpdate.model_validate(response.json()))
        return updates
```

### Pattern 4 — Checkpoint the global state

```python
import json
from aumai_fedtrain.models import GlobalState

# After each round, checkpoint the state
state_json = json.dumps(state.model_dump(mode="json"), indent=2)
with open(f"checkpoint_round_{state.round_id}.json", "w") as f:
    f.write(state_json)

# Reload from checkpoint
with open("checkpoint_round_5.json") as f:
    restored_state = GlobalState.model_validate_json(f.read())
print(f"Restored from round {restored_state.round_id}")
```

### Pattern 5 — Add node capabilities metadata

```python
from aumai_fedtrain.models import FederatedNode

nodes = [
    FederatedNode(
        node_id="gpu-node-0",
        address="10.0.0.1:5000",
        capabilities={
            "gpu": True,
            "gpu_model": "A100",
            "ram_gb": 80,
            "network_mbps": 10000,
        },
        status="idle",
    ),
    FederatedNode(
        node_id="cpu-node-1",
        address="10.0.0.2:5000",
        capabilities={
            "gpu": False,
            "ram_gb": 32,
            "network_mbps": 1000,
        },
        status="idle",
    ),
]
```

Node capabilities are available to your `UpdateProvider` for scheduling decisions (e.g. assigning larger batches to GPU nodes).

---

## Troubleshooting FAQ

**Q: `ValueError: Cannot average an empty list of updates.`**

Your `UpdateProvider.get_updates` returned an empty list. The coordinator requires at least one update per round. Check that your nodes are reachable and returning valid `LocalUpdate` objects.

---

**Q: `ValueError: No updates provided for aggregation.`**

Same root cause — `collect_updates` received an empty list. If you are using `run_training` with a custom provider, make sure `get_updates` returns at least one update for every round. The training loop will `break` early if an empty list is returned, completing fewer rounds than configured.

---

**Q: The verification command exits with code 1 even though I computed the hash myself.**

The `compute_hash` method serialises the gradient dict with `sort_keys=True`. Ensure you pass exactly the same gradient dict that was used to compute the hash — including key order and float precision. Even a difference of `0.1` vs `0.10000000000000001` in float representation will produce a different hash.

---

**Q: All my averaged gradient vectors are `[]` (empty).**

This means `GradientAverager.average` encountered a parameter key where every node's gradient list was either absent or empty. Check that your `LocalUpdate` objects have non-empty lists for every key in `gradients`.

---

**Q: How do I add more local steps without changing the coordinator?**

`local_steps` is a field on `TrainingConfig` and is visible to your `UpdateProvider` via the `config` argument passed to `run_training`. Your provider can use `config.local_steps` to control how many gradient steps each node takes before calling `get_updates`.

---

**Q: Can I use this with PyTorch?**

Yes. The library is framework-agnostic. Your `UpdateProvider` calls your PyTorch training loop and converts the resulting gradient tensors to `dict[str, list[float]]` before returning `LocalUpdate` objects. For example:

```python
gradients = {
    name: param.grad.detach().cpu().numpy().tolist()
    for name, param in model.named_parameters()
    if param.grad is not None
}
```

---

## Next Steps

- Read the [API Reference](api-reference.md) for complete class and method documentation.
- Run the [quickstart example](../examples/quickstart.py) for a self-contained Python demo.
- See [CONTRIBUTING.md](../CONTRIBUTING.md) for scope boundaries and contribution guidelines.
- Join the [AumAI Discord](https://discord.gg/aumai) for community help.
