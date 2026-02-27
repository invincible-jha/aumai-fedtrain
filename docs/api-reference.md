# aumai-fedtrain API Reference

Generic federated training coordinator implementing the DiLoCo (Distributed
Low-Communication) protocol. This reference documents every public class,
method, and Pydantic model in the package.

---

## Table of Contents

1. [Models](#models)
   - [FederatedNode](#federatednode)
   - [TrainingConfig](#trainingconfig)
   - [LocalUpdate](#localupdate)
   - [GlobalState](#globalstate)
   - [TrainingResult](#trainingresult)
2. [Core Classes](#core-classes)
   - [GradientAverager](#gradientaverager)
   - [HashVerifier](#hashverifier)
   - [CreditTracker](#credittracker)
   - [DiLoCoCoordinator](#dilococoordinator)
   - [UpdateProvider](#updateprovider)
3. [CLI Commands](#cli-commands)
4. [Package Exports](#package-exports)

---

## Models

All models are Pydantic `BaseModel` subclasses and are importable from either
`aumai_fedtrain.models` or directly from `aumai_fedtrain`.

---

### FederatedNode

A participating node in a federated training run.

```python
from aumai_fedtrain import FederatedNode

node = FederatedNode(
    node_id="node-001",
    address="10.0.0.1:5000",
    capabilities={"gpu": True, "ram_gb": 32},
    status="idle",
)
```

#### Fields

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `node_id` | `str` | Yes | — | Unique identifier for the node. |
| `address` | `str` | Yes | — | Network address or identifier, e.g. `"host:port"`. |
| `capabilities` | `dict[str, object]` | No | `{}` | Arbitrary hardware or software capability descriptors. |
| `status` | `str` | No | `"idle"` | Current node status. Conventional values: `"idle"`, `"training"`, `"done"`, `"failed"`. |

---

### TrainingConfig

Configuration for a complete federated training run. All numeric fields are
validated to be strictly positive.

```python
from aumai_fedtrain import TrainingConfig

config = TrainingConfig(
    model_name="bert-small",
    learning_rate=1e-4,
    batch_size=32,
    local_steps=10,
    global_rounds=20,
    num_nodes=4,
)
```

#### Fields

| Field | Type | Required | Constraint | Description |
|---|---|---|---|---|
| `model_name` | `str` | Yes | — | Identifier for the model being trained. |
| `learning_rate` | `float` | Yes | `> 0.0` | Learning rate for local optimisation on each node. |
| `batch_size` | `int` | Yes | `> 0` | Batch size for local training steps. |
| `local_steps` | `int` | Yes | `> 0` | Number of local gradient steps each node performs per global round. |
| `global_rounds` | `int` | Yes | `> 0` | Total number of global aggregation rounds to run. |
| `num_nodes` | `int` | Yes | `> 0` | Expected number of participating nodes. |

---

### LocalUpdate

Gradient update produced by a single node for one global round. Submitted
to the coordinator after local training is complete.

```python
from aumai_fedtrain import LocalUpdate

update = LocalUpdate(
    node_id="node-001",
    round_id=1,
    gradients={
        "weight": [0.01, -0.02, 0.005],
        "bias": [0.001],
    },
    loss=0.4312,
    samples_used=128,
)
```

#### Fields

| Field | Type | Required | Constraint | Description |
|---|---|---|---|---|
| `node_id` | `str` | Yes | — | ID of the node that produced this update. |
| `round_id` | `int` | Yes | `>= 0` | Global round number this update belongs to. |
| `gradients` | `dict[str, list[float]]` | Yes | — | Maps parameter name to its gradient vector. |
| `loss` | `float` | Yes | — | Local training loss reported by the node for this round. |
| `samples_used` | `int` | Yes | `>= 0` | Number of training samples processed in this round. |

---

### GlobalState

Aggregated global model state produced after each round of federation.
Returned by `DiLoCoCoordinator.collect_updates` and `DiLoCoCoordinator.run_round`.

```python
from aumai_fedtrain import GlobalState

state = GlobalState(
    round_id=3,
    global_weights={"weight": [0.005, -0.01], "bias": [0.0005]},
    participating_nodes=["node-001", "node-002"],
    avg_loss=0.3821,
)
```

#### Fields

| Field | Type | Required | Constraint | Description |
|---|---|---|---|---|
| `round_id` | `int` | Yes | `>= 0` | Current global round number. |
| `global_weights` | `dict[str, list[float]]` | No | `{}` | Maps parameter name to the averaged weight vector after this round. |
| `participating_nodes` | `list[str]` | No | `[]` | IDs of nodes that contributed to this round. |
| `avg_loss` | `float` | No | `0.0` | Average loss across all participating nodes for this round. |

---

### TrainingResult

Summary of a completed federated training run. Returned by
`DiLoCoCoordinator.run_training`. Not normally constructed directly.

```python
# Typically received from run_training:
result = coordinator.run_training(config, nodes, provider)

print(f"Final loss:       {result.final_loss:.4f}")
print(f"Rounds completed: {result.rounds_completed}")
print(f"Total samples:    {result.total_samples}")
print(f"Duration:         {result.duration_seconds:.3f}s")
```

#### Fields

| Field | Type | Required | Constraint | Description |
|---|---|---|---|---|
| `config` | `TrainingConfig` | Yes | — | The training configuration used for this run. |
| `final_loss` | `float` | Yes | — | Final average loss after all rounds complete. |
| `rounds_completed` | `int` | Yes | `>= 0` | Total rounds actually completed. May be less than `config.global_rounds` if the update provider returns an empty list early. |
| `total_samples` | `int` | Yes | `>= 0` | Cumulative samples processed across all nodes and all rounds. |
| `duration_seconds` | `float` | Yes | `>= 0.0` | Wall-clock duration of the training run in seconds (4 decimal places). |

---

## Core Classes

Import from `aumai_fedtrain.core` or directly from `aumai_fedtrain`.

---

### GradientAverager

Averages gradients element-wise across all participating nodes. Nodes that do
not report a particular parameter key are treated as contributing a zero vector
of the same length as the first node that does report that key.

```python
from aumai_fedtrain import GradientAverager

averager = GradientAverager()
```

`GradientAverager` has no constructor parameters.

#### `GradientAverager.average`

```python
def average(self, updates: list[LocalUpdate]) -> dict[str, list[float]]
```

Compute the element-wise mean of gradients from all submitted updates.

**Parameters**

| Name | Type | Description |
|---|---|---|
| `updates` | `list[LocalUpdate]` | One `LocalUpdate` per participating node. |

**Returns**

`dict[str, list[float]]` — Maps each parameter name to its averaged gradient
vector. Keys are sorted alphabetically for determinism.

**Raises**

| Exception | Condition |
|---|---|
| `ValueError: "Cannot average an empty list of updates."` | `updates` is an empty list. |

**Example**

```python
from aumai_fedtrain import GradientAverager, LocalUpdate

averager = GradientAverager()

updates = [
    LocalUpdate(
        node_id="n1", round_id=1,
        gradients={"weight": [1.0, 2.0], "bias": [0.1]},
        loss=0.5, samples_used=64,
    ),
    LocalUpdate(
        node_id="n2", round_id=1,
        gradients={"weight": [3.0, 4.0], "bias": [0.3]},
        loss=0.3, samples_used=64,
    ),
]

averaged = averager.average(updates)
# {"bias": [0.2], "weight": [2.0, 3.0]}
```

---

### HashVerifier

Produces and verifies SHA-256 hashes of gradient dictionaries. The hash is
computed over the JSON serialisation of the gradients with sorted keys,
ensuring determinism across Python versions and platforms.

```python
from aumai_fedtrain import HashVerifier

verifier = HashVerifier()
```

`HashVerifier` has no constructor parameters.

#### `HashVerifier.compute_hash`

```python
def compute_hash(self, gradients: dict[str, list[float]]) -> str
```

Compute a deterministic SHA-256 hash of a gradients dictionary.

**Parameters**

| Name | Type | Description |
|---|---|---|
| `gradients` | `dict[str, list[float]]` | Parameter-name to gradient-vector mapping. |

**Returns**

`str` — Lowercase hexadecimal SHA-256 digest (64 characters).

**Example**

```python
grads = {"bias": [0.001], "weight": [0.01, -0.02]}
digest = verifier.compute_hash(grads)
# e.g. "a3f1c2d8..."  (64-char hex string)
```

#### `HashVerifier.verify`

```python
def verify(self, update: LocalUpdate, expected_hash: str) -> bool
```

Verify that the gradient payload of a `LocalUpdate` matches an expected hash.

**Parameters**

| Name | Type | Description |
|---|---|---|
| `update` | `LocalUpdate` | The update whose `gradients` field is to be verified. |
| `expected_hash` | `str` | SHA-256 hex digest to compare against. |

**Returns**

`bool` — `True` if the computed hash equals `expected_hash`, `False` otherwise.

**Example**

```python
digest = verifier.compute_hash(update.gradients)
assert verifier.verify(update, digest) is True
assert verifier.verify(update, "deadbeef" * 8) is False
```

---

### CreditTracker

Tracks cumulative sample contributions from each node across all rounds.
Useful for auditing participation and implementing sample-count-based
incentive schemes. Internally maintained by `DiLoCoCoordinator` and exposed
via the `credit_tracker` property.

```python
from aumai_fedtrain import CreditTracker

tracker = CreditTracker()
```

`CreditTracker` has no constructor parameters. Internal state initialises to
empty defaultdicts.

#### `CreditTracker.record_contribution`

```python
def record_contribution(self, node_id: str, samples: int, round_id: int) -> None
```

Add a node's sample count for a given round. Accumulates into that node's
running total and appends a `(round_id, samples)` entry to the node's history.

**Parameters**

| Name | Type | Description |
|---|---|---|
| `node_id` | `str` | The contributing node's identifier. |
| `samples` | `int` | Number of samples processed in this round. |
| `round_id` | `int` | The global round number. |

**Returns**

`None`

#### `CreditTracker.get_credits`

```python
def get_credits(self, node_id: str) -> int
```

Return the cumulative sample count for a single node.

**Parameters**

| Name | Type | Description |
|---|---|---|
| `node_id` | `str` | The node to query. |

**Returns**

`int` — Total samples processed by this node across all recorded rounds.
Returns `0` for unknown node IDs (no `KeyError`).

#### `CreditTracker.all_credits`

```python
def all_credits(self) -> dict[str, int]
```

Return a snapshot of cumulative credits for all nodes that have ever
contributed. Returns a plain `dict` copy; mutations do not affect the tracker.

**Returns**

`dict[str, int]` — Maps node ID to cumulative sample count.

**Example**

```python
tracker.record_contribution("node-001", samples=128, round_id=1)
tracker.record_contribution("node-001", samples=64,  round_id=2)
tracker.record_contribution("node-002", samples=256, round_id=1)

tracker.get_credits("node-001")   # 192
tracker.get_credits("node-999")   # 0  (unknown node)
tracker.all_credits()             # {"node-001": 192, "node-002": 256}
```

---

### DiLoCoCoordinator

Orchestrates federated training using the DiLoCo (Distributed
Low-Communication) protocol.

Each global round proceeds as follows:

1. All nodes perform `local_steps` gradient steps locally.
2. Nodes submit `LocalUpdate` objects to the coordinator.
3. The coordinator averages gradients element-wise and updates the global state.
4. Updated global weights are distributed back to all nodes.

`DiLoCoCoordinator` owns a `GradientAverager`, a `HashVerifier`, and a
`CreditTracker` internally. Access them through the `credit_tracker` and
`hash_verifier` properties.

```python
from aumai_fedtrain import DiLoCoCoordinator

coordinator = DiLoCoCoordinator()
```

`DiLoCoCoordinator` has no constructor parameters.

#### `DiLoCoCoordinator.initialize`

```python
def initialize(self, config: TrainingConfig, nodes: list[FederatedNode]) -> GlobalState
```

Create the initial `GlobalState` for a training run. Global weights are
initialised to an empty dict; the actual parameter structure is discovered
from node updates in the first round.

**Parameters**

| Name | Type | Description |
|---|---|---|
| `config` | `TrainingConfig` | Training configuration. |
| `nodes` | `list[FederatedNode]` | Participating nodes. |

**Returns**

`GlobalState` — Initial state at `round_id=0`, `global_weights={}`,
`avg_loss=0.0`, with `participating_nodes` set to all node IDs.

**Example**

```python
state = coordinator.initialize(config, nodes)
assert state.round_id == 0
assert state.avg_loss == 0.0
```

#### `DiLoCoCoordinator.collect_updates`

```python
def collect_updates(self, round_id: int, updates: list[LocalUpdate]) -> GlobalState
```

Aggregate node updates into a new `GlobalState`. Gradients are averaged
element-wise. Each node's sample contribution is recorded with the
`CreditTracker`.

**Parameters**

| Name | Type | Description |
|---|---|---|
| `round_id` | `int` | The current global round number. |
| `updates` | `list[LocalUpdate]` | One `LocalUpdate` per participating node. |

**Returns**

`GlobalState` — Updated state with averaged gradients as `global_weights`,
mean loss as `avg_loss`, and the contributing node IDs as
`participating_nodes`.

**Raises**

| Exception | Condition |
|---|---|
| `ValueError: "No updates provided for aggregation."` | `updates` is empty. |

#### `DiLoCoCoordinator.distribute_state`

```python
def distribute_state(self, state: GlobalState, nodes: list[FederatedNode]) -> None
```

Distribute the updated global state to all nodes. The reference implementation
is a deliberate no-op — there is no real network layer. In a production
system this method should serialise `state.global_weights` and push it to
each node's endpoint.

**Parameters**

| Name | Type | Description |
|---|---|---|
| `state` | `GlobalState` | The global state to distribute. |
| `nodes` | `list[FederatedNode]` | Target nodes. |

**Returns**

`None`

#### `DiLoCoCoordinator.run_round`

```python
def run_round(self, state: GlobalState, updates: list[LocalUpdate]) -> GlobalState
```

Execute a single full DiLoCo round. Increments the round counter by one and
aggregates updates. Distribution to nodes is handled separately via
`distribute_state`.

**Parameters**

| Name | Type | Description |
|---|---|---|
| `state` | `GlobalState` | Current global state before this round. |
| `updates` | `list[LocalUpdate]` | Node updates for this round. |

**Returns**

`GlobalState` — New state after aggregation, with
`round_id = state.round_id + 1`.

**Example**

```python
new_state = coordinator.run_round(state, updates)
assert new_state.round_id == state.round_id + 1
coordinator.distribute_state(new_state, nodes)
```

#### `DiLoCoCoordinator.run_training`

```python
def run_training(
    self,
    config: TrainingConfig,
    nodes: list[FederatedNode],
    update_provider: UpdateProvider,
) -> TrainingResult
```

Run the full federated training loop for `config.global_rounds` rounds.
Internally calls `update_provider.get_updates(round_id, state, nodes)` each
round. If the provider returns an empty list, training stops early.

**Parameters**

| Name | Type | Description |
|---|---|---|
| `config` | `TrainingConfig` | Training configuration. |
| `nodes` | `list[FederatedNode]` | Participating nodes. |
| `update_provider` | `UpdateProvider` | Callable-style object supplying updates each round. |

**Returns**

`TrainingResult` — Summary including `final_loss`, `rounds_completed`,
`total_samples`, and `duration_seconds`.

**Example**

```python
result = coordinator.run_training(config, nodes, my_provider)
print(f"Final loss: {result.final_loss:.4f} over {result.rounds_completed} rounds")
print(f"Samples processed: {result.total_samples}")
```

#### `DiLoCoCoordinator.credit_tracker` (property)

```python
@property
def credit_tracker(self) -> CreditTracker
```

Expose the internal `CreditTracker` for external querying after (or during)
training.

```python
credits = coordinator.credit_tracker.all_credits()
for node_id, samples in sorted(credits.items()):
    print(f"{node_id}: {samples} samples")
```

#### `DiLoCoCoordinator.hash_verifier` (property)

```python
@property
def hash_verifier(self) -> HashVerifier
```

Expose the internal `HashVerifier` for external gradient integrity checks.

```python
digest = coordinator.hash_verifier.compute_hash(update.gradients)
```

---

### UpdateProvider

Protocol-style base class for injecting custom node behaviour into
`DiLoCoCoordinator.run_training`. Subclass and override `get_updates` with
your own logic — either simulated local training or real distributed calls.

```python
from aumai_fedtrain import UpdateProvider, LocalUpdate, GlobalState, FederatedNode

class MyProvider(UpdateProvider):
    def get_updates(
        self,
        round_id: int,
        state: GlobalState,
        nodes: list[FederatedNode],
    ) -> list[LocalUpdate]:
        # Compute or retrieve real gradient updates here.
        ...
```

#### `UpdateProvider.get_updates`

```python
def get_updates(
    self,
    round_id: int,
    state: GlobalState,
    nodes: list[FederatedNode],
) -> list[LocalUpdate]
```

Return updates for all nodes for the current round. Must be overridden in
subclasses; the base class raises `NotImplementedError`.

Returning an empty list signals the coordinator to stop training early.

**Parameters**

| Name | Type | Description |
|---|---|---|
| `round_id` | `int` | Current global round number (1-indexed, starts at 1). |
| `state` | `GlobalState` | Current global state before this round's aggregation. |
| `nodes` | `list[FederatedNode]` | All participating nodes registered with the coordinator. |

**Returns**

`list[LocalUpdate]` — One `LocalUpdate` per active node. Return an empty
list to trigger early stopping.

**Raises**

| Exception | Condition |
|---|---|
| `NotImplementedError` | Called on the base class without overriding. |

---

## CLI Commands

The `aumai-fedtrain` entry point groups several sub-commands. All accept
`--help` for full option descriptions.

```
aumai-fedtrain [--version] [--help] COMMAND [ARGS]...
```

---

### `init`

Initialise a federated training session in memory with a config file and a
set of simulated nodes. Prints node addresses and configuration summary.

```
aumai-fedtrain init --config train.json [--nodes 4]
```

| Option | Type | Default | Description |
|---|---|---|---|
| `--config PATH` | `Path` | Required | Path to training config JSON. Must match `TrainingConfig` schema. |
| `--nodes INT` | `int` | `4` | Number of simulated nodes to create. |

**Config JSON example**

```json
{
  "model_name": "my-model",
  "learning_rate": 0.0001,
  "batch_size": 32,
  "local_steps": 10,
  "global_rounds": 20,
  "num_nodes": 4
}
```

---

### `run`

Run a complete federated training simulation from initialisation to final
round using synthetic Gaussian gradient updates. Prints per-run statistics
and per-node credit totals.

```
aumai-fedtrain run --config train.json [--nodes 4] [--param-names weight,bias] [--param-size 8]
```

| Option | Type | Default | Description |
|---|---|---|---|
| `--config PATH` | `Path` | Required | Path to training config JSON. |
| `--nodes INT` | `int` | `4` | Number of simulated nodes. |
| `--param-names STR` | `str` | `"weight,bias"` | Comma-separated list of parameter names for the simulation. |
| `--param-size INT` | `int` | `8` | Vector length per parameter in simulation. |

---

### `status`

Show status information for a given round within the current in-process
session (requires prior `init` in the same process invocation).

```
aumai-fedtrain status --round-id 5 [--config train.json]
```

| Option | Type | Default | Description |
|---|---|---|---|
| `--round-id INT` | `int` | Required | Round ID to query. |
| `--config PATH` | `Path` | `None` | Optional path to training config for additional context output. |

---

### `verify`

Verify the SHA-256 integrity of a `LocalUpdate` gradient payload stored as
a JSON file. Exits with code `0` on success, `1` on hash mismatch.

```
aumai-fedtrain verify --update update.json --hash <sha256_hex>
```

| Option | Type | Description |
|---|---|---|
| `--update PATH` | `Path` | Path to a `LocalUpdate` JSON file. |
| `--hash STR` | `str` | Expected SHA-256 hex digest to verify against. |

**Example workflow**

```bash
# Compute the hash of a known-good update:
aumai-fedtrain verify --update update.json --hash $(python -c "
from aumai_fedtrain import HashVerifier, LocalUpdate
import json, pathlib
data = json.loads(pathlib.Path('update.json').read_text())
u = LocalUpdate.model_validate(data)
print(HashVerifier().compute_hash(u.gradients))
")
```

---

## Package Exports

All public symbols are importable directly from `aumai_fedtrain`:

```python
from aumai_fedtrain import (
    # Models
    FederatedNode,
    TrainingConfig,
    LocalUpdate,
    GlobalState,
    TrainingResult,
    # Core classes
    GradientAverager,
    HashVerifier,
    CreditTracker,
    DiLoCoCoordinator,
    UpdateProvider,
)

import aumai_fedtrain
print(aumai_fedtrain.__version__)  # "0.1.0"
```
