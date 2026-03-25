# Nemotron Server Debug Notes

## Purpose
This note records the current server-side Nemotron debugging state so we can stop rediscovering the same facts.

The goal is not just "make it run once", but:

- describe the exact failure precisely
- reproduce the failure inside a smoke flow
- test mitigation ideas one by one
- keep the important server workflow files documented and versioned

## Current Exact Failure

The current blocking error is **not** a download error, a missing package error, or an out-of-memory error.

The current failure happens at the **first real training step** after the model is fully loaded and wrapped with LoRA:

```text
RuntimeError: mat1 and mat2 shapes cannot be multiplied (153x4096 and 1x5505024)
```

The relevant stack is:

- `modeling_nemotron_h.py`
- `mamba_ssm/ops/triton/ssd_combined.py`
- `F.linear(out, outproj_weight, outproj_bias)`

This means:

- model download succeeds
- 13/13 checkpoint shards load successfully
- Mamba dependencies import successfully
- 4-bit loading succeeds
- training enters `Trainer.train()`
- the crash happens in the **first forward pass used for training**

So the current bug is a **shape mismatch after LoRA injection**, not a memory ceiling.

Concrete reproduced shapes so far:

- full official-style set on HF path:
  - `mat1 = (256 x 4096)`
  - `mat2 = (1 x 5505024)`
- minimal HF-attention-style test with only `q_proj`:
  - `mat1 = (256 x 4096)`
  - `mat2 = (1 x 5505024)`

The fact that the right-hand projection shape stays broken even for a very small target subset is an important clue.

## Why The Official NVIDIA Recipe Does Not Directly Transfer

We are using the official NVIDIA Nemotron 3 documentation as guidance:

- `https://docs.nvidia.com/nemo/megatron-bridge/latest/models/llm/nemotron3.html`

That page is correct, but it is describing the **Megatron Bridge** recipe, not a plain Hugging Face + PEFT workflow.

Important differences:

- the recipe is designed for **Megatron checkpoints**
- it runs inside the Megatron Bridge / NeMo stack
- the documented target modules are defaults for that stack

The documented LoRA target modules are:

- `linear_qkv`
- `linear_proj`
- `linear_fc1`
- `linear_fc2`
- `in_proj`
- `out_proj`

Those names may exist in the HF remote-code model too, but that does **not** guarantee that PEFT wrapping those modules behaves identically in the HF execution path.

## Why Inference Smoke Can Pass While Training Still Fails

We already observed exactly this.

Inference smoke passed:

- short prompt
- plain forward
- no labels
- no real trainer step

Training failed:

- real encoded training sample
- labels included
- model in training mode
- Trainer / accelerate / autocast path
- Mamba kernel path actually exercised in the training flow

So "smoke passed" only proved:

- the environment is good enough to build and load the model
- LoRA does not immediately explode on a tiny inference-style forward

It did **not** prove that the LoRA-injected modules are safe for the real training forward path.

## Reproduced In Training-Style Smoke

We now have a proper training-style smoke and have already reproduced the failure there.

The training-style smoke:

- loads a real row from `data/splits/default/train.jsonl`
- encodes it with the same formatting path as training
- includes `labels`
- runs the model in training mode
- uses the same 4-bit load path

That means the current failure is no longer "only observed in the full trainer".

It is now reproducible in a controlled smoke step.

## Confirmed Environment Facts

The following were verified on the remote server:

- GPU: `NVIDIA A100-SXM4-40GB`
- Python: `3.10`
- Torch: `2.1.1+cu118`
- Transformers: `4.57.3`
- Accelerate: upgraded to a compatible release for current `Trainer`
- Bitsandbytes: upgraded to a release that supports the 4-bit path we use
- `mamba_ssm`: `2.0.4`
- `causal-conv1d`: updated to a version compatible with `mamba_ssm 2.0.4`
- 13/13 Nemotron checkpoint shards download successfully

We also verified a base forward and a simple LoRA forward smoke on the server.

## What Is Already Solved

These are **not** the current blocker anymore:

- model download instability
- missing HF token
- missing `transformers`
- old `bitsandbytes`
- old `causal-conv1d`
- plain model load OOM on the earlier low-RAM machine
- Mamba import / kernel dependency issues

## What Is Not Yet Solved

The unresolved issue is:

- which `target_modules` are actually safe for HF + PEFT training on this Nemotron model

The current failure strongly suggests:

- at least one currently targeted module produces an invalid projected weight layout during the real training forward

## Current Nemotron Target Module Logic

Current local file:

- [src/train/lora_utils.py](src/train/lora_utils.py)

Current Nemotron target list:

```python
[
    "linear_qkv",
    "linear_proj",
    "linear_fc1",
    "linear_fc2",
    "in_proj",
    "out_proj",
]
```

This mirrors the NVIDIA Megatron Bridge recipe, but still needs empirical validation on the HF path.

## What We Learned From Target Module Tests

### Test 0: full official-style set on the HF path

Targets:

- `linear_qkv`
- `linear_proj`
- `linear_fc1`
- `linear_fc2`
- `in_proj`
- `out_proj`

Result:

- reproducible shape mismatch in training-style smoke

### Test 1: only `linear_qkv`

Result:

- PEFT reports that `linear_qkv` is **not found** in the HF base model

This is a very strong signal that the documented Megatron Bridge target module names do not directly map to the HF remote-code module names.

### Test 2: only `q_proj`

Result:

- training-style smoke still fails with the same shape mismatch:
  - `mat1 and mat2 shapes cannot be multiplied (256x4096 and 1x5505024)`

This is also important:

- the failure is not caused only by the "full official set"
- even a tiny HF target subset can already break the Mamba projection path

## Required Debugging Method From Now On

We should not restart full training first.

We should do this instead:

1. Precisely describe the current failure.
2. Reproduce it in a **training-style smoke**.
3. Change exactly one variable.
4. Re-run the same smoke.
5. Record the result.
6. Only restart full training after one target set survives the smoke.

## New Training-Style Smoke Script

New local helper:

- [scripts/remote_nemotron_train_smoke.py](scripts/remote_nemotron_train_smoke.py)

What it does:

- connects to the server over SSH
- activates the Nemotron conda env
- loads the first real training sample from `data/splits/default/train.jsonl`
- encodes it using the same formatting pipeline as training
- builds the model in the same 4-bit loading path
- injects LoRA with a caller-provided `target_modules` list
- runs a real **training-style forward with labels**
- optionally can also run `loss.backward()`

This is the right place to reproduce the shape-mismatch error.

## Recommended Target Module Experiment Order

Run these in order, one by one:

1. base training-style smoke with **no LoRA**
2. `q_proj`
3. `k_proj`
4. `v_proj`
5. `o_proj`
6. `q_proj,k_proj,v_proj`
7. `q_proj,k_proj,v_proj,o_proj`

Interpretation:

- if base training-style smoke passes, we confirm the bug is introduced only by LoRA wrapping
- if every single HF projection target fails on its own, the problem is not "too many targets", but likely "wrong module family for PEFT on this model"
- if one or more individual targets pass, we can build back up incrementally

## Important Server Workflow Files

These are the files that matter most right now.

### Training / environment

- [scripts/setup_server.sh](scripts/setup_server.sh)
  Sets up the remote training environment.

- [scripts/run_nemotron_lora.sh](scripts/run_nemotron_lora.sh)
  Main Nemotron launch entrypoint used on the server.

- [scripts/finish_run.sh](scripts/finish_run.sh)
  End-of-run cleanup / follow-up workflow.

- [src/train/sft_local.py](src/train/sft_local.py)
  The actual training entrypoint currently used for HF + PEFT experiments.

- [src/train/lora_utils.py](src/train/lora_utils.py)
  Current LoRA target module selection logic.

### Remote debugging helpers

- [scripts/remote_nemotron_smoke.py](scripts/remote_nemotron_smoke.py)
  Runs a remote inference-style LoRA smoke.

- [scripts/remote_nemotron_train_smoke.py](scripts/remote_nemotron_train_smoke.py)
  Runs a remote training-style LoRA smoke with caller-specified targets.

- [scripts/remote_nemotron_status.py](scripts/remote_nemotron_status.py)
  Pulls remote logs, status, and GPU state.

- [scripts/remote_check_sync.py](scripts/remote_check_sync.py)
  Checks whether the Git sync loop is running on the server.

### Git / monitoring

- [scripts/monitor_training.sh](scripts/monitor_training.sh)
  Writes server-side JSON status updates.

- [scripts/sync_run_to_git.sh](scripts/sync_run_to_git.sh)
  Copies selected run artifacts into `/root/nemotron_git` and pushes them.

- [scripts/sync_loop.sh](scripts/sync_loop.sh)
  Periodic Git sync loop for the server.

## Current Decision Rule

Do **not** treat the current problem as an OOM problem.

Do **not** keep solving it by changing RAM, batch size, or memory caps alone.

Do this instead:

- isolate a safe LoRA target subset with training-style smoke
- then restart full training using only that validated subset

At this point, the highest-value next experiment is:

- run the training-style smoke with **no LoRA at all**
- then continue single-target sweeps (`k_proj`, `v_proj`, `o_proj`)
- decide whether HF+PEFT target matching is fundamentally viable for this model, or whether a model-specific adapter strategy is required

Once one subset survives:

1. patch `lora_utils.py`
2. sync to server
3. re-run one final training-style smoke
4. restart the real run
