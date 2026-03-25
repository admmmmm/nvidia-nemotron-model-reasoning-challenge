# Nemotron 4-bit Mamba Shape Mismatch Debug Notes

## Purpose

This note records the current server-side Nemotron debugging state so we can stop rediscovering the same facts.

The goal is not just "make it run once", but:

- describe the exact failure precisely
- reproduce the failure inside a smoke flow
- test mitigation ideas one by one
- keep the important server workflow files documented and versioned

## Current Exact Failure

The original blocking error is **not** a download error, a missing package error, or an out-of-memory error.

The original failure happens at the **first real training step** after the model is fully loaded and the training path reaches the Mamba fused kernel:

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

So the original bug is a **shape mismatch in the Mamba projection path under 4-bit quantization**, not a memory ceiling.

Concrete reproduced shapes so far:

- full official-style set on HF path:
  - `mat1 = (256 x 4096)`
  - `mat2 = (1 x 5505024)`
- minimal HF-attention-style test with only `q_proj`:
  - `mat1 = (256 x 4096)`
  - `mat2 = (1 x 5505024)`
- 4-bit with **no LoRA at all**:
  - still reproduces the same shape mismatch

The fact that the right-hand projection shape stays broken even with no LoRA is the critical clue.

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

So the earlier "smoke passed" only proved:

- the environment is good enough to build and load the model
- LoRA does not immediately explode on a tiny inference-style forward

It did **not** prove that the quantized Mamba path was safe for the real training forward.

## Reproduced In Training-Style Smoke

We now have a proper training-style smoke and have already reproduced the failure there.

The training-style smoke:

- loads a real row from `data/splits/default/train.jsonl`
- encodes it with the same formatting path as training
- includes `labels`
- runs the model in training mode
- uses the same 4-bit load path as training

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

- how to keep Nemotron trainable on the HF remote-code path while avoiding 4-bit quantization of the Mamba projection layers that are consumed by the fused Triton kernel

At this point, the strongest working hypothesis is:

- the original shape mismatch is **not primarily caused by LoRA target selection**
- the original shape mismatch is caused by **4-bit quantization touching Mamba projection layers** such as `in_proj`, `out_proj`, `x_proj`, and `dt_proj`
- LoRA target selection still matters for the final recipe, but it is a secondary problem after the quantization-path issue is addressed

## Current Local Mitigation Logic

Current local files:

- [src/train/lora_utils.py](src/train/lora_utils.py)
- [src/train/sft_local.py](src/train/sft_local.py)

Current Nemotron LoRA target list:

```python
[
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "down_proj",
    "up_proj",
    "gate_proj",
]
```

Current Nemotron 4-bit skip list:

```python
[
    "in_proj",
    "out_proj",
    "x_proj",
    "dt_proj",
]
```

This is now the main mitigation path under test on the HF route.

## What We Learned From Target Module And Quantization Tests

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

This is important, but not yet decisive by itself.

### Test 3: 4-bit + skip Mamba projection quantization + HF LoRA targets

Targets:

- `q_proj`
- `k_proj`
- `v_proj`
- `o_proj`
- `down_proj`
- `up_proj`
- `gate_proj`

Skipped from 4-bit conversion:

- `in_proj`
- `out_proj`
- `x_proj`
- `dt_proj`

Result:

- the original shape mismatch does **not** appear
- training-style forward completes and returns a loss value

This is the strongest evidence so far that the primary incompatibility is:

- **4-bit quantization touching Mamba kernel projection layers**

### Test 4: no 4-bit + HF LoRA targets

Targets:

- `q_proj`
- `k_proj`
- `v_proj`
- `o_proj`
- `down_proj`
- `up_proj`
- `gate_proj`

Result:

- no shape mismatch
- instead the run hits GPU OOM

Interpretation:

- removing 4-bit removes the shape-mismatch failure mode
- but the model no longer fits comfortably for training on this machine

### Test 5: 4-bit + no LoRA

Result:

- the same shape mismatch still appears:
  - `mat1 and mat2 shapes cannot be multiplied (256x4096 and 1x5505024)`

Interpretation:

- this is the decisive result
- **LoRA is not the root cause**
- the root cause is the 4-bit quantized Mamba path itself

### Test 6: 4-bit + skip Mamba projection quantization + HF LoRA targets + backward

Setup:

- same as Test 3
- shorter sequence length (`64`) to reduce memory pressure
- explicit `loss.backward()`

Result:

- no shape mismatch
- backward still does not complete cleanly:
  - `valid_labels = 0`
  - `loss = nan`
  - backward reports that the tensor does not require grad

Interpretation:

- the skip strategy appears to fix the original shape mismatch
- this specific backward failure is **not** the original bug
- the current smoke sample at this truncation length likely produced an all-masked loss
- we still need a meaningful non-NaN training-style smoke before restarting a full run

## Conversation Digest And Reasoning Update

This section summarizes the later debugging conversation so the reasoning trail is not lost.

### Initial mistaken path: "LoRA target modules are wrong"

The earlier working assumption was:

- full training crashes because the LoRA `target_modules` list is wrong for Nemotron on the HF path

That assumption was reasonable at first because:

- the official NVIDIA Megatron Bridge recipe documents a Nemotron LoRA target list
- the first reproducible crash only appeared after the model was wrapped with LoRA
- `linear_qkv` was not found on the HF base model

However, that explanation became much weaker after this finding:

- even **4-bit with no LoRA at all** still reproduced the **same** shape mismatch in the Mamba `out_proj` path

That changed the direction of the investigation completely.

### Revised hypothesis: 4-bit quantization and Mamba fused kernels are fighting

The key suspicious detail in the failure is the right-hand matrix shape:

```text
mat2 = (1 x 5505024)
```

That is not a normal 2D projection matrix shape for a standard `F.linear` call.

The hypothesis proposed during debugging was:

- `load_in_4bit=True` replaces standard linear layers with bitsandbytes 4-bit wrappers
- the Mamba fused training path in `ssd_combined.py` does **not** simply call the wrapped module in the same way as standard Transformer blocks
- instead, the fused path directly consumes projection weights and expects a regular 2D matrix layout
- once one of those Mamba projection layers is converted into a 4-bit packed representation, the fused kernel path receives an invalid weight layout and fails with the observed matrix multiplication error

This hypothesis explains several previously confusing observations:

1. Why `q_proj`-only LoRA still crashed in the Mamba `out_proj` path:
   - because 4-bit quantization is global and affects the Mamba projections too
2. Why inference smoke could pass while training failed:
   - because the short inference smoke did not exercise the same fused training path
3. Why the official Megatron target list was not enough to explain the bug:
   - because the failure sits below the LoRA target list, in the quantized Mamba execution path

### Ordered ablation study and its outcomes

The later experiments were intentionally reordered to validate the quantization hypothesis first:

1. 4-bit + skip Mamba projection quantization + HF LoRA targets
2. no 4-bit + HF LoRA targets
3. 4-bit + no LoRA

This ordering produced the cleanest causal picture:

- the skip strategy removes the original shape mismatch
- turning off 4-bit removes the original shape mismatch but causes OOM
- leaving 4-bit on with no LoRA still reproduces the original shape mismatch

That is the strongest evidence chain we have so far.

## Required Debugging Method From Now On

We should not restart full training first.

We should do this instead:

1. Precisely describe the current failure.
2. Reproduce it in a **training-style smoke**.
3. Change exactly one variable.
4. Re-run the same smoke.
5. Record the result.
6. Only restart full training after one target set and one quantization strategy survive the smoke.

## New Training-Style Smoke Script

New local helper:

- [scripts/remote_nemotron_train_smoke.py](scripts/remote_nemotron_train_smoke.py)

What it does:

- connects to the server over SSH
- activates the Nemotron conda env
- loads a real training sample from `data/splits/default/train.jsonl`
- encodes it using the same formatting pipeline as training
- builds the model in the same HF remote-code path as training
- optionally enables 4-bit quantization
- optionally applies a quantization skip list
- optionally injects LoRA with caller-provided `target_modules`
- runs a real **training-style forward with labels**
- optionally can also run `loss.backward()`

This is the right place to reproduce the original shape-mismatch error and to validate the current mitigation.

## Recommended Next Validation Order

Run these in order:

1. 4-bit + skip Mamba projection quantization + **no LoRA**
2. 4-bit + skip Mamba projection quantization + HF LoRA targets
3. same as #2, but on a sample / truncation length that yields `valid_labels > 0`
4. same as #3, plus `loss.backward()`
5. only then restart full training

Interpretation:

- if #1 passes, we confirm the skip list is sufficient to remove the original 4-bit/Mamba failure mode
- if #2 also passes, LoRA target selection is no longer the primary blocker
- if #3 gives a non-NaN loss, the training path is meaningfully alive
- if #4 succeeds, we have enough confidence to relaunch the full server training run

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
  Runs a remote training-style smoke with caller-controlled LoRA targets and quantization options.

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

Do **not** treat the original `1 x 5505024` problem as an OOM problem.

Do **not** keep solving it by changing RAM, batch size, or memory caps alone.

Do this instead:

- validate the 4-bit skip-module mitigation on training-style smoke
- then validate a meaningful non-NaN loss
- then validate backward
- only then restart the full training run

At this point, the highest-value next experiment is:

- verify that the skip-module mitigation produces a valid, non-NaN training loss on a sample with unmasked labels
- verify that LoRA parameters still require gradients after the skip strategy is applied
- if those pass, restart the full run with the skip strategy enabled by default


## New Confirmed Root Cause: Quantized `lm_head` Kills Gradients

A later single-load diagnostic finally pinpointed where the graph dies.

Observed in one forward pass with dummy `input_ids` and the current 4-bit + skip setup:

- `A.embeddings_out`: `requires_grad=True`
- `B.layer0_mamba_out`: `requires_grad=True`
- `C.layer1_moe_out`: `requires_grad=True`
- `D.layer5_attn_out`: `requires_grad=True`
- `Zpre.lm_head_in`: `dtype=torch.uint8`, `requires_grad=False`
- `Z.lm_head_out`: `dtype=torch.uint8`, `requires_grad=False`
- final `logits.requires_grad=False`
- final `loss.requires_grad=False`

This proves the graph was not dying inside early Mamba / MoE / Attention blocks.
It was dying when the backbone output was fed into a **quantized `lm_head`**.

The critical implementation detail is in the Nemotron HF remote-code forward path:

- logits are computed with the equivalent of `self.lm_head(hidden_states.to(self.lm_head.weight.dtype))`
- when `lm_head` is converted to `Linear4bit`, its weight dtype becomes `uint8`
- `hidden_states` therefore get cast to `uint8` right before the head
- that cast destroys autograd for the training loss path

### Confirmed Fix

Add `lm_head` to the Nemotron 4-bit skip list.

Current skip list in `src/train/sft_local.py`:

```python
NEMOTRON_4BIT_SKIP_MODULES = ["in_proj", "out_proj", "x_proj", "dt_proj", "lm_head"]
```

After that change, the single-load remote validation produced:

- `lm_head_type=Linear`
- `Zpre.lm_head_in`: `dtype=torch.float32`, `requires_grad=True`
- `Z.lm_head_out`: `dtype=torch.float32`, `requires_grad=True`
- `final logits.requires_grad=True`
- `final loss.requires_grad=True`
- `backward ok`

So the current state is:

- original Mamba shape mismatch fixed by skipping Mamba projection layers from 4-bit conversion
- no-grad backward failure fixed by also skipping `lm_head` from 4-bit conversion
- training-style forward + backward smoke now succeeds on the remote server
