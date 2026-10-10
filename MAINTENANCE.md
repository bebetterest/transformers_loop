# Maintaining Transformers Loop

This unofficial fork follows Hugging Face Transformers while keeping the decoder loop implementation modular and
opt-in. Use `main` for reviewed updates and a fixed tag or commit for experiments. Upstream history, the Apache 2.0
license, and attribution are preserved. The Python package and import remain `transformers`.

The [loop guide](loop_readme.md) explains configuration, supported adapters, caches, and training. This document
describes how to update the fork without treating an older validation result as evidence for new source code.

## Pinned baseline and upstream revision

| Item | Revision |
|---|---|
| Validated baseline before the first upstream sync | [`loop-baseline-2026-10-11`](https://github.com/bebetterest/transformers_loop/tree/loop-baseline-2026-10-11) |
| Baseline fork commit | `e709e88b9d8db18a1648c023ba9d10f732dd2237` |
| Baseline upstream commit | `e598fbad926d80bc356c0c4d96532030e3dfdb62` |
| Upstream target for the 2026-10-11 sync | [`536ecc007387a50e77603bb5d92100e9b07514cc`](https://github.com/huggingface/transformers/commit/536ecc007387a50e77603bb5d92100e9b07514cc) |

The target adds 41 upstream commits. It includes configuration/subconfig changes, cache and compiled-mask fixes,
distributed updates, and new models. Its minimum supported PyTorch version is 2.6. New upstream model definitions do
not automatically gain loop support: they must satisfy the decoder execution protocol and have an appropriate adapter.
Constant hidden width alone is insufficient for models with architecture-specific state or input/output transforms.

The baseline's full CUDA and TRL matrices are recorded in the [loop guide](loop_readme.md#recorded-trl-validation) and
the [detailed feature documentation](docs/source/en/layer_execution.md). Synchronization results below apply to the
merged source and list the checks rerun for that update.

## 2026-10-11 synchronization record

The integration merge is `8f4c311922049361d57b35f50312b773be58c893`, with the baseline fork and reviewed upstream target
as its two parents. The loop modules remain unchanged by this merge. The affected upstream hooks are covered by targeted
regressions. All 4,776 Python source/test files were hashed and checked against the copy sent to the GPU server.

Local validation uses Python 3.11.15 and PyTorch 2.14.1:

| Check | Passed | Skipped |
|---|---:|---:|
| Loop plans, adapters, caches, gradients, generation, PEFT, CPU DDP/FSDP2/TP/PP, and independent local launchers | 123 | 88 |
| Trainer save/reload and resume | 10 | 0 |
| Affected upstream configuration, weight conversion/loading, and masking | 31 | 1 |

The loop suite's skips require CUDA, optional packages, or other unavailable runtime conditions. The upstream
configuration fixture was prefetched into a writable cache and its initially failing offline check then passed.
`make check-repo` and `make style` pass. The optional reviewer-assignment resolver is absent, so that check reports a
skip; this synchronization does not create an upstream PR. The model-date checker passed once the merge was committed
and the incoming models' original Git history became reachable.

GPU validation uses two NVIDIA H20 GPUs on one physical host, driver 580.82.07, CUDA 13.0, NCCL 2.29.7, Python 3.12.3,
and PyTorch 2.12.1+cu130. Optional packages are unmodified TRL 1.15.0, PEFT 0.21.2, Accelerate 1.15.0, DeepSpeed 0.19.7,
and bitsandbytes 0.50.2. The run has a separate directory and does not overwrite the baseline's evidence.

The native loop suite passes 204 unique tests, with 7 skips and 8 additional passing subtests. It covers CUDA caches,
shared gradients, compiled execution, DDP, FSDP2, tensor parallelism, standalone pipeline parallelism, and independent
local launchers. Three skips require PyTorch 2.14 for CPU FSDP2; four require the optional `optimum-quanto` or `hqq`
cache backends, which are absent from this server environment. One tensor-parallel launcher initially failed before
model startup because its rendezvous port was already in use. That exact test passed on an isolated rerun with a fresh
port; both attempts are retained.

Three single-GPU TRL checks pass: tiny Llama SFT/LoRA with the whole stack repeated, tiny Qwen3.5 DPO/full with a range
repeated, and tiny Qwen3.5 GRPO/LoRA with the second layer repeated. They check fused loss/log-probability computation,
shared gradients against independently expanded native layers, reference invariance, independent generation caches,
model/adapter reload, optimizer restore, and resumed training against uninterrupted controls. The first SFT launch
exposed both GPUs to a plain Python process and hit TRL's DataParallel guard. The corrected single-GPU launch uses
`CUDA_VISIBLE_DEVICES=0`, as explained in the [loop guide](loop_readme.md#trl-integration-status); the failed attempt is
retained.

The selected two-GPU TRL matrix passes all 25 configurations: 50 successful process phases and 100 independently
checked rank reports. This is a regression selection for the merged source; it does not claim that the previous full
87-configuration matrix was rerun.

| Models and plan | Selected coverage | Configurations |
|---|---|---:|
| Tiny Qwen3.5, second layer repeated | SFT: full/DDP, LoRA/ZeRO-2, QLoRA/ZeRO-3; cold resume and uninterrupted controls | 3 |
| Tiny Qwen3.5, second layer repeated | DPO: full/ZeRO-2, LoRA/ZeRO-3, QLoRA/DDP; cold resume and uninterrupted controls | 3 |
| Tiny Qwen3.5, second layer repeated | GRPO: full/ZeRO-3, LoRA/DDP, QLoRA/ZeRO-2; cold resume and uninterrupted controls | 3 |
| Tiny Qwen3.5, nonzero frozen reference adapter | DPO/LoRA/ZeRO-2 and DPO/QLoRA/ZeRO-3; cold resume and uninterrupted controls | 2 |
| Tiny Llama, second layer repeated | QLoRA with SFT/DDP, DPO/ZeRO-2, and GRPO/ZeRO-3; two training steps | 3 |
| Tiny Qwen3.5, range or whole stack repeated | SFT/QLoRA with DDP, ZeRO-2, and ZeRO-3; two training steps | 6 |
| Original Qwen3.5-9B, second layer repeated | SFT/full/DDP, DPO/LoRA/ZeRO-2, and GRPO/QLoRA/ZeRO-3; cold resume | 3 |
| Original Qwen3.5-9B, range or whole stack repeated | Range/SFT/LoRA/ZeRO-3 and whole-stack/SFT/QLoRA/DDP; two training steps and four generation cache modes | 2 |

The checks confirm execution plans and physical parameter identities, complete finite shared gradients, actual
parameter updates, agreement between ranks, AdamW moments/steps where applicable, independent GRPO rollout caches,
reference invariance, and exact cold checkpoint/optimizer restoration. The 11 tiny checkpoint configurations include
uninterrupted controls; an independent local audit of all 22 rank parameter-file pairs found a maximum difference of
zero. The original-weight cases use `Qwen/Qwen3.5-9B` revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`, with original
file sizes and SHA-256 hashes verified. Original full-parameter 9B runs use SGD; tiny and adapter runs use AdamW.
The two original-weight range/whole-stack cases compare dynamic, static, offloaded, and offloaded-static greedy
generation. These results do not validate every combination of trainer, plan, cache, precision, and parallel backend.

After the first 20 configurations completed, SSH disconnected. On reconnection the validation processes were absent
and the first 9B attempt had no completion receipt; the cause of termination is unknown. The completed results and
interrupted attempt are retained. The remaining five original-weight configurations passed in a separate run directory
after rechecking source, dependencies, GPUs, and model inputs against their pre-interruption values. The incomplete
attempt is excluded from passing counts. All 4,776 Python source/test hashes were also verified after the completed run.
Raw records, validation scripts, and tensor evidence were copied back with a SHA-256 inventory; environments and model
weights remain outside Git. The previous full CUDA/TRL matrices remain attached to the fixed baseline tag.

## Upstream synchronization procedure

1. Start from a clean `main`, fetch upstream, and choose a specific commit. Review the upstream diff, especially
   configuration, model loading, masks, generation, caches, Trainer, and distributed code. Record the target SHA before
   validation; do not change it halfway through the run.
2. Tag the existing validated fork revision and create a `codex/` integration branch. Merge upstream normally so its
   authors and history are retained. Resolve conflicts while preserving the opt-in loop hooks and native execution.
3. Run the local feature regressions and tests for the affected upstream paths. Run CUDA and distributed checks on
   suitable hardware when those paths are affected. Keep source/dependency versions and passing, failing, and skipped
   results together. Fix genuine regressions and rerun the affected checks.
4. Run `make check-repo`, then `make style`. Inspect generated or automatically fixed files. Stage only the intended
   files, inspect the staged diff, and commit with the configured user as the main author.
5. Merge the validated branch back to `main` and push `main` and the explicit baseline tag. Use normal merges and
   pushes; preserve published commits and existing baseline tags. Verify the remote commit and a clean worktree.

Example setup, using your chosen date and reviewed upstream SHA:

```bash
git switch main
git status --short
# Add this remote once, if it is not already configured.
git remote add upstream https://github.com/huggingface/transformers.git
git fetch upstream
git tag -a loop-baseline-YYYY-MM-DD -m "Validated loop baseline before upstream sync"
git switch -c codex/sync-upstream-YYYY-MM-DD
git merge --no-ff --no-commit REVIEWED_UPSTREAM_SHA
```

Do not run the merge with uncommitted work. During a merge, upstream changes are already staged; inspect them along
with any explicit conflict fixes. Avoid broad staging of unrelated files, environments, downloaded weights, or raw
reports. Merge and maintenance commits retain the user's Git author and include the requested model co-author footer.

Sync in manageable batches or around useful upstream fixes/releases. Re-run validation when changing a dependency
that controls execution, cache allocation, checkpoint loading, or gradient reduction. There is no automatic sync job.

## Required behavior checks

The check set follows the affected paths; it is not a claim that every upstream architecture is tested.

| Area | What to verify |
|---|---|
| Plans and adapters | Single-layer, block-range, and whole-stack repeats; shared original parameter identities and state-dict keys; native forward when the plan is disabled |
| Configuration and loading | Original weights load without expansion; plans/options survive config save/reload and nested text configs; custom adapter registration is available in every worker |
| Caches and generation | Every logical execution owns independent KV/recurrent state; compare prefill/decode with independently expanded native layers; exercise supported static/offloaded/quantized backends and cache mutation |
| Training and checkpoints | Shared gradients accumulate every repeated use; checkpoint recomputation and optimizer updates match references; model, adapter, frozen reference, and optimizer states reload/resume correctly |
| Distributed execution | Consistent plans across ranks; DDP/FSDP2/TP/standalone PP gradients, generation, ownership, and checkpoint behavior; distinct local/global rank behavior in simulated launch checks |
| TRL / Accelerate / ZeRO | Unmodified package imports and actual SFT/DPO/GRPO runs; full/LoRA/NF4 QLoRA, reference stability, independent rollout caches, gradient completion, and checkpoint/resume on selected DDP/ZeRO-2/ZeRO-3 paths |
| Native upstream behavior | Targeted config/loading/masking/Trainer regressions without an enabled loop, plus repository consistency and code style |

Run the feature suite from a development environment with the repository's test dependencies installed:

```bash
PYTHONPATH=src HF_HUB_OFFLINE=1 python -m pytest tests/test_layer_execution*.py -q
make check-repo
make style
```

Use a writable Hugging Face cache and prefetch any fixtures required by the selected upstream tests. A missing
offline fixture is an environment failure to resolve, not a passing result. Record test skips and distinguish
unavailable hardware/dependencies from unimplemented integrations. Keep raw logs and large checkpoints outside Git;
publish a concise version-bound validation summary here.

For GPU checks, record GPU model/count, driver, CUDA/NCCL, Python, PyTorch, TRL, PEFT, Accelerate, DeepSpeed, and
bitsandbytes versions. For original-weight checks, pin the model revision and validate the downloaded shards. Use
separate run directories so an upstream sync cannot overwrite the baseline's evidence.

## Validation limits

Short synthetic runs establish execution and training mechanics. They do not establish model quality, convergence,
production throughput, or long-run reliability. Floating-point differences must be assessed against independent
native execution with appropriate tolerances, especially for BF16/NF4 and tensor-parallel reductions.

Two GPUs and two independent local launchers still use one physical host. They do not verify physical multi-host
networking, separate filesystems, IB/RoCE, elastic restarts, or failure recovery. These require a real multi-host run.
vLLM integration remains deferred, including its Transformers backend and TRL server/colocate modes; GRPO uses
`use_vllm=False`. Other unverified combinations remain listed in the [loop guide](loop_readme.md#recorded-trl-validation).
