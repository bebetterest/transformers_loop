<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Configurable decoder loops

This checkout can load an original checkpoint and execute selected decoder layers more than once. A configurable
execution plan controls the order. Repeated visits share the original parameters, while every logical execution
position owns its own cache history. Both `model(...)` and Hugging Face `model.generate(...)` follow the plan.

The feature is opt-in. Models without an enabled plan use their native forward. Extra executions increase computation,
cache storage, and potentially training activation memory; they do not guarantee better model quality.

Cached decoding and recomputing the entire prefix can use different floating-point kernels. With BF16/NF4 weights,
small score differences can change a greedy token when candidates have close scores. A two-GPU 9B QLoRA check
observed this after training; both paths matched independently expanded native layers exactly, including every
generation score. Token equality is therefore checked on the tested examples, rather than promised for every input.

For the complete API, custom adapters, cache backends, and distributed validation details, see
[Layer execution plans](docs/source/en/layer_execution.md).

- [Quick start](#quick-start-repeat-the-second-layer)
- [Configure the order](#configure-the-execution-order)
- [Execution and cache behavior](#how-the-loop-works)
- [Training and checkpoints](#training-saving-and-reloading)
- [TRL references and LoRA](#trl-references-and-lora-adapter-checkpoints)
- [Support and validation](#supported-architectures-and-integrations)
- [TRL integration status](#trl-integration-status)
- [Accelerate integration status](#accelerate-integration-status)
- [vLLM integration status](#vllm-integration-status)
- [Implementation and extensions](#implementation-map-and-extension-points)

## Install this checkout

Clone the loop-enabled fork:

```bash
git clone https://github.com/bebetterest/transformers_loop.git
cd transformers_loop
```

Run this from the repository root in your Python environment:

```bash
python -m pip install -e ".[torch]"
```

The APIs below require this checkout or another version containing the layer-execution implementation. Installing a
separate upstream release does not make it support this checkout's plans. PEFT and TRL are optional dependencies for
their respective examples.

## Quick start: repeat the second layer

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, RepeatRange

checkpoint = "Qwen/Qwen3.5-9B"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint, dtype="auto", device_map="auto")

model.set_layer_execution_plan(repeats=[RepeatRange(start=1, stop=2, total_passes=2)])
print(model.get_layer_execution_plan().layer_order)
# (0, 1, 1, 2, ..., 31): 32 physical layers, 33 logical executions.

inputs = tokenizer("Explain how shared parameters work.", return_tensors="pt").to(model.device)
model.eval()
with torch.no_grad():
    tokens = model.generate(**inputs, max_new_tokens=32)
print(tokenizer.decode(tokens[0], skip_special_tokens=True))
```

The device map above is for inference. For training, let your trainer or distributed setup place the model.
The original checkpoint can also be a local directory. There is no need to duplicate or expand its weight tensors.

## Configure the execution order

Indices are **zero-based**. Ranges are half-open: `start` is inclusive and `stop` is exclusive.
`total_passes=2` means two executions in total, including the first pass. Range repetition repeats the range as a
block, rather than repeating each layer individually.

| Intended behavior | Plan on a four-layer decoder |
|---|---|
| Second layer twice | `0, 1, 1, 2, 3` |
| Second through third layers twice as a block | `0, 1, 2, 1, 2, 3` |
| Whole stack twice | `0, 1, 2, 3, 0, 1, 2, 3` |

```python
from transformers import RepeatRange

num_layers = model.config.get_text_config().num_hidden_layers

# Repeat the second through fourth layers as a block.
model.set_layer_execution_plan(repeats=[RepeatRange(1, 4, total_passes=2)])

# Repeat the complete stack twice.
model.set_layer_execution_plan(repeats=[RepeatRange(0, num_layers, total_passes=2)])

# Multiple non-overlapping ranges; earlier calls above are replaced, not accumulated.
model.set_layer_execution_plan(repeats=[RepeatRange(1, 2, 3), RepeatRange(3, 4, 2)])

# Explicit orders can repeat, reorder, or omit source layers.
model.set_layer_execution_plan([0, 1, 1, *range(2, num_layers)])
print(model.get_layer_execution_plan().layer_order)

# Restore native decoder execution using the current weights.
model.set_layer_execution_plan(None)
```

Each setter call replaces the complete plan. The getter returns an immutable snapshot. Use the setter when changing a
loaded model; editing `config.json` takes effect on the next load. Disabling a plan does not undo training updates.

Change plans only between complete forward/backward steps. Discard existing caches and recompute the prompt after
any change. Stop continuous batching before reconfiguring the plan.

## How the loop works

For the plan `[0, 1, 1, 2]`, the decoder computes:

```text
embeddings -> layer 0 -> layer 1 -> layer 1 -> layer 2 -> final norm -> output head
              cache 0    cache 1    cache 2    cache 3
```

Both visits to layer 1 call the **same registered module and parameters**, with different incoming hidden states and
cache histories. Embeddings and stack input projections run once; final normalization, output projections, and the
LM head run once after the complete plan. Token positions advance once per decoder forward, rather than once per
loop pass. Depth repetition does not consume additional sequence positions.

The executor replaces the decoder's forward with plan dispatch and calls each original block through `layer(...)`.
This preserves its module hooks, parameter identities, optimizer registration, and checkpoint weight keys. The source
layer count remains unchanged; `output_hidden_states=True` follows logical execution order.

Autograd connects every repeated call. All contributions accumulate into the shared source parameters, including
gradients flowing through earlier loop passes. There are no detached loop boundaries or extra optimizer parameters.
An execution plan changes computation, so loading unchanged original weights does not imply unchanged outputs.

## Cache behavior

The default policy gives every logical position independent KV, convolution, recurrent, and custom state histories.
Cache position 1 and cache position 2 above remain separate despite sharing layer 1's weights. Subsequent tokens reuse
the history of their corresponding execution position. Separate generation requests own separate caches.

`generate` creates a `LayerExecutionCache` automatically. For manual decoding:

```python
from transformers import LayerExecutionCache

model.set_layer_execution_plan(repeats=[RepeatRange(1, 2)])
model.eval()
cache = LayerExecutionCache(model.config)
with torch.no_grad():
    prefill = model(**inputs, past_key_values=cache, use_cache=True)
    # Feed subsequent token IDs with the same cache and the appropriate attention mask.
```

Keep a cache attached to its original plan. Ordinary `DynamicCache` instances and caches from incompatible plans are
rejected. Supported backends include dynamic, static, offloaded, offloaded-static, and quantized caches. Quantization
applies to supported attention KV storage; recurrent and sliding states retain native precision. The portable Torch
quantized backend and optional Quanto/HQQ backends have different dependencies. Cache quantization does not provide
weight quantization or QLoRA support by itself. See the [cache API](docs/source/en/layer_execution.md#independent-cache-states).

Architectures that require native cross-layer KV sharing must select `kv_sharing="native"` on a `LayerExecutionPlan`.
This preserves their required producer/consumer dependency; every repeated producer visit still has an independent
history. Gemma3n requires this policy. Missing or invalid producers are rejected before enabling the plan. See
[native KV dependencies](docs/source/en/layer_execution.md#native-cross-layer-kv-dependencies).

## Training, saving, and reloading

Gradient-enabled training disables caching and requires `past_key_values=None`. Non-reentrant gradient checkpointing
supports repeated layer calls:

```python
model.train()
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
loss = model(input_ids=input_ids, labels=labels, use_cache=False).loss
loss.backward()
```

Provide `input_ids` and task-appropriate `labels` in this training example. For generation while the policy stays in
training mode, `torch.no_grad()` or `torch.inference_mode()` permits independent execution caches. Temporarily disable
gradient checkpointing during generation and restore it for training. The validated TRL GRPO helper performs that
transition.

An execution plan is configuration, rather than a new set of weights. `save_pretrained` writes `layer_execution_plan`
and nondefault `layer_execution_options` into the text configuration. `from_pretrained` restores the plan automatically
when using this implementation:

```python
model.save_pretrained("loop-checkpoint")
tokenizer.save_pretrained("loop-checkpoint")
reloaded = AutoModelForCausalLM.from_pretrained("loop-checkpoint", dtype="auto")
assert reloaded.get_layer_execution_plan() == model.get_layer_execution_plan()
```

This saves model weights and configuration. Use your trainer's checkpoint/resume API to save and restore optimizer,
scheduler, RNG, and training progress as well.

## TRL references and LoRA adapter checkpoints

**An in-memory plan is not automatically copied into models loaded from the original checkpoint.** A TRL trainer may
load its reference model from the policy's base checkpoint path. If that path still points to the original model,
the reference may have no loop even though the policy has one. Likewise, **saving only a PEFT adapter does not save
an in-memory base execution plan**. The adapter records its base model path; automatic reload follows that path.

When the policy, automatic reference, and adapter reload should all use the same plan, save and reload the planned
base **before** creating the trainer or adapter:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, RepeatRange

checkpoint = "Qwen/Qwen3.5-9B"
base = AutoModelForCausalLM.from_pretrained(checkpoint, dtype="bfloat16")
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
base.set_layer_execution_plan(repeats=[RepeatRange(1, 2, total_passes=2)])
base.save_pretrained("planned-base")
tokenizer.save_pretrained("planned-base")
del base

policy = AutoModelForCausalLM.from_pretrained("planned-base", dtype="bfloat16")
# Pass policy to SFTTrainer, DPOTrainer, or GRPOTrainer, with your dataset and configuration.
# For GRPO's validated generation path, configure use_vllm=False.
```

For standalone LoRA creation and automatic reload:

```python
from peft import AutoPeftModelForCausalLM, LoraConfig, get_peft_model

lora = get_peft_model(
    policy,
    LoraConfig(task_type="CAUSAL_LM", r=4, lora_alpha=8, target_modules=["down_proj"]),
)
lora.save_pretrained("loop-adapter")
restored = AutoPeftModelForCausalLM.from_pretrained("loop-adapter", dtype="bfloat16")
assert restored.get_layer_execution_plan() == lora.get_layer_execution_plan()
```

Keep the planned base accessible at the path or repository ID recorded in the adapter configuration. Moving only
the adapter does not move its base checkpoint. Alternatively, explicitly restore the base plan before loading the
adapter with `PeftModel.from_pretrained`, and explicitly construct the DPO reference when needed. Policy and reference
plans may intentionally differ; the module does not guess the desired reference computation.

The standalone `get_layer_execution_plan(model)` and `set_layer_execution_plan(model, ...)` functions also accept
standard PEFT and distributed containers. The setter returns the same container. Configure a plan before constructing
the trainer, and use `gradient_checkpointing_kwargs={"use_reentrant": False}` in its configuration.
TRL 1.15.0's GRPOTrainer sets reentrant checkpointing for PEFT with ZeRO-3 during construction. Its generation utility
later re-enables checkpointing without keyword arguments, selecting this checkout's non-reentrant default. The
recorded checks use those ordinary TRL utilities. For a custom generation context, pass your saved checkpointing
keyword arguments explicitly when re-enabling it.

### NF4 / QLoRA loading and reload

Save the planned, unquantized base first. Then quantize that base and add the adapter through the usual PEFT APIs.
An execution plan does not change which weights bitsandbytes quantizes or which parameters PEFT trains.
For distributed loading, place each rank's quantized model on its own GPU instead of using `device_map="auto"`:

```python
import os
import torch
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

quantization = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_storage=torch.bfloat16,
)

def load_quantized_base():
    base = AutoModelForCausalLM.from_pretrained(
        "planned-base",
        dtype=torch.bfloat16,
        quantization_config=quantization,
        device_map={"": int(os.environ.get("LOCAL_RANK", "0"))},
    )
    return prepare_model_for_kbit_training(
        base, gradient_checkpointing_kwargs={"use_reentrant": False}
    )

policy = get_peft_model(
    load_quantized_base(),
    LoraConfig(task_type="CAUSAL_LM", r=4, lora_alpha=8, target_modules=["down_proj"]),
)
# Train through your TRL trainer and its distributed configuration.
policy.save_pretrained("loop-qlora-adapter")
restored = PeftModel.from_pretrained(load_quantized_base(), "loop-qlora-adapter").eval()
assert restored.get_layer_execution_plan() == policy.get_layer_execution_plan()
```

The BF16 quantization storage matches the loading dtype and allows ZeRO-3 to partition packed NF4 parameters.
Quantization storage and KV cache storage are separate settings. Keep quantization settings, compute dtype, base
preparation, and the backend's precision policy consistent when reloading. In particular,
`prepare_model_for_kbit_training` promotes non-quantized weights to FP32; a plain `AutoPeftModelForCausalLM` reload
does not repeat that preparation. DeepSpeed may subsequently cast model parameters according to its BF16 setting.
When reloading a DeepSpeed-trained adapter, prepare it through the same engine and precision configuration.
DeepSpeed 0.19.7 preserves FP32 buffers by default, including RoPE frequencies; a blanket `model.bfloat16()` also
rounds those buffers and therefore does not reproduce that precision policy.
Use the trainer's checkpoint/resume API for optimizer, scheduler, RNG, and training progress; an adapter directory
alone contains none of those states.

Trainer checkpoint loading preserves the dtype of an already configured PEFT adapter. This matters when training
with BF16 adapters: promoting them to FP32 during resume also changes AdamW's arithmetic, even if the restored
values are initially identical. Loading a new adapter retains PEFT's default precision policy. A standalone
`PeftModel.from_pretrained` call remains governed by PEFT's own `autocast_adapter_dtype` option.
Multi-adapter Trainer resume restores both PEFT's root-level `default` adapter and named adapter directories,
including TRL's frozen `ref` adapter. Best-checkpoint loading selects the active adapter's directory.
With multiple PEFT adapters, DeepSpeed checkpoints also retain frozen parameters so that reference adapters survive
cold resume. DeepSpeed's exclusion flag applies to every frozen parameter, so these training checkpoints can include
the frozen base and be larger than single-adapter checkpoints. ZeRO-3 adapter exports also gather the reference
weights rather than emitting an empty frozen adapter. The single-adapter optimization remains enabled.

Trainer model checkpoints use the model's current `state_dict` key format so that direct training-state reload can
restore every parameter. Exporting through `model.save_pretrained` can retain a checkpoint's original external
key format; load those exports with `from_pretrained`, which applies the model's weight conversions.

For a two-GPU TRL job, launch the training script with `torchrun --standalone --nproc_per_node=2 train_loop.py`.
Use the same planned base and plan on both ranks. DDP needs no additional loop-specific configuration. For
DeepSpeed, pass its configuration through the trainer configuration's `deepspeed` field; for example:

```python
from trl import SFTConfig, SFTTrainer

config = SFTConfig(
    output_dir="loop-training",
    bf16=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    deepspeed={
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "gradient_accumulation_steps": "auto",
        "gradient_clipping": "auto",
        "bf16": {"enabled": "auto"},
        "zero_optimization": {
            "stage": 3,  # Use 2 for ZeRO-2.
            "stage3_gather_16bit_weights_on_model_save": True,
        },
    },
)
trainer = SFTTrainer(model=policy, args=config, train_dataset=dataset, processing_class=tokenizer)
trainer.train()
# Later, construct the same policy/configuration and call:
# trainer.train(resume_from_checkpoint="loop-training/checkpoint-...")
```

The policy in this example is loaded before constructing the trainer configuration. This is also the loading order
used by the distributed TRL validation; the engine partitions it during trainer preparation. ZeRO-3 initialization
during `from_pretrained`, CPU/NVMe optimizer offload, and other DeepSpeed configurations require their own checks.

Trainer installs the ZeRO-3 shared-gradient integration automatically. Repeated calls can produce several gradient
contributions for one source parameter; the integration lets them accumulate before invoking ZeRO's normal gradient
reduction and partitioning. It preserves source parameters, optimizer groups, and independent generation caches.
Manual DeepSpeed users configure the initialized engine explicitly:

```python
import deepspeed
from transformers.layer_execution.deepspeed import configure_deepspeed_layer_execution

# Set the model's plan before initializing DeepSpeed.
# deepspeed_config contains resolved batch sizes and optimizer settings.
engine, optimizer, _, _ = deepspeed.initialize(model=policy, config=deepspeed_config)
configure_deepspeed_layer_execution(engine)
```

Use `set_layer_execution_plan(engine, ...)` to reconfigure an existing engine between complete steps. This integration
is verified with DeepSpeed 0.19.7; other releases need validation of their gradient-reduction API and hook ordering.

Prefer DDP for multi-GPU Qwen3.5 training with the reference PyTorch recurrent kernels. On the tested PyTorch build,
the first CUDA linear algebra call can race between `DataParallel` replica threads, including without a loop plan.
If using DataParallel, perform a single-threaded, no-gradient forward with `use_cache=False` before training to
initialize those kernels. The DataParallel regression test uses this initialization.
See the [upstream PyTorch initialization issue](https://github.com/pytorch/pytorch/issues/90613).

## Supported architectures and integrations

The target is a **causal decoder-only stack whose prepared state streams keep the same shapes and structure across
layers**. Embedding width may differ if the adapter handles one-time input/output projections. Encoder stacks,
encoder-decoder models, and changing-width decoder blocks are excluded. Constant width alone does not establish
compatibility: a new architecture needs an adapter for its masks, layer arguments, cache protocol, and outputs.

| Area | Current support and boundaries |
|---|---|
| Built-in model adapters | Llama, dense Qwen3.5 text, Gemma3n text with native KV dependencies and AltUp streams, and RecurrentGemma with execution-local recurrent state |
| Other decoders | Register a custom adapter. GPT-2, OPT, and Mamba are covered by extension examples/tests; they are not automatically registered built-ins. |
| Multimodal wrappers | A plan can target a supported text backbone. Vision/audio encoders are not looped. |
| Native training and distributed execution | Existing checks cover DataParallel, DDP, FSDP2, native TP, and standalone PP on CPU/two H20 GPUs, including original Qwen3.5-9B weights. |
| TRL / PEFT / DeepSpeed | Unmodified TRL runs SFT/DPO/GRPO with Hugging Face generation, shared-gradient ZeRO-3 integration, and PEFT training-state resume through this checkout's Trainer. See the recorded matrix below for tested combinations and versions. |
| Accelerate | Unmodified Accelerate is exercised through Trainer on single-GPU, DDP, and ZeRO-2/ZeRO-3 paths. Standalone Accelerate/ZeRO-3 loops need the explicit shared-gradient integration call described below. |
| Trainer checkpoints | Full training-state resume is checked for DDP and standalone PP. Native load-time TP/FSDP2 requires `save_only_model=True` or `save_strategy="no"`; optimizer-state resume is not supported on that path. |
| Pipeline scheduling | Source weights stay on their owning stage; state moves when the next execution changes owner. Scheduling is sequential. Overlapped GPipe/1F1B and combined parallel Trainer layouts are not integrated. |
| Generation and batching | Greedy/beam and assisted generation, portable continuous batching, and supported attention-only paged scheduling have integrations. Pipeline execution uses the portable batching manager. |
| Physical multi-host operation | Deferred. Two independent torchrun agents with Socket transport were checked on one physical host. |
| External execution engines | vLLM loop execution is not implemented or runtime-validated. Other independent executors also need their own plan integration; the Hugging Face model definition alone does not enable loops inside them. |

All distributed ranks must use the same plan, and changes must be synchronized between complete training steps. If an
explicit order omits source layers, manual DDP needs `find_unused_parameters=True`; `Trainer` selects it for the
initial plan unless overridden. Prefer non-reentrant checkpointing. Explicit reentrant DDP checkpointing needs a fixed
execution graph with `static_graph=True`.

The [detailed distributed guide](docs/source/en/layer_execution.md#distributed-execution) describes validated paths,
checkpoint restrictions, manual TP optimizer groups, pipeline input requirements, and numerical tolerances.

## TRL integration status

**The validated paths require no changes to TRL's source.** They use TRL 1.15.0 with this Transformers checkout.
The loop runs inside the model's forward/generation methods and keeps the original parameter registrations, so TRL
can use its ordinary trainers, losses, PEFT adapters, and optimizers. Plan dispatch and execution-local caches live in
`transformers.layer_execution`; checkpoint restoration and automatic ZeRO-3 shared-gradient setup live in this
checkout's `transformers.Trainer`. Installing TRL alongside an upstream Transformers release does not supply these changes.

The [recorded validation](#recorded-trl-validation) covers SFT, DPO, and GRPO with Hugging Face generation on Llama and
Qwen3.5, including original Qwen3.5-9B. Single-H20 runs cover full-parameter and LoRA training; the two-H20 matrix
adds NF4 QLoRA, DDP, ZeRO-2/ZeRO-3, checkpoint resume, and frozen-reference checks in the listed combinations.
Original full-parameter 9B runs use SGD; tiny and adapter runs use AdamW. These are short integration checks.

For use with TRL:

- Load a supported model through this checkout and set its plan before constructing the trainer. Preserve the planned base for automatic reference-model and adapter reloads, as described in [TRL references and LoRA](#trl-references-and-lora-adapter-checkpoints).
- Use the normal trainer configuration and non-reentrant gradient checkpointing. For GRPO, set `use_vllm=False`; [TRL's configuration](https://huggingface.co/docs/trl/en/grpo_trainer) then uses Hugging Face generation. Both vLLM modes require the separate integration described below.
- For DDP, use the same plan on every rank. For ZeRO-2/ZeRO-3, pass the usual DeepSpeed configuration to the trainer; this checkout installs the shared-gradient integration automatically. Follow the NF4 loading and precision rules above for QLoRA.

Other TRL trainers, TRL FSDP/QLoRA combinations, packing/padding-free training, Liger/Unsloth, and other package
versions require separate validation. Native FSDP2/TP/PP checks do not establish the corresponding TRL combinations.

## Accelerate integration status

**No changes to Accelerate's source were needed for the recorded paths.** Accelerate 1.15.0 was exercised through
Transformers/TRL Trainer on single-GPU, two-GPU DDP, and DeepSpeed ZeRO-2/ZeRO-3 jobs. It manages placement,
distributed wrapping, mixed precision, and backward execution; the underlying Transformers model executes the loop.
The same shared parameters remain visible to the optimizer and distributed wrappers.

Configure the model's plan before Trainer or `Accelerator.prepare(...)`, use the same plan on every rank, and let the
selected backend place the model. Reserve `device_map="auto"` for inference; distributed NF4 loading uses an explicit
rank-local map as shown above. The recorded GPU jobs used `torchrun`. An ordinary DDP script can also be launched with
`accelerate launch --multi_gpu --num_processes=2 train_loop.py`; choose its backend through the normal Accelerate
configuration. The launch command does not add loop support to an independent inference engine.

For a standalone single-GPU/DDP loop, the [usual Accelerate API](https://huggingface.co/docs/accelerate/en/quicktour)
applies. This is a usage recipe; the recorded GPU matrix exercises Accelerate through Trainer rather than this standalone loop:

```python
from accelerate import Accelerator
from transformers import RepeatRange, set_layer_execution_plan

# Provide a supported model, its optimizer, and a DataLoader with input_ids and labels.
accelerator = Accelerator()
set_layer_execution_plan(model, repeats=[RepeatRange(1, 2, total_passes=2)])
model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)

model.train()
for batch in train_dataloader:
    optimizer.zero_grad()
    loss = model(**batch, use_cache=False).loss
    accelerator.backward(loss)
    optimizer.step()
```

For a **standalone Accelerate/DeepSpeed ZeRO-3** loop, call
`configure_deepspeed_layer_execution(model)` from `transformers.layer_execution.deepspeed` on the engine returned by
`accelerator.prepare(...)`, before the first training forward. That explicit call installs the shared-gradient
coordinator; Trainer performs it automatically. `Accelerator.prepare(...)` alone does not install this checkout's
coordinator. Use the backend's checkpoint API for full training state and preserve the model's plan configuration
when reconstructing the model for resume.

The Trainer matrix does not validate every Accelerate plugin or launcher. Standalone Accelerate FSDP/TP/PP,
Megatron-LM, FP8, TPU/other accelerators, combined parallel layouts, and physical multi-host operation require their
own integration or validation. Native PyTorch FSDP2/TP/PP results are recorded separately.

## vLLM integration status

**Loop execution in vLLM is not implemented or runtime-validated in this checkout.** This includes native vLLM model
implementations, `model_impl="transformers"`, and TRL's vLLM server and colocate modes. For the current implementation,
use Hugging Face `model.generate(...)` and configure TRL GRPO with `use_vllm=False`. Loading a checkpoint containing
an execution plan into vLLM does not establish that generation follows the plan or keeps independent execution states.

Native vLLM models use their own forward paths. The Transformers backend also substitutes vLLM attention and manages
its cache allocation. In the inspected [vLLM 0.31.0 backend](https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/model_executor/models/transformers/base.py),
attention instances are created for the original physical layers; repeating a source module does not create separate
cache histories for its logical execution positions. Qwen3.5 additionally needs independent GatedDeltaNet recurrent
states, as reflected in its [native vLLM implementation](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen3_5.py).

A future vLLM adapter would need to:

- Read the saved plan, execute its logical order, and preserve original shared weights during loading and training-weight synchronization.
- Allocate and route paged KV caches and recurrent states by logical execution position, including hybrid layer types.
- Handle the plan in tensor/pipeline parallel execution and graph compilation, then validate offline generation, serving, and TRL server/colocate modes independently.

The existing Transformers plan format can be reused. vLLM supports
[out-of-tree model registration](https://docs.vllm.ai/en/latest/contributing/model/registration/#out-of-tree-models),
so an external adapter may avoid maintaining a vLLM fork if its extension points cover the required state and
execution behavior. Otherwise, vLLM engine changes would also be necessary. This integration is deferred.

## Recorded TRL validation

On 2026-10-10, single-H20 validation used TRL 1.15.0, PEFT 0.21.2, Accelerate 1.15.0, and PyTorch 2.12.1+cu130.
The checks used short synthetic data and rewards, and establish training mechanics rather than convergence or quality.

| Model and training mode | Validation | Passing configurations |
|---|---|---:|
| Tiny Llama/Qwen3.5, full and LoRA | SFT/DPO/GRPO, three repeat plans, SGD | 36 |
| Tiny Llama/Qwen3.5, full and LoRA | SFT/DPO/GRPO, second layer twice, default fused AdamW state restore | 12 |
| Original Qwen3.5-9B, LoRA | SFT/DPO/GRPO, three repeat plans, SGD | 9 |
| Original Qwen3.5-9B, full | SFT/DPO/GRPO, second layer twice, SGD | 3 |

These 60 configurations checked execution order, shared gradients, independent GRPO caches, reference-model stability,
cached/uncached greedy agreement, checkpoint reload, and training resume. Tiny models also compared gradients with
independently expanded native layers and resumed updates with uninterrupted training. Default AdamW moment and step
states were checked on tiny models; full-parameter 9B AdamW was not tested. GRPO used `use_vllm=False`.

The two-H20 follow-up uses the same TRL/PEFT/Accelerate/PyTorch versions, DeepSpeed 0.19.7, and bitsandbytes 0.50.2.
Both GPUs are on one physical host. Its completed primary matrix is:

| Model and plan | Training modes and backends | Passing configurations |
|---|---|---:|
| Tiny Qwen3.5, second layer twice | SFT/DPO/GRPO × full/LoRA/NF4 QLoRA × DDP/ZeRO-2/ZeRO-3; cold resume and uninterrupted control | 27 |
| Tiny Llama, second layer twice | SFT/DPO/GRPO × NF4 QLoRA × DDP/ZeRO-2/ZeRO-3; training | 9 |
| Tiny Qwen3.5, range/whole-stack twice | SFT × NF4 QLoRA × DDP/ZeRO-2/ZeRO-3; training | 6 |
| Original Qwen3.5-9B, second layer twice | SFT/DPO/GRPO × full/LoRA/NF4 QLoRA × DDP/ZeRO-2/ZeRO-3; cold reload and resume | 27 |
| Original Qwen3.5-9B, range/whole-stack twice | SFT × LoRA/NF4 QLoRA × DDP/ZeRO-2/ZeRO-3; training and four unquantized cache modes | 12 |
| Tiny Qwen3.5, nonzero frozen reference adapter | DPO × LoRA/NF4 QLoRA × DDP/ZeRO-2/ZeRO-3; cold resume and uninterrupted control | 6 |

These 87 configurations cover 180 training phases and 360 passing rank records, reconciled from process exits and
both workers' evidence. Interrupted and corrected attempts are retained separately and excluded from these counts.
Full-parameter 9B runs use SGD; tiny and adapter runs use AdamW. The checks inspect complete shared gradients before
the optimizer step, actual FP32 master partitions and AdamW moments, rank agreement, execution-local rollout caches,
reference invariance, and exact cold state restoration. Independent Llama/Qwen3.5 NF4 and ZeRO regressions compare
every shared gradient and optimizer update with expanded native logical layers, including accumulation,
checkpointing, overlapping communication, and reference-adapter switches. These checks use short synthetic data and
rewards; they establish integration mechanics rather than quality, convergence, or production performance.

The four cache modes are dynamic, static, offloaded, and offloaded-static. They describe KV/recurrent storage,
separately from NF4 weight quantization. The 33 tiny cases with resumed/uninterrupted controls have exact final
parameter and optimizer summaries in this run; independent local tensor comparisons also reproduce zero parameter
difference on both ranks. The observed BF16/NF4 greedy ties are checked against independent native logical layers;
each recorded cached/recomputed path matches all of its native generation scores exactly.

Physical multi-host operation, TRL FSDP/QLoRA combinations, CPU/NVMe optimizer offload, load-time ZeRO-3
initialization, full-parameter 9B AdamW, Liger, Unsloth, packing/padding-free modes, other TRL trainers, and other
architecture combinations require separate validation. FSDP1, combined parallel layouts, and optimized recurrent
kernels also require separate integration or validation. ZeRO-3 integration is version-bounded because it relies on
DeepSpeed's gradient-readiness and reduction APIs.

## Implementation map and extension points

The plan/cache engine is separate from architecture-specific execution and optional distributed backends:

| Module | Responsibility |
|---|---|
| [plan.py](src/transformers/layer_execution/plan.py) | Immutable orders, range expansion, and logical KV dependency binding |
| [executor.py](src/transformers/layer_execution/executor.py) | Public configuration, native-forward dispatch, shared block execution, and output capture |
| [adapters.py](src/transformers/layer_execution/adapters.py) / [model_adapters.py](src/transformers/layer_execution/model_adapters.py) | Registration protocol and built-in architecture semantics |
| [cache.py](src/transformers/layer_execution/cache.py) / [backends.py](src/transformers/layer_execution/backends.py) | Execution-local views and dynamic/static/offloaded/quantized storage |
| [state.py](src/transformers/layer_execution/state.py) | Tensor pytrees and independently evolving state streams |
| [pipeline.py](src/transformers/layer_execution/pipeline.py) | Stage ownership, state transfer, autograd communication, and standalone Trainer integration |
| [deepspeed.py](src/transformers/layer_execution/deepspeed.py) | Completion of shared gradient contributions before ZeRO-3 reduction and partitioning |
| [continuous.py](src/transformers/layer_execution/continuous.py) / [paged.py](src/transformers/layer_execution/paged.py) | Request scheduling, request-local caches, and supported paged-cache routing |

Loading/configuration, generation, and Trainer hooks restore plans and route these modules without expanding the
registered decoder layers. To change repetition, use the public plan API. To support another model, implement and
register a `DecoderLayerExecutionAdapter`; its `prepare`, step execution, and `finalize` hooks preserve the model's
input transformations and output semantics. Persistent request state belongs in the execution cache, rather than on
shared layer objects. Custom adapters must be importable and registered in each process before loading a loop
checkpoint; the checkpoint stores configuration and weights, not executable adapter code.

See [Add a decoder adapter](docs/source/en/layer_execution.md#add-a-decoder-adapter) for the complete protocol and examples.
Run the feature regressions in a development environment with the repository's test dependencies installed:

```bash
HF_HUB_OFFLINE=1 python -m pytest tests/test_layer_execution*.py -q
```

Tests skip integrations whose required devices, optional packages, or PyTorch version are unavailable.
