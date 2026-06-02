# Policy-Size × Data-Budget Sweep — RUNBOOK

Holds the VLM backbone (Qwen3-VL-2B) **frozen** and trains only the flow-matching
policy at different sizes, on one LBM sim task. Metric is **closed-loop success
rate** on `lbm-eval-oss`. All commands are copy-pasteable from the repo root.

---

## 0. Environment

```bash
# project deps
uv sync
# inference + dashboard for eval
uv sync --group inference
uv sync --group dashboard
# docker + NVIDIA Container Toolkit must be installed for sim eval
```

Pick a workspace for runs and rollouts:

```bash
export EXP_ROOT=$PWD/experiments/policy_size_sweep
mkdir -p $EXP_ROOT
export TASK=BimanualPutRedBellPepperInBin   # single pick-and-place LBM task
```

---

## 1. Data acquisition (preprocessed shards, no preprocessing required)

The repo ships preprocessed LBM sim shards on a public bucket
(`vla_foundry/data/scripts/download_dataset.py` — registry `BASE_URL`
`https://vla-foundry.s3.amazonaws.com/datasets/lbm_sim/preprocessed/v0.1.0`).
No AWS creds needed.

```bash
python vla_foundry/data/scripts/download_dataset.py \
  --task $TASK \
  --local_path $EXP_ROOT/data/$TASK
```

### 1.1 Dataset sizes (probed live from the public registry)

Pulled by HEAD-ing each task's manifest + first/last shard. Numbers are
estimates (avg of first & last shard × num shards) — exact totals are a few
percent off but the relative ordering is correct. "Sequences" is the count of
(past, future) action-chunk windows in `manifest.jsonl`; the underlying demo
count is roughly `sequences / (lowdim_future_timesteps × stride)`.

| Task | shards | sequences | ~GB |
|---|---:|---:|---:|
| PushBox | 22 | 2,111 | 0.32 |
| PickAndPlaceBox | 52 | 5,175 | 1.21 |
| PutKiwiInCenterOfTable | 51 | 5,031 | 1.23 |
| PutOrangeOnSaucer | 54 | 5,324 | 1.26 |
| PutGreenAppleOnSaucer | 51 | 5,067 | 1.59 |
| PutBananaInCenterOfTable | 52 | 5,172 | 1.66 |
| PutBananaOnSaucer | 60 | 5,976 | 1.98 |
| PutOrangeInCenterOfTable | 116 | 11,503 | 2.23 |
| PutKiwiOnSaucer | 113 | 11,209 | 2.28 |
| PutGreenAppleInCenterOfTable | 146 | 14,538 | 3.72 |
| BimanualPutSpatulaOnTableFromDryingRack | 319 | 31,809 | 5.18 |
| BimanualPutSpatulaOnPlateFromTable | 201 | 20,081 | 5.19 |
| PutSpatulaInUtensilCrockFromDryingRack | 240 | 23,950 | 5.20 |
| BimanualPutSpatulaOnPlateFromDryingRack | 262 | 26,166 | 6.31 |
| BimanualLayCerealBoxOnCuttingBoardFromTopShelf | 296 | 29,554 | 6.78 |
| PutSpatulaInUtensilCrock | 283 | 28,274 | 7.04 |
| BimanualPlacePearFromBowlIntoBin | 413 | 41,216 | 7.08 |
| BimanualHangMugsOnMugHolderFromTable | 492 | 49,103 | 7.16 |
| PlaceCupByCoaster | 315 | 31,425 | 7.30 |
| PutCupInCenterOfTable | 244 | 24,365 | 7.50 |
| BimanualPutMugsOnPlatesFromTable | 366 | 36,547 | 7.79 |
| **BimanualPutRedBellPepperInBin** | **385** | **38,438** | **7.81** |
| PutMugOnSaucer | 298 | 29,742 | 7.92 |
| BimanualPutSpatulaOnTableFromUtensilCrock | 465 | 46,444 | 9.56 |
| BimanualLayCerealBoxOnCuttingBoardFromUnderShelf | 445 | 44,447 | 9.78 |
| BimanualPlaceAppleFromBowlOnCuttingBoard | 398 | 39,766 | 9.83 |
| BimanualStoreCerealBoxUnderShelf | 419 | 41,858 | 9.85 |
| BimanualPlaceAppleFromBowlIntoBin | 413 | 41,283 | 11.36 |
| BimanualHangMugsOnMugHolderFromDryingRack | 484 | 48,363 | 11.40 |
| BimanualPlaceAvocadoFromBowlOnCuttingBoard | 619 | 61,832 | 12.15 |
| PushCoasterToCenterOfTable | 428 | 42,759 | 12.64 |
| PushCoasterToMug | 477 | 47,643 | 12.77 |
| BimanualPlaceFruitFromBowlOnCuttingBoard | 759 | 75,811 | 12.83 |
| PlaceCupOnCoaster | 514 | 51,354 | 14.81 |
| BimanualPutMugsOnPlatesFromDryingRack | 706 | 70,560 | 16.57 |
| PutCupOnSaucer | 523 | 52,277 | 17.23 |
| BimanualPlaceFruitFromBowlIntoBin | 899 | 89,833 | 17.80 |
| BimanualPlacePearFromBowlOnCuttingBoard | 627 | 62,696 | 18.09 |
| TurnMugRightsideUp | 563 | 56,276 | 18.70 |
| BimanualStackPlatesOnTableFromDryingRack | 786 | 78,590 | 21.30 |
| BimanualStackPlatesOnTableFromTable | 1,133 | 113,249 | 24.50 |
| TurnCupUpsideDown | 1,003 | 100,232 | 25.49 |
| **TOTAL (all 42)** | **16,492** | **1,647,049** | **~392** |

Re-derive any time with `--dry_run`:
```bash
python vla_foundry/data/scripts/download_dataset.py --task <TaskName> --local_path /tmp/_probe --dry_run
```

Pick by trade-off:
- **Smallest sensible target**: `PickAndPlaceBox` (1.2 GB, 5K sequences) or
  `PutBananaOnSaucer` (2 GB) for fastest iteration.
- **Brief's chosen task**: `BimanualPutRedBellPepperInBin` (7.8 GB, 38K seq) —
  good mid-sized bimanual pick-and-place; matches the existing reference scripts.
- **Plenty of headroom**: anything ≥ 50K sequences for budget-sweep top end.

After download you have:

```
$EXP_ROOT/data/$TASK/
  manifest.jsonl
  stats.json
  preprocessing_config.yaml
  processing_metadata.json
  shards/
    shard_*.tar
```

> Use `--list` instead of `--task` to see all 42 available tasks. Stick to a
> single task for this study; multi-task is out of scope.

The training entry points reference these two paths explicitly:

- `--data.dataset_manifest [$EXP_ROOT/data/$TASK/manifest.jsonl]`
- `--data.dataset_statistics [$EXP_ROOT/data/$TASK/stats.json]`

---

## 2. Model configuration (per-rung)

### 2.1 Base architecture (held constant across all rungs)

We start from the released single-task recipe
`vla_foundry/config_presets/training_jobs/vla_diffusion_bellpepper.yaml`
(which composes `vla_diffusion_paligemma2.yaml` — we override the backbone on the
CLI to Qwen3-VL-2B). Then we vary **only**:

| What | How |
|---|---|
| Policy size | swap `--model.transformer "include vla_foundry/config_presets/models/transformer_{11m,100m,410m}.yaml"` |
| Backbone frozen | `--model.vision_language_backbone.freeze True` (see `vla_foundry/models/base_model.py:93` — `ModelParams.freeze` flips `requires_grad=False` on the whole submodule) |
| Backbone weights | `--model.vision_language_backbone.hf_pretrained "Qwen/Qwen3-VL-2B-Thinking"` (stock HF; see §3 for the optional "tuned backbone" path) |
| Policy init | random (default — the `transformer_*.yaml` presets don't carry `hf_pretrained`, so `create_model` builds from scratch) |

Everything else — noise scheduler, flow matching, action/proprio linear encoders,
data normalization, image augmentation, camera set, processor — comes from the
base preset and is byte-identical across rungs.

### 2.2 Why CLI overrides instead of new YAMLs

The repo has a documented YAML-include limitation (README §1.6, and the explicit
warning inside `diffusion_policy_bellpepper.yaml`): re-defining only a few fields
of an included block silently resets the rest. Overriding the whole `transformer:`
block via `include` on the CLI sidesteps it cleanly:

```bash
--model.transformer "include vla_foundry/config_presets/models/transformer_100m.yaml"
--model.transformer.is_causal False    # policy uses bidirectional attention
```

### 2.3 Size rungs

The repo ships these transformer presets (confirmed by inspecting
`vla_foundry/config_presets/models/transformer_*.yaml`):

| Preset | hidden | layers | heads | approx params (policy only) |
|---|---|---|---|---|
| `transformer_11m.yaml` | 96 | 8 | 4 | ~11M (small) |
| `transformer_100m.yaml` | 512 | 12 | 8 | ~100M (medium) |
| `transformer_410m.yaml` | 1024 | 24 | 16 | ~410M (full ≈ released policy) |

Use these three as the three policy-size rungs. (`transformer_tiny` and
`transformer_1b` are available if you want to widen the axis later.)

> Note on `vocab_size: 50432` inside the transformer presets — that field is
> only used when the transformer is the LLM head of a VLM. In `diffusion_policy`
> mode the transformer consumes VLM hidden states + a noised-action token; the
> vocab embedding table is unused. Leaving it as-is is fine. Don't set it to 0
> or you'll change initialization.

### 2.4 `action_dim` / `proprioception_dim`

Owned by the data config and injected into the model via
`DiffusionPolicyParams.init_shared_attributes` (`vla_foundry/params/model_params.py:297`).
**Do not set them manually.**

---

## 3. Loading the backbone (frozen)

### 3.1 Primary path — stock HF Qwen3-VL-2B-Thinking (recommended)

This is the simplest, leakage-free baseline and is what the existing reference
script (`examples/training/vla_diffusion_redbellpepper_qwen_2b_thinking.sh`)
already uses for the backbone weights. We add `freeze: True`:

```bash
--model.vision_language_backbone.type vlm_backbone
--model.vision_language_backbone.hf_pretrained "Qwen/Qwen3-VL-2B-Thinking"
--model.vision_language_backbone.freeze True
--data.processor "Qwen/Qwen3-VL-2B-Thinking"
```

`load_state_dict` is strict in this repo (`vla_foundry/file_utils.py:495`), so
we never call top-level `--model.resume_from_checkpoint` for this study — the
shape mismatch on the resized policy would crash it.

### 3.2 What the HF collection actually ships (verified live)

The `TRI-ML/vla-foundry` HF collection contains both **VLMs (backbone-only)** and
**VLAs (full diffusion_policy)**. Verified by fetching each repo's `config_model.yaml`:

| Repo | `config_model.yaml` `type:` | Has `vlm_config_model.yaml`? | Loadable as backbone via `vlm_foundry_backbone`? |
|---|---|---|---|
| `TRI-ML/Foundry-VLM-1.3B-200M` | `vlm` (or `vlm_hf`) | n/a (it *is* the VLM) | ✅ yes, direct |
| `TRI-ML/Foundry-VLM-1.3B-165M` | `vlm` (or `vlm_hf`) | n/a | ✅ yes, direct |
| `TRI-ML/Foundry-VLA-1.7B-full` | `diffusion_policy` (backbone = `vlm_foundry_backbone` ≈ SmolVLM 1.5B) | **✅ yes** — bundled sibling config | ✅ yes (see §3.2.A) |
| `TRI-ML/Foundry-VLA-1.7B-sim` | `diffusion_policy` (same shape, sim-only training) | likely yes | ✅ yes |
| `TRI-ML/Foundry-VLA-1.7B-real` | `diffusion_policy` (same shape, real-only training) | likely yes | ✅ yes |
| `TRI-ML/Foundry-Qwen3VLA-2.1B` | `diffusion_policy` (backbone = `vlm_backbone` = `Qwen/Qwen3-VL-2B-Thinking`, transformer 24L/1024d) | **❌ no** | ❌ no (use §3.2.B extraction) |

The key plumbing: when the backbone is `vlm_foundry_backbone`,
`create_vlm_foundry_backbone` (`vla_foundry/models/diffusion_policy/__init__.py:46-58`)
falls back to the config-origin's sibling `vlm_config_model.yaml` to build the
VLM architecture, then loads the VLM weights from the bundled checkpoint. That
sibling file is **only** present in the SmolVLM-based 1.7B releases — the
Qwen3VLA-2.1B release was trained end-to-end from stock HF Qwen3-VL-2B-Thinking
and doesn't ship a separable backbone config.

#### 3.2.A SmolVLM-1.5B-backbone path (clean, no script needed)

If you're OK using the SmolVLM 1.5B backbone (instead of Qwen3-VL-2B), you get
the *Foundry-tuned* backbone for free with no extraction:

```bash
--model.vision_language_backbone.type vlm_foundry_backbone
--model.vision_language_backbone.resume_from_checkpoint \
    "hf://TRI-ML/Foundry-VLA-1.7B-full/checkpoints/checkpoint_4.pt"
--model.vision_language_backbone.freeze True
--data.processor "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"
```

This is the cleanest "tuned backbone + fresh resized policy" setup in the repo
— it touches only the backbone submodule's loader, so the strict policy
`load_state_dict` in `file_utils.py:495` is never triggered. **If the brief
will accept SmolVLM 1.5B in place of Qwen3-VL-2B, prefer this over §3.2.B.**
The trade-off is that the SmolVLM backbone is ~1.5B params (vs. ~2B for Qwen)
and was trained on a different VLM mixture; closed-loop performance on LBM is
similar but not identical to the Qwen3VLA-2.1B baseline.

#### 3.2.B Qwen3-VL-2B-backbone path (requires extraction)

To use the Foundry-tuned **Qwen3-VL-2B** backbone, you have to extract it from
the full `Foundry-Qwen3VLA-2.1B` diffusion-policy checkpoint, because the
release doesn't ship `vlm_config_model.yaml`. Save this as
`scripts/extract_backbone.py`:

```python
import sys, torch
from vla_foundry.hf_hub import resolve_hf_path
from vla_foundry.file_utils import pt_load, unwrap_state_dict

src = resolve_hf_path("hf://TRI-ML/Foundry-Qwen3VLA-2.1B/checkpoints/checkpoint_4.pt")
dst = sys.argv[1]  # e.g. $EXP_ROOT/backbone_only.pt

sd = unwrap_state_dict(pt_load(src, map_location="cpu")["state_dict"])
backbone = {k[len("vision_language_backbone."):]: v
            for k, v in sd.items() if k.startswith("vision_language_backbone.")}
torch.save({"state_dict": backbone, "checkpoint_num": 0, "global_step": 0}, dst)
print(f"wrote {len(backbone)} backbone tensors to {dst}")
```

Then in training, instead of relying on `hf_pretrained`, point the **backbone's**
own `resume_from_checkpoint` at the extracted file:

```bash
--model.vision_language_backbone.resume_from_checkpoint $EXP_ROOT/backbone_only.pt
--model.vision_language_backbone.freeze True
```

Only do this if (a) you specifically want the Qwen3-VL-2B backbone and (b) you
want it Foundry-tuned. For a clean ablation on the **policy axis** the §3.1
stock backbone is preferable — you remove one confound (backbone provenance)
at the cost of a small absolute success-rate offset.

#### 3.2.C Decision flow

```
Want Foundry-tuned backbone?
├── No  → §3.1 (stock Qwen/Qwen3-VL-2B-Thinking, frozen)        ← simplest
└── Yes → Need specifically the Qwen3-VL-2B architecture?
          ├── No  → §3.2.A (SmolVLM 1.5B via Foundry-VLA-1.7B-full) ← cleanest
          └── Yes → §3.2.B (extract from Foundry-Qwen3VLA-2.1B)     ← scripted
```

> **Correction to an earlier note**: I previously stated "no published
> backbone-only Qwen3VLA checkpoint." That phrasing was correct for the
> *Qwen3-VL-2B* lineage specifically (Qwen3VLA-2.1B doesn't ship
> `vlm_config_model.yaml`), but the broader collection **does** publish
> separable VLM backbones — both as standalone `Foundry-VLM-1.3B-*` repos and
> bundled inside `Foundry-VLA-1.7B-*` via the `vlm_foundry_backbone` mechanism.

---

## 4. Training command (single GPU, one rung)

This is the canonical command. Copy-paste, set `SIZE` and `BUDGET`, run. It uses
the bellpepper preset as the chassis and overrides only the size knob, the
backbone, and the data paths.

```bash
export SIZE=100m                # 11m | 100m | 410m
export BUDGET=200000            # total_train_samples
export RUN=$EXP_ROOT/runs/${TASK}_size-${SIZE}_budget-${BUDGET}
mkdir -p $RUN

uv run python vla_foundry/main.py \
  --config_path vla_foundry/config_presets/training_jobs/vla_diffusion_bellpepper.yaml \
  --logs $RUN \
  --name policy_size_sweep \
  --wandb False \
  --distributed.fsdp False \
  --hparams.torchcompile False \
  --hparams.per_gpu_batch_size 16 \
  --hparams.global_batch_size 128 \
  --hparams.lr 5e-5 \
  --hparams.warmup "1000" \
  --hparams.optimizer adamw \
  --hparams.lr_scheduler cosine \
  --total_train_samples $BUDGET \
  --num_checkpoints 5 \
  --data.dataset_manifest "[$EXP_ROOT/data/$TASK/manifest.jsonl]" \
  --data.dataset_statistics "[$EXP_ROOT/data/$TASK/stats.json]" \
  --data.dataset_modality "[robotics]" \
  --data.dataset_weighting "[1.0]" \
  --data.processor "Qwen/Qwen3-VL-2B-Thinking" \
  --model.vision_language_backbone.type vlm_backbone \
  --model.vision_language_backbone.hf_pretrained "Qwen/Qwen3-VL-2B-Thinking" \
  --model.vision_language_backbone.freeze True \
  --model.transformer "include vla_foundry/config_presets/models/transformer_${SIZE}.yaml" \
  --model.transformer.is_causal False
```

Notes:
- **No `torchrun`, no FSDP** (`--distributed.fsdp False`). Frozen 2B backbone +
  small policy ⇒ single-GPU is fine. Per-GPU batch may need to drop on smaller
  cards; **keep `--hparams.global_batch_size 128` constant across all 15 runs**
  (the repo computes accumulation automatically — see README §5.1).
- LR `5e-5`, warmup 1000, cosine, AdamW — the single-task recipe the brief
  specifies; matches the existing single-task scripts.
- `--num_checkpoints 5` saves 5 evenly-spaced checkpoints so you can eval the
  best one (smaller policies overfit earlier on this tiny dataset).
- A dry-run with `--resolve_configs True --resolve_configs_path $RUN/` writes
  `$RUN/resolved_config.yaml`. Diff this between rungs to confirm only the
  transformer block differs.
- Replace `--wandb False` with `--wandb True` if you want logging; the project
  defaults wandb on.

### Sanity check the first run

```bash
# from a separate shell, after a few hundred steps:
tail -f $RUN/policy_size_sweep/*/out.log
```

Make sure: (a) param count of the trainable subset matches the rung
(~11M/100M/410M, NOT ~2.3B — that means the backbone freeze didn't take); (b)
loss decreases monotonically; (c) no `RuntimeError: size mismatch` from the
state-dict loader (means you accidentally passed `--model.resume_from_checkpoint`
top-level).

---

## 5. Sweep matrix

3 sizes × 5 budgets = 15 runs. Pick budgets that bracket "way too little" up to
"single-task plateau" — adjust after the first 5-rung pass if the largest budget
is still under-trained.

```bash
for SIZE in 11m 100m 410m; do
  for BUDGET in 20000 60000 200000 600000 1800000; do
    RUN=$EXP_ROOT/runs/${TASK}_size-${SIZE}_budget-${BUDGET}
    mkdir -p $RUN
    # …same command as §4, with the two env vars substituted…
  done
done
```

A simple driver script that wraps §4's command and reads `$SIZE` / `$BUDGET`
from env is the cleanest way; keep one log dir per (size, budget) pair so
checkpoints don't collide.

Guidance:
- Hold `global_batch_size = 128` and `lr = 5e-5` fixed for every run.
- Hold `seed`, augmentation, camera set, processor, action/proprio fields fixed.
- The only knobs that vary across runs are `--model.transformer "include ..."`
  and `--total_train_samples`.

---

## 6. Closed-loop evaluation on `lbm-eval-oss`

The repo ships a Docker-based sim and an automated driver
(`vla_foundry/eval/run_evaluation.py`); flow is documented in
`tutorials/sim_evaluation_tutorial.ipynb` and `tutorials/STATISTICAL_COMPARISON.md`.

```bash
docker pull toyotaresearch/lbm-eval-oss:vla-foundry
```

For each trained checkpoint dir (e.g. `$EXP_ROOT/runs/.../<exp>/checkpoints/`):

```bash
SIZE=100m; BUDGET=200000
CKPT_DIR=$EXP_ROOT/runs/${TASK}_size-${SIZE}_budget-${BUDGET}/policy_size_sweep/<resolved-exp-name>
MODEL_TAG=size-${SIZE}_budget-${BUDGET}

uv run python vla_foundry/eval/run_evaluation.py $CKPT_DIR \
  --model_name $MODEL_TAG \
  --tasks $TASK \
  --num_episodes 0:200 \
  --max_sample_size 200 \
  --num_gpus 1 \
  --output_dir $EXP_ROOT/rollouts
```

Critical rules from `STATISTICAL_COMPARISON.md`:
- **`--max_sample_size 200` is locked in before any episodes run** and **must be
  identical** across every model you intend to compare (15 rungs + the baseline).
  Changing it after the fact invalidates STEP's stopping boundary.
- Use the same `--output_dir` for all 16 models — the dashboard auto-combines
  matching `(model, task, episode)` triplets and runs Bonferroni-corrected
  pairwise STEP tests with Compact-Letter-Display.
- `--num_episodes 0:200` matches the paper. Start with `0:50` to smoke-test the
  pipeline, then extend to `:200` (the dashboard combines non-overlapping
  ranges automatically — same `--max_sample_size`, no duplicate indices).

### Inspect results

```bash
uv run --group dashboard python vla_foundry/eval/results_explorer.py \
  $EXP_ROOT/rollouts/
# open http://localhost:8505 — Model Comparison tab has the violin + CLD plots
```

The success-rate-vs-size, success-rate-vs-budget, and (size × budget) heatmap
plots come from the `results.json` files under
`$EXP_ROOT/rollouts/<model>/$TASK/rollouts/<timestamp>/`. The schema is
documented at the bottom of `STATISTICAL_COMPARISON.md` (`success_rate`,
`num_success`, `num_evaluated`, per-episode `is_success`). A short pandas
script can produce all three plots from those JSONs.

---

## 7. In-house baseline (do this run too)

Per the brief, the only valid baseline is the full-size policy trained in the
**same frozen-backbone regime** with the **same recipe**. That is exactly the
`size=410m`, `budget=1800000` cell of the sweep matrix above — no extra config
needed. Label it `baseline_410m_full_budget` in eval so it appears as a
distinct entry in the CLD table.

Do **not** add `TRI-ML/Foundry-Qwen3VLA-2B` as a comparison row in the same
dashboard — that model was trained end-to-end with the backbone **unfrozen** on
the full LBM mixture, so co-plotting it would suggest like-for-like and it
isn't. If you want it as labeled context, render a separate side-by-side figure
with a clear "different regime" caption.

---

## 8. Things flagged from the repo that bear on the brief

1. **No public backbone-only Qwen3VLA checkpoint.** The brief calls the
   backbone "well-tuned"; the only published artifact is the full diffusion-
   policy checkpoint. Practical reading: either use stock HF Qwen3-VL-2B-Thinking
   (§3.1, recommended) or extract the backbone tensors yourself (§3.2). The
   `vlm_foundry_backbone` model type is for resuming from a pre-trained
   *VLM-stage* run, not a VLA-stage release.
2. **Top-level `resume_from_checkpoint` is strict.** `load_model_checkpoint` in
   `vla_foundry/file_utils.py:495` calls `load_state_dict(sd)` without
   `strict=False`. Any path that pulls a 410M-policy checkpoint into a 11M-policy
   model crashes. The §3.2 script targets the backbone *submodule* checkpoint
   path, which is loaded by a different code path (`vlm_foundry_backbone` /
   submodule `resume_from_checkpoint`) and only sees backbone tensors.
3. **YAML `<<: !include` cannot partial-override.** Documented in README §1.6
   and warned about in `diffusion_policy_bellpepper.yaml` itself. We side-step
   it by using CLI `--model.transformer "include …"` overrides instead of
   writing per-rung YAML files.
4. **The `vocab_size: 50432` field is dead weight** in policy mode. Don't
   "fix" it — the transformer presets are shared with LLM/VLM code paths and
   editing them ripples elsewhere.
5. **Single-task overfit risk** (brief's own caveat). A single LBM task is on
   the order of a few hundred demos × 14-step action chunks. At `budget = 1.8M`
   samples the 410M policy will see each demo hundreds of times. Plan to eval
   every checkpoint (`--num_checkpoints 5` writes 5) and pick the best by
   closed-loop success rate, not by terminal training loss.
6. **`action_dim`/`proprioception_dim` are shared params.** They're populated
   from the data config in `DiffusionPolicyParams.init_shared_attributes`. Do
   not set them on the CLI or in a YAML — you'll get a stale mismatch the
   moment the data fields change.
