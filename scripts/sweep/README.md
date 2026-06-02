# Policy-Size Sweep: Runbook

Train three sizes of the diffusion policy (full / half / quarter) on
`PickAndPlaceBox` with a frozen SmolVLM-1.5B backbone, then evaluate.

All scripts source `env.sh` for shared paths and hyperparameters — **edit
`env.sh` to change anything global** (data path, budget, batch size,
val-loss toggle, eval settings). Each step script is idempotent and can
be re-run.

## Files

| File | Purpose |
|---|---|
| `env.sh` | Shared env vars. Edit this first if you want to change defaults. |
| `00_check_gpu.sh` | Verifies PyTorch can use the GPU. Fails loudly if not. |
| `01_download_data.sh` | Pulls PickAndPlaceBox shards (~1.2 GB). Idempotent. |
| `02_split_val.sh` | Splits manifest into 90% train / 10% val. Only needed if `WITH_VAL_LOSS=1`. |
| `03_smoke_test.sh` | 256-sample training run on the full rung — flushes integration bugs in ~5 min. |
| `04_train_rungs.sh` | The real sweep. Trains all three rungs (or one if you pass a rung name). |
| `05_val_loss_post_hoc.sh` | Computes held-out val loss on each saved checkpoint via `val_loss_post_hoc.py`. Decoupled from training so train runs at full throughput. No Docker needed. |
| `val_loss_post_hoc.py` | Standalone python: loads one checkpoint + builds val dataloader + reuses `validate_one_checkpoint`. Called by `05_*`. |
| `06_eval_sim.sh` | Closed-loop sim eval. **Requires Docker.** Skip if unavailable. |
| `07_open_dashboard.sh` | Opens the streamlit dashboard at `http://localhost:8505`. |

## Order

Run them in numeric order, stopping at each gate to confirm the output looks right.

```bash
cd /home/jianch2/CS295/vla_foundry

bash scripts/sweep/00_check_gpu.sh                       # gate: GPU works
bash scripts/sweep/01_download_data.sh                   # ~5 min, idempotent
bash scripts/sweep/02_split_val.sh                       # creates 90/10 train/val split
bash scripts/sweep/03_smoke_test.sh                      # ~5 min, gate: pipeline works end-to-end
bash scripts/sweep/04_train_rungs.sh                     # ~1 h × 3 rungs at full throughput

# After training, two independent eval signals (run either or both):
bash scripts/sweep/05_val_loss_post_hoc.sh                    # no Docker — open-loop val MSE per rung; writes $EXP_ROOT/val_losses.csv
bash scripts/sweep/06_eval_sim.sh                             # Docker — closed-loop success rate
bash scripts/sweep/07_open_dashboard.sh                       # after 06
```

## Toggles you'll likely flip

In `env.sh`:

- `WITH_VAL_LOSS=0` (default) — skip inline val during training, run val post-hoc via
  `05_val_loss_post_hoc.sh` instead. Inline val (set to `1`) is ~1.75× slower because
  the val dataloader's worker pool steals throughput from training even when val itself
  only runs once at the end.
- `BUDGET=40000` — total training samples per rung. Halve for faster smoke; double for
  better-converged results.
- `PER_GPU_BATCH_SIZE=16` — drop to 8 or 4 if you OOM. Don't touch `GLOBAL_BATCH_SIZE`
  unless you've thought about it; it's the actual hyperparameter.
- `EVAL_NUM_EPISODES="0:200"` — paper default. Drop to `"0:50"` for a first quick read.
  `EVAL_MAX_SAMPLE_SIZE` must stay at the budget you'll *ever* collect (locks STEP's
  decision boundary — see `tutorials/STATISTICAL_COMPARISON.md`).

## Running a single rung

```bash
bash scripts/sweep/04_train_rungs.sh transformer_205m
```

## Picking a different task

In `env.sh`, change `TASK=PickAndPlaceBox` to any name from
`vla_foundry/data/scripts/download_dataset.py --list`. Then re-run from `01_`.

You probably also want to verify the new task uses the same camera setup
(6 bimanual cameras) and action space (20-dim) as PickAndPlaceBox — if
not, the training-job preset `vla_foundry/config_presets/training_jobs/vla_pickandplace_box.yaml`
needs camera/action overrides.

## What success looks like at each step

| Step | Pass signal | Fail mode |
|---|---|---|
| `00` | `GPU gate PASSED.` printed | sm_120 warning → upgrade PyTorch (see comment in script) |
| `01` | `~52 shards under .../shards/` | network error → re-run, it resumes |
| `02` | `Train: 46 shards / Val: 6 shards` | missing manifest → run `01` first |
| `03` | `Trainable parameters: 325,026,836` and a `checkpoint_0.pt` written | trainable ≈ 1.85B → backbone freeze broke; OOM → drop `PER_GPU_BATCH_SIZE` |
| `04` | 3 run dirs each with `policy_size_<rung>/checkpoints/checkpoint_{0..4}.pt` | OOM → drop batch; loss NaN → drop LR |
| `05` | `$EXP_ROOT/val_losses.csv` with one row per (rung, checkpoint) | empty CSV → no checkpoints under `runs/*/policy_size_*/checkpoints/`; verify `04` finished. `02` must also have produced `manifest_val.jsonl` |
| `06` | `rollouts/<rung>/<task>/rollouts/<ts>/results.json` per rung | needs Docker (see script) |
| `07` | Dashboard at `http://localhost:8505` with violin plot + CLD letters | empty plot → `06` didn't write any results |

## Without Docker — interpreting val loss

Validation loss is open-loop action-prediction MSE on held-out data — *not*
closed-loop success rate. The brief is right that loss can disagree with
success rate, but the ranking signal is useful:

- If `full ≈ half ≈ quarter` on val loss → mild evidence the policy axis isn't
  capacity-bottlenecked. **Pursue closed-loop confirmation** before claiming.
- If `quarter ≫ full` (much worse loss) → strong evidence quarter is undersized.
- If `quarter ≪ full` → likely overfit; check whether the val split is
  representative (small dataset, only 6 shards).

Plot `val_losses.csv` with the rung as a line and `checkpoint` on x. Look for
divergence between rungs around checkpoint 2-3 (when the cosine schedule has
moved through warmup).
