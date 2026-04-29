# Experiments

Last curated update: 2026-04-27. Log status for active runs was inspected through about 15:42 UTC on 2026-04-27.

This file is the living experiment notebook for the OceanTACO / Global_DL_SSH work. Each launched experiment should keep its immutable launcher metadata in `experiments/<id>/`, while this file records the scientific intent, differences from previous runs, current status, and interpretation of results.

## Current Questions

The experiments are converging on three related scientific questions:

1. Can a SimVP-style model reconstruct useful sea-surface-height fields in the Gulf Stream from sparse or irregular observations?
2. Does adding an auxiliary geophysical variable such as L4 SST improve SSH mapping, especially in a dynamically sharp western-boundary-current region?
3. How do target choices change the meaning of the result: predicting SWOT-like L3 SSH tests high-resolution along-track / swath reconstruction, while predicting L4 SSH / DUACS-like fields tests recovery of a smoother gridded analysis product?

Important caveat: experiments `a001` and `a002` used `l4_sst` before commit `30269af`, which repaired a double-converted SST case where physically impossible SST values below -100 Celsius were shifted back by 273.15. Treat those SST-input results as useful pipeline evidence, but not as final evidence for the value of SST.

## Experiment Index

| ID | Status | Recorded commit | Base config | Inputs | Target | Main purpose |
| --- | --- | --- | --- | --- | --- | --- |
| `a000` | finished training, no prediction artifacts found | `d8a9d3f` | `configs/oceantaco_gulf_stream.yaml` | `l3_ssh`, `l4_sst` | `l3_swot` | Minimal Gulf Stream / SST smoke experiment. |
| `a001` | finished training and prediction | `a774b6c` | `configs/oceantaco_gulf_stream.yaml` | `l3_ssh`, `l4_sst` | `l3_swot` | First meaningful reserved-satellite Gulf Stream run with shorter sequences. |
| `a002` | finished training and prediction | `8ddbade` | `configs/oceantaco_gulf_stream.yaml` | `l3_ssh`, `l4_sst` | `l3_swot` | Expanded Gulf Stream region and training set. |
| `a007` | running / incomplete when inspected | `30269af` | `experiments/a002/a002_base_config.yaml` | `l3_ssh`, `l4_sst` | `l3_swot` | Re-run of `a002` after the SST unit repair. |
| `a008` | downloading / not yet training when inspected | `78e49f7` | `configs/config_gulf_stream_duacs_no_sst.yaml` | `l3_ssh` | `l4_ssh` | DUACS/L4 target baseline without SST. |
| `a009` | downloading / not yet training when inspected | `78e49f7` | `configs/config_gulf_stream_duacs_target.yaml` | `l3_ssh`, `l4_sst` | `l4_ssh` | DUACS/L4 target with SST; direct comparison to `a008`. |
| `a010` | downloading / not yet training when inspected | `78e49f7` | `configs/swot_as_input.yaml` | `l3_ssh`, `l3_swot` | `l4_ssh` | Observation-rich DUACS/L4 target run using SWOT as an input. |

No experiment directories for `a003` through `a006` were present in the workspace at this update.

## Git Context

| Commit | Message | Used by | Experiment relevance |
| --- | --- | --- | --- |
| `d8a9d3f` | `minimal test for SST` | `a000` | Early SST-enabled Gulf Stream prototype. |
| `a774b6c` | `diminish sequence lenght` | `a001` | Reduced temporal sequence length from 5 to 3 frames, making runs cheaper and centered on a 3-frame context. |
| `8ddbade` | `extend region` | `a002` | Expanded the Gulf Stream box from `[-65, -60, 30, 35]` to `[-70, -60, 30, 40]`. |
| `30269af` | `fix unit conversion in SST` | `a007` | Repairs double-converted `l4_sst` values, making later SST-input experiments scientifically cleaner. |
| `78e49f7` | `different experiment configs` | `a008`, `a009`, `a010` | Adds the three DUACS/L4 target comparison configs. |

## Cross-Experiment Design Notes

All runs use the same basic model family unless noted: `no_skip_configurable` SimVP with `gsta` blocks, `96x96` grids, 3-degree patches, batch size 2, learning rate `0.001`, 15 epochs, validation every 3 epochs, and seed 42.

The target changed across phases. `a000` to `a007` predict `l3_swot`, so they ask whether the model can produce a SWOT-like SSH field from other observations. `a008` to `a010` predict `l4_ssh`, so they ask whether the model can reproduce a gridded DUACS/L4-style analysis product.

The reserved satellite rule is active from `a001` onward: `Cryosat-2 New Orbit` is removed from `l3_ssh` inputs for train/validation/test and used as an independent reference on the test split. This is scientifically valuable because it tests whether the network can reconstruct information from an observing platform it did not see as an input source.

The date split semantics need attention. `a001`, `a002`, and `a007` train on 2024-09-01 through 2025-09-01 and test on 2024-04-01 through 2024-04-15, so the evaluation is temporally disjoint and earlier than the training window. `a008` to `a010` train from 2023-03-29 through 2025-01-01 and test on 2024-04-01 through 2024-04-15, so the test dates fall inside the training date range. If the goal is strict temporal generalization, those DUACS-target configs should be revisited.

## Results Summary

| ID | Train queries | Validation queries | Test queries | Loss status | Prediction status | Reserved-reference metrics |
| --- | ---: | ---: | ---: | --- | --- | --- |
| `a000` | 1 | 12 | not generated | final train `0.7385`, final val `0.9937`, best val `0.9937` at epoch 14 | none found | none |
| `a001` | 128 | 4 | 52 | final train `0.3308`, val reported `0.0000` | 32 saved prediction files, 20 skipped empty-target batches | RMSE `0.0870` m, MAE `0.0636` m, bias `-0.0048` m, corr `0.7130`, valid pixels `4644` |
| `a002` | 256 | 16 | 208 | final train `1.1095`, final val `0.3127`, best val `0.2851` at epoch 8 | 103 saved prediction files, 105 skipped empty-target batches | RMSE `0.1286` m, MAE `0.0943` m, bias `-0.0289` m, corr `0.8117`, valid pixels `12710` |
| `a007` | 256 | 16 | 208 | incomplete: epoch 2 train `1.3376`, val `0.4277` | none yet | pending |
| `a008` | 512 | 1856 | 112 | pending | download in progress, about `625/4710` missing files by last inspected log | pending |
| `a009` | 512 | 1856 | 112 | pending | download in progress, about `625/5005` missing files by last inspected log | pending |
| `a010` | 512 | 1856 | 112 | pending | download in progress, about `625/5003` missing files by last inspected log | pending |

## Per-Experiment Notes

### `a000`

This was a minimal Gulf Stream run with `l3_ssh` and `l4_sst` as inputs and `l3_swot` as target. It used only 1 training query in a small box `[-65, -60, 30, 35]`, 5-frame sequences, and validation in March 2025.

Result: the training loss decreased from about `2.01` to `0.74`, and validation loss decreased from `1.14` to `0.99`. That is enough to show the pipeline and optimizer were functioning, but with one training sample it is not a scientifically meaningful skill estimate. The model can memorize / adapt to the sample; it does not demonstrate generalization.

Scientific meaning: treat `a000` as a plumbing and feasibility run. It shows the data path, checkpointing, and validation loop were working before the later reserved-satellite experiments.

### `a001`

This run moved to a more realistic 128 training queries, 3-frame sequences, and the reserved `Cryosat-2 New Orbit` evaluation. The region remained the smaller Gulf Stream box `[-65, -60, 30, 35]`. Inputs were still `l3_ssh` and `l4_sst`; target was `l3_swot`.

Result: training loss fell to `0.3308`. Validation loss was reported as `0.0000` at every validation checkpoint, which is suspicious rather than excellent. The validation set had only 4 queries, and the prediction log later skipped 20 of 52 test batches because the target was empty. The reserved-reference comparison gave RMSE `0.0870` m, MAE `0.0636` m, near-zero bias, and correlation `0.7130` over 4644 valid pixels.

Scientific meaning: the reserved-reference metrics suggest the model captured some SSH structure relative to held-out Cryosat-2 observations, but the validation signal is not trustworthy. The small validation set and empty-target behavior make this run better evidence for workflow viability than for final model skill. Because it also predates the SST unit repair, it should not be used to claim that SST helps.

### `a002`

This run expanded the Gulf Stream region to `[-70, -60, 30, 40]` and doubled the training queries to 256. It kept the same `l3_ssh` + `l4_sst` to `l3_swot` task and the same reserved Cryosat-2 rule.

Result: the broader run has a meaningful validation curve: best validation loss `0.2851` at epoch 8 and final validation loss `0.3127`. Training loss did not monotonically improve and ended at `1.1095`, so the model was not simply overfitting down to zero. Prediction produced 103 files and skipped 105 empty-target batches. Reserved-reference metrics were RMSE `0.1286` m, MAE `0.0943` m, bias `-0.0289` m, and correlation `0.8117` over 12710 valid pixels.

Scientific meaning: compared with `a001`, the larger region is harder in absolute error but more informative: many more valid pixels and a higher correlation with the reserved satellite. The negative bias means the model is low relative to Cryosat-2 by about 2.9 cm on average in the valid comparison pixels. This could indicate amplitude damping, target/input normalization mismatch, missing mesoscale energy, or simply the harder spatial domain. Because this run predates the SST repair, rerun `a007` is the cleaner version to trust.

### `a007`

This is an `a002` rerun at commit `30269af`, after repairing double-converted `l4_sst`. It uses the same base config snapshot as `a002`: 256 train queries, expanded Gulf Stream region, `l3_ssh` + `l4_sst` inputs, `l3_swot` target, and reserved Cryosat-2.

Result so far: the loss CSV had epochs 0 through 2 when inspected. Training loss moved from `2.0561` to `1.4768` to `1.3376`, and the first validation point was `0.4277` at epoch 2. The training log showed later training still in progress.

Scientific meaning: this is the first clean test of the SSH+SST-to-SWOT hypothesis after the SST repair. Its comparison to `a002` is especially important: if `a007` improves, the old SST values likely degraded learning; if it does not, SST may be weakly useful for this target or the architecture may not be exploiting thermal-front information yet.

### `a008`

This is the first DUACS/L4 target baseline. It uses only `l3_ssh` as input and predicts `l4_ssh`, with 512 train queries, the expanded Gulf Stream region, validation from 2025-01-12 through 2025-09-01, and test from 2024-04-01 through 2024-04-15.

Result so far: download/query setup was still running when inspected. No checkpoints, losses, or predictions were present.

Scientific meaning: this is the altimetry-only baseline for the DUACS target. It asks how much of the gridded analysis can be reproduced from sparse L3 SSH alone. This baseline is necessary before interpreting any SST or SWOT-as-input improvement.

### `a009`

This is the paired DUACS/L4 target experiment with `l3_ssh` and repaired `l4_sst` as inputs, predicting `l4_ssh`. It is otherwise aligned with `a008`.

Result so far: download/query setup was still running when inspected. No checkpoints, losses, or predictions were present.

Scientific meaning: compare directly with `a008` to isolate the added value of SST. If `a009` improves over `a008`, the model is likely using thermal-front structure as a proxy for SSH gradients and mesoscale dynamics. If it does not, SST may be redundant with L3 SSH for this target, misaligned with SSH at the chosen temporal scale, or not well exploited by the current architecture.

### `a010`

This DUACS/L4 target experiment uses `l3_ssh` plus `l3_swot` as inputs and predicts `l4_ssh`. It is an observation-rich setup rather than a pure operational baseline.

Result so far: download/query setup was still running when inspected. No checkpoints, losses, or predictions were present.

Scientific meaning: this can act as an upper-bound or diagnostic run: adding SWOT-like input should provide high-resolution SSH information that helps recover the smoother L4 target. Interpret with care, because if the L4 target already assimilates related altimetry, this is not an independent physical validation. It is still valuable for testing how much wide-swath SSH information changes reconstruction relative to `a008`.

## Interpretation Priorities

The immediate scientific priority is to finish `a007`, then compare it with `a002`. That comparison controls for region, target, architecture, and split definitions while changing the SST data repair. It is the cleanest way to decide how much the earlier SST bug affected the SWOT-target results.

The next priority is the `a008` versus `a009` pair. These runs isolate the effect of SST on a DUACS/L4 target. The important comparison is not just final loss; also compare reserved Cryosat metrics, spatial error maps near the Gulf Stream front, and whether SST reduces bias or merely smooths fields.

Finally, compare `a010` against `a008` and `a009` as an information-content experiment. If SWOT-as-input strongly improves L4 reconstruction, it supports the idea that high-resolution SSH observations contain structure that the current L3-only setup misses. If gains are small, the L4 target may already be too smooth or too close to the L3 information content for the model to show a large difference.

## Maintenance Rules

For every new experiment, add or complete an entry in this file with:

- experiment id, creation time, git commit, branch, and base config
- input variables, target variables, geographic region, date splits, query counts, and model changes
- status: downloading, training, predicting, finished, failed, or superseded
- final train/validation loss, best validation epoch, skipped batches, prediction counts, and reserved-reference metrics when available
- scientific interpretation: what this run tests, what it changed, what the result suggests, and which caveats apply

`run_experiment.sh` now appends a minimal entry under the generated ledger below when a new run starts and adds a completion line when the launcher finishes. After the run, promote the generated note into a curated per-experiment section like the ones above.

## Generated Run Ledger

Future launcher-created experiments will append short records here.

### a008

- Created: 2026-04-28T14:12:46Z
- Git: `329919d98a19c77d24113c15bdda6bb64f2d65dc` on branch `ref-new`
- Base config: ``
- Experiment dir: `experiments/a008`
- Status: started by `run_experiment.sh`; results pending.
- Scientific note: fill in after training and prediction complete.
