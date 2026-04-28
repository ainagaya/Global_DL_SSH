# Global_DL_SSH
Neural network method for mapping sea surface height (SSH) from ocean observations.

This repo uses a config-driven OceanTACO workflow for training and inference with configurable regional or global subsets.

The new default path uses the installable `ocean-taco` package and lets you choose:

- which OceanTACO variables are used as model inputs and targets
- which dates belong to train, validation, and test splits
- which region bbox presets to use for each split, or `global`
- the query patch size, time window, output grid, normalization, model hyperparameters, and output paths
- the SimVP model variant (`no_skip_configurable`, `no_skip`, or `no_skip_sst`) and hidden block type (`gsta`, `convnext`, `swin`, `tau`, etc.)

The main configuration lives in [`configs/oceantaco.yaml`](configs/oceantaco.yaml).

For a quick pipeline sanity check, use [`configs/oceantaco_smoke_test.yaml`](configs/oceantaco_smoke_test.yaml). It is still lightweight, but it now uses the same broad North Pacific region as the main config, a slightly longer date window, a few more queries, and a `64x64` grid so you are more likely to get preliminary but meaningful outputs while keeping runtime short.

For a more meaningful regional experiment in 2025, use [`configs/oceantaco_gulf_stream.yaml`](configs/oceantaco_gulf_stream.yaml). It targets the Gulf Stream region in the western North Atlantic with a larger grid, `3`-frame sequences, more training queries, and `15` epochs.

Additional regional variants following the same template are available for the Kuroshio Current, Brazil Current, Agulhas Current, and East Australia Current:
[`configs/oceantaco_kuroshio_current.yaml`](configs/oceantaco_kuroshio_current.yaml),
[`configs/oceantaco_brazil_current.yaml`](configs/oceantaco_brazil_current.yaml),
[`configs/oceantaco_agulhas.yaml`](configs/oceantaco_agulhas.yaml), and
[`configs/oceantaco_east_australia.yaml`](configs/oceantaco_east_australia.yaml).

MLflow tracking is configured in the same YAML under `tracking.mlflow`. By default it is disabled. To enable local tracking, set:

```yaml
tracking:
  mlflow:
    enabled: true
    tracking_uri: ./mlruns
    experiment_name: Global_DL_SSH
```

When enabled, training logs the flattened config, dataset sizes, epoch losses, skipped-batch counts, the CSV loss log, and checkpoints. Prediction logs the config, checkpoint artifact, split metadata, and saved prediction counts. Prediction `.npz` artifacts are optional and controlled by `tracking.mlflow.log_prediction_artifacts`.

Runtime console logging is configured under `logging.level` in the YAML. The default is `INFO`. Set it to `DEBUG` for more verbose progress output or `WARNING` to reduce noise.

Run training with:

```bash
python3 simvp_ddp_training.py --config configs/oceantaco.yaml
```

To download a local OceanTACO mirror for the files required by your configured
splits, run:

```bash
python3 download_oceantaco_local.py --config configs/oceantaco.yaml
```

The downloader reads the configured splits, variables, and query settings,
downloads the matching OceanTACO files plus dataset metadata into
`oceantaco.download_path`, and the training / inference / query-analysis
pipeline will automatically use that local mirror when it exists.

Smoke-test training:

```bash
python3 simvp_ddp_training.py --config configs/oceantaco_smoke_test.yaml
```

To launch a fully tracked experiment run under `experiments/a000`, `a001`, ...
use:

```bash
./run_experiment.sh configs/oceantaco_smoke_test.yaml
```

To resume an existing experiment in place from its latest checkpoint, use:

```bash
./run_experiment.sh --resume=a000
```

The launcher will:

- refuse to run if the git worktree is dirty
- assign the next experiment id
- record the current git commit
- copy the base config into the experiment folder
- write a frozen runtime config with experiment-specific output paths
- download the OceanTACO files required by the configured splits
- run training
- run inference with the latest checkpoint from that experiment
- plot the training curves from the loss CSV
- generate per-file, regional, and merged mosaic prediction plots for all available prediction dates
- store weights, logs, predictions, queries, MLflow artifacts, and analysis plots inside that experiment directory

In resume mode, the launcher reuses that experiment's frozen runtime config and
latest checkpoint, appends to the existing launcher logs, continues training,
then reruns inference and analysis with the newest checkpoint.

Run inference with:

```bash
python3 simvp_predict_ssh.py --config configs/oceantaco.yaml --checkpoint model_weights/simvp_oceantaco_epoch0.pt
```

Smoke-test inference:

```bash
python3 simvp_predict_ssh.py --config configs/oceantaco_smoke_test.yaml --checkpoint model_weights/simvp_oceantaco_smoke_epoch0.pt
```

To visualize prediction outputs saved as `.npz`, use:

```bash
python3 plot_predictions.py predictions
python3 plot_predictions.py predictions/example_file.npz --time-index 0 --channel-index 0
```

To generate one geographic summary figure for a specific date across all saved
prediction regions, use:

```bash
python3 plot_prediction_regions.py predictions --date 2025-03-21
python3 plot_prediction_regions.py predictions --date 2025-03-21 --source-key input_l4_sst
python3 plot_prediction_regions.py predictions --all-dates
```

This script creates one figure with one row per predicted region. Each row
contains a wider context map around the bbox, the selected bbox outline, and
the source, prediction, and target fields for that date. If `cartopy` is
installed, coastlines and land are added automatically.

To merge all prediction regions into one shared lon/lat canvas per date, use:

```bash
python3 plot_prediction_mosaics.py predictions --date 2025-03-21
python3 plot_prediction_mosaics.py predictions --all-dates --alpha 0.5
```

This creates one mosaic figure per target date with source, prediction, target,
and squared-error panels. Overlapping patches are drawn with transparency so
repeated coverage remains visible.

To plot training curves manually, use:

```bash
python3 plot_training_curves.py experiments/a000/logs/a000_losses.csv
```

Notes for the OceanTACO workflow:

- `splits.<name>.regions: global` uses a global query bbox.
- Region selection is now query-based through bbox presets defined in the YAML `regions` section.
- Training splits use `QueryGenerator.generate_training_queries(...)`.
- Validation and test splits use `QueryGenerator.generate_eval_queries(...)`.
- Training and prediction require `ocean-taco`, `torch`, `xarray`, `numpy`, `pandas`, and `PyYAML`.
- Local OceanTACO mirroring uses `oceantaco.download_path` and may use `huggingface_hub` when available for efficient dataset downloads.
- MLflow is optional, but the provided Dockerfile installs it.

Model selection notes:

- `model.variant: no_skip_configurable` is the current default and supports different input/output channel counts.
- `model.variant: no_skip` requires the same number of input and target channels.
- `model.variant: no_skip_sst` uses a dedicated two-encoder SSH+SST path and expects exactly two inputs and one target.
- `model.type` selects the hidden translator family inside the SimVP model, such as `gsta`, `convnext`, `swin`, `vit`, `uniformer`, `tau`, or `incepu`.

Checkpoint and prediction provenance:

- Training checkpoints now store `model_metadata`, including the chosen model variant, model type, configured input variables, and whether `l4_sst` was part of the model inputs.
- Prediction `.npz` files now distinguish raw observed inputs from actual model inputs:
  - `input_l4_sst`: raw observed SST from the dataset, or `None` if unavailable
  - `model_input_l4_sst`: the SST field that the model actually received after preprocessing / zero-fill fallback
  - `input_l4_sst_dataset_status`: `observed`, `missing_from_dataset`, or `degenerate`
  - `input_l4_sst_model_status`: `observed`, `zero_filled_missing`, or `zero_filled_degenerate`

Reserved satellite inputs:

- Configs can define `reserved_inputs.<variable>` rules to hold out one satellite/platform from model inputs.
- The Gulf Stream config includes a commented, active example for `reserved_inputs.l3_ssh`.
- The preferred L3 SSH method is `method: xarray_platform`, which opens local `l3_ssh.nc` files with xarray and uses `track_platforms` plus `primary_track` to mask pixels from a selected platform.
- To inspect available platforms before choosing one, run `python3 inspect_l3_ssh_satellites.py oceantaco_data/OceanTACO/DATA --max-files 5`.
- Training / validation / test can exclude the reserved satellite via `exclude_from_splits`.
- Prediction can compare model output against the reserved satellite on selected splits via `metrics_splits`.
- Reserved test metrics are written as `reserved_l3_ssh_metrics.csv` and `reserved_l3_ssh_metrics_summary.json` in the predictions directory.

This repo contains python code for training and inference workflows for SSH mapping from OceanTACO data.

For a more user-friendly implementation designed for production, please see: https://github.com/smartin98/NeurOST.

Current pipeline:

1. Install the runtime dependencies, including `ocean-taco`.
2. Edit [`configs/oceantaco.yaml`](configs/oceantaco.yaml) to choose variables, splits, query settings, and region bbox presets.
3. Run `python3 simvp_ddp_training.py --config configs/oceantaco.yaml`.
4. Run `python3 simvp_predict_ssh.py --config configs/oceantaco.yaml --checkpoint <checkpoint>`.

Minor adaptations to simvp_ddp_training.py would allow any PyTorch model that takes the right input/output dimensions to be used instead.

The SimVP code was only minorly adapted from the original implementation (https://github.com/chengtan9907/OpenSTL) to remove skip connections and allow for the inclusion of SST.
