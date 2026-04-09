# Global_DL_SSH
Neural network method for mapping sea surface height (SSH) from ocean observations.

This repo uses a config-driven OceanTACO workflow for training and inference with configurable regional or global subsets.

The new default path uses the installable `ocean-taco` package and lets you choose:

- which OceanTACO variables are used as model inputs and targets
- which dates belong to train, validation, and test splits
- which region bbox presets to use for each split, or `global`
- the query patch size, time window, output grid, normalization, model hyperparameters, and output paths

The main configuration lives in [`configs/oceantaco.yaml`](configs/oceantaco.yaml).

For a quick pipeline sanity check, use [`configs/oceantaco_smoke_test.yaml`](configs/oceantaco_smoke_test.yaml). It is still lightweight, but it now uses the same broad North Pacific region as the main config, a slightly longer date window, a few more queries, and a `64x64` grid so you are more likely to get preliminary but meaningful outputs while keeping runtime short.

For a more meaningful regional experiment in 2025, use [`configs/oceantaco_gulf_stream.yaml`](configs/oceantaco_gulf_stream.yaml). It targets the Gulf Stream region in the western North Atlantic with a larger grid, `5`-frame sequences, more training queries, and `15` epochs.

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

Notes for the OceanTACO workflow:

- `splits.<name>.regions: global` uses a global query bbox.
- Region selection is now query-based through bbox presets defined in the YAML `regions` section.
- Training splits use `QueryGenerator.generate_training_queries(...)`.
- Validation and test splits use `QueryGenerator.generate_eval_queries(...)`.
- Training and prediction require `ocean-taco`, `torch`, `xarray`, `numpy`, `pandas`, and `PyYAML`.
- Local OceanTACO mirroring uses `oceantaco.download_path` and may use `huggingface_hub` when available for efficient dataset downloads.
- MLflow is optional, but the provided Dockerfile installs it.

This repo contains python code for training and inference workflows for SSH mapping from OceanTACO data.

For a more user-friendly implementation designed for production, please see: https://github.com/smartin98/NeurOST.

Current pipeline:

1. Install the runtime dependencies, including `ocean-taco`.
2. Edit [`configs/oceantaco.yaml`](configs/oceantaco.yaml) to choose variables, splits, query settings, and region bbox presets.
3. Run `python3 simvp_ddp_training.py --config configs/oceantaco.yaml`.
4. Run `python3 simvp_predict_ssh.py --config configs/oceantaco.yaml --checkpoint <checkpoint>`.

Minor adaptations to simvp_ddp_training.py would allow any PyTorch model that takes the right input/output dimensions to be used instead.

The SimVP code was only minorly adapted from the original implementation (https://github.com/chengtan9907/OpenSTL) to remove skip connections and allow for the inclusion of SST.
