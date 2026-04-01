# Global_DL_SSH
Neural network method for mapping sea surface height (SSH) from ocean observations.

This repo uses a config-driven OceanTACO workflow for training and inference with configurable regional or global subsets.

The new default path uses the installable `ocean-taco` package and lets you choose:

- which OceanTACO variables are used as model inputs and targets
- which dates belong to train, validation, and test splits
- which region bbox presets to use for each split, or `global`
- the query patch size, time window, output grid, normalization, model hyperparameters, and output paths

The main configuration lives in [`configs/oceantaco.yaml`](configs/oceantaco.yaml).

For a quick pipeline sanity check, use [`configs/oceantaco_smoke_test.yaml`](configs/oceantaco_smoke_test.yaml). It uses a tiny bbox, short date range, small grid, shallow model, batch size `1`, and `1` epoch so training and prediction finish much faster.

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

Notes for the OceanTACO workflow:

- `splits.<name>.regions: global` uses a global query bbox.
- Region selection is now query-based through bbox presets defined in the YAML `regions` section.
- Training splits use `QueryGenerator.generate_training_queries(...)`.
- Validation and test splits use `QueryGenerator.generate_eval_queries(...)`.
- Training and prediction require `ocean-taco`, `torch`, `xarray`, `numpy`, `pandas`, and `PyYAML`.
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
