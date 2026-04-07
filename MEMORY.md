# MEMORY

## Project overview

This repository was refactored from a more ad hoc SSH mapping workflow into a config-driven OceanTACO-based training and inference pipeline.

Current main entrypoints:

- `simvp_ddp_training.py`
- `simvp_predict_ssh.py`
- `analyze_queries.py`
- `plot_predictions.py`

Current core support modules:

- `src/oceantaco.py`
- `src/config_utils.py`
- `src/simvp_model.py`
- `src/pytorch_losses.py`
- `src/mlflow_utils.py`
- `src/logging_utils.py`

Debugging utility:

- `analyze_queries.py`
- `plot_predictions.py`

Main configs:

- `configs/oceantaco.yaml`
- `configs/oceantaco_smoke_test.yaml`


## High-level decisions taken

### 1. Move to OceanTACO package API

Decision:

- Remove custom SQL-style / tacoreader / indexing logic.
- Use the installable `ocean-taco` package API directly, guided by the OceanTACO documentation.

Why:

- The user explicitly asked to remove SQL implementation and follow the documented OceanTACO API.
- This makes the pipeline closer to upstream behavior and easier to maintain.

Implementation direction:

- Query generation is now done with OceanTACO `QueryGenerator`.
- Dataset loading is done with `OceanTACODataset`.
- Collation uses `collate_ocean_samples`.
- Region selection is bbox-driven instead of SQL region matching.


### 2. Move hard-coded parameters into YAML

Decision:

- Consolidate configurable values into YAML.

Why:

- The user asked to move configuration out of code.
- This makes experiments reproducible and easier to vary.

Configuration now includes:

- paths
- OceanTACO source path
- regions and bbox presets
- query parameters
- data variables, normalization, sequence length, target index, grid size
- split definitions
- model hyperparameters
- training parameters
- prediction parameters
- MLflow settings
- logging verbosity


### 3. Keep support for regional and global runs

Decision:

- Support both named regions and `global`.

Why:

- The user explicitly requested training and testing on specific regions and also globally.

Implementation:

- `regions` section defines bbox presets.
- Split configs can reference one region, many regions, or `global`.


### 4. Remove old unused scripts from the main pipeline

Decision:

- Delete scripts that no longer belong to the maintained OceanTACO pipeline.

Deleted:

- `create_coord_grids.py`
- `generate_global_data.py`
- `pre_process_training.py`
- `pre_process_testing.py`
- `merge_maps.py`
- `calculate_currents.py`
- `subset_for_flux.py`
- `plot_ics.py`
- `src/global_data_utils.py`
- `src/merging.py`

Why:

- The user requested deleting scripts not used in the main pipeline.


### 5. Add Docker support

Decision:

- Add a `Dockerfile` based on:
  `pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime`

Why:

- The user requested a Dockerfile including project dependencies and `vim`.

Added to Docker image:

- requested utilities: `tmux`, `htop`, `nano`, `vim`, `zip`, `git`
- geo/system libs: `libgeos-dev`, `libproj-dev`, `proj-bin`, `proj-data`
- Python dependencies for current workflow
- `pytest`, `pytest-cov`
- `mlflow`
- OceanTACO package install


### 6. Add tests and coverage

Decision:

- Create a pytest suite covering key config, adapter, and training behaviors.

Why:

- The user requested pytest coverage.

Files added:

- `pytest.ini`
- `tests/conftest.py`
- `tests/test_config_utils.py`
- `tests/test_oceantaco_adapter.py`
- `tests/test_training_script.py`


### 7. Add MLflow instrumentation

Decision:

- Add optional MLflow tracking controlled through YAML.

Why:

- The user explicitly requested MLflow instrumentation.

Implementation:

- New helper: `src/mlflow_utils.py`
- Training logs config, dataset sizes, metrics, checkpoints, loss CSV
- Prediction logs config, checkpoint artifact, prediction counts, skipped counts
- Config controls whether checkpoints / config / prediction artifacts are logged


### 8. Add runtime logging

Decision:

- Add standard Python logging with configurable verbosity.

Why:

- The user requested more information during runs.

Implementation:

- New helper: `src/logging_utils.py`
- YAML config:
  ```yaml
  logging:
    level: INFO
  ```
- Progress logs added to query generation, dataset creation, training, and prediction.


### 9. Add smoke-test config

Decision:

- Create `configs/oceantaco_smoke_test.yaml`.

Why:

- The user wanted a fast way to validate the pipeline without changing the main experiment config.

Smoke-test characteristics:

- North Pacific regional bbox matching the main config
- short but nontrivial date ranges
- low but nontrivial query count
- `64x64` grid
- smaller model
- batch size `1`
- `1` epoch


### 10. Prediction behavior for all-missing inputs

Decision:

- Keep training strict, but allow prediction to run even when all input variables are missing for a test window.

Why:

- The user reported all prediction batches being skipped over a test window because inputs were missing.
- For sparse satellite windows, zero-filled fallback inference is better than dropping every batch.

Implementation:

- `batch_to_model_tensors(..., allow_empty_inputs=True)` can build zero-filled inputs if all configured inputs are missing.
- `simvp_predict_ssh.py` enables this only for prediction.
- Config flag:
  `prediction.allow_empty_inputs: true`
- Training and validation still skip unusable batches.


### 11. Add a query-inspection debugging script

Decision:

- Add a standalone script to inspect saved query files and summarize what they contain.

Why:

- The user reported that datasets were coming back empty and wanted a way to debug whether the issue starts at query generation or later during dataset loading.
- Query files are one of the fastest things to inspect when debugging empty OceanTACO datasets.

Implementation:

- Added `analyze_queries.py` at the repo root.
- The script recursively scans a query directory and supports common saved formats:
  - `.json`
  - `.jsonl`
  - `.csv`
  - `.parquet`
- It summarizes:
  - number of records
  - available keys
  - bbox coverage
  - date-range coverage
  - invalid bbox counts
  - duplicate record signatures
  - sample records
- It can also optionally reload stored queries through OceanTACO and inspect
  actual sample variables for:
  - `None`
  - all-NaN
  - all-non-finite
  - all-zero values

Design choice:

- The analyzer is intentionally format-tolerant because this workspace did not currently contain saved query files to lock onto one exact OceanTACO export schema.
- The script includes many inline comments because the user explicitly asked for that.
- It now also reports OceanTACO file-index match counts per query so we can
  distinguish between:
  - no candidate files being found at all
  - candidate files existing but variables still loading as empty


### 12. Add a prediction-plotting debugging script

Decision:

- Add a standalone plotting utility for saved prediction `.npz` files.

Why:

- The user wanted a way to inspect model outputs visually for either a single file or a whole directory.
- This is useful for debugging whether prediction files contain sensible maps, whether targets line up spatially, and whether outputs are degenerate.

Implementation:

- Added `plot_predictions.py` at the repo root.
- It accepts either:
  - one `.npz` file
  - one directory containing many `.npz` files
- It plots:
  - prediction
  - target
  - prediction minus target
- It supports:
  - `--time-index`
  - `--channel-index`
  - `--output-dir`
  - `--dpi`

Design choice:

- The plotting code is tolerant of `2D`, `3D`, and `4D` arrays because saved predictions may have shapes like:
  - `[H, W]`
  - `[C, H, W]`
  - `[T, H, W]`
  - `[T, C, H, W]`
- If bbox is present in the `.npz`, it is used as the longitude/latitude extent in plots.


## Major code structure and behavior

### `src/config_utils.py`

Purpose:

- repository root resolution
- config loading
- path resolution
- directory creation
- split lookup

Key functions:

- `repo_root()`
- `load_config()`
- `resolve_path()`
- `ensure_dir()`
- `get_split_config()`

Behavior:

- Adds internal keys:
  - `__config_path__`
  - `__repo_root__`


### `src/oceantaco.py`

Purpose:

- OceanTACO query generation
- dataset construction
- bbox handling
- tensor conversion into model input / target shapes
- prediction metadata extraction

Important functions:

- `_import_oceantaco()`
- `_resolve_bbox()`
- `_split_region_bboxes()`
- `_build_patch_size()`
- `build_queries()`
- `build_dataset()`
- `get_collate_fn()`
- `_normalise_tensor()`
- `_prepare_variable_tensor()`
- `batch_to_model_tensors()`
- `prediction_records()`

Important behavior:

- Bbox is treated as `(lon_min, lon_max, lat_min, lat_max)`.
- Queries are built with OceanTACO `QueryGenerator`.
- Training splits use `generate_training_queries`.
- Validation / test splits use `generate_eval_queries`.
- Query files are saved under `queries/<split>`.
- Dataset uses a patched subclass of `OceanTACODataset`.

Patched dataset behavior:

- OceanTACO sometimes returns malformed fragments that are not 2D grids.
- These could crash merging in upstream `GridMerger`.
- The patched `_load_variable()` skips malformed fragments instead of crashing.

Specific robustness behavior:

- Catches `IndexError` and `ValueError` around `merger.add(...)`.
- If a fragment is malformed, it is ignored.
- Valid fragments are merged or temporally aggregated.

Normalization behavior:

- Optional per-variable normalization from config.
- Supports:
  - `mean`
  - `std`
  - `min_valid`
  - `mask_zeros`

Temporal behavior:

- If a variable is shorter than `sequence_length`, it is padded by repeating the last frame.
- If it is longer, it is trimmed.

Prediction-specific behavior:

- If all inputs are missing and `allow_empty_inputs=True`, zero-filled input tensors are created instead of skipping.


### `simvp_ddp_training.py`

Purpose:

- training loop
- validation loop
- checkpointing
- loss CSV logging
- MLflow metric logging

Important behaviors:

- Uses `torch.amp.GradScaler(device.type, enabled=use_amp)`.
- Uses `torch.amp.autocast(device_type=device.type, enabled=use_amp)`.
- Skips empty-input batches during training and validation.
- Saves checkpoint per epoch to `model_weights/`.
- Logs `epoch`, `train_loss`, `val_loss` to CSV in `loss_logs/`.

MLflow behavior:

- Starts training run
- Logs config and dataset sizes
- Logs per-epoch metrics
- Optionally logs checkpoints
- Optionally logs loss CSV

Runtime logging behavior:

- logs config path
- logs artifact directories
- logs dataset sizes
- logs device and AMP mode
- logs checkpoint resume information
- logs epoch starts and finishes


### `simvp_predict_ssh.py`

Purpose:

- inference over an OceanTACO split
- output `.npz` files with prediction and target
- MLflow logging for inference artifacts/metrics

Important behaviors:

- Loads configured split or CLI override
- Writes compressed `.npz` per sample to `predictions/`
- Uses zero-filled fallback inputs if `prediction.allow_empty_inputs: true` and all inputs are missing
- Tracks:
  - `saved_prediction_files`
  - `skipped_prediction_batches`
  - `zero_input_prediction_batches`

Runtime logging behavior:

- logs config path
- logs dataset size
- logs checkpoint path
- logs output directory
- warns when a batch has no observed inputs and zero-fill fallback is used
- logs saved batch counts


### `analyze_queries.py`

Purpose:

- inspect saved query exports before dataset loading
- debug empty-data situations by checking whether queries themselves look sane

What it reports:

- readable files under a query directory
- record counts per file
- metadata if present
- bbox and date-range coverage
- invalid bbox counts
- duplicate record counts
- common keys across records
- sample records
- optional variable-level data flags when `--check-data` is used

Why it matters:

- If dataset loading later returns empty inputs / targets, this script helps determine whether:
  - query files are missing
  - query files are empty
  - query ranges are outside intended dates
  - bboxes are malformed
  - all queries are duplicates
  - schema differs from expectation
  - specific input/target variables load as `None`
  - specific variables are entirely NaN / non-finite / zero after loading


### `plot_predictions.py`

Purpose:

- visualize saved `.npz` predictions from the prediction pipeline
- inspect one file or batch-convert an entire directory

What it plots:

- prediction field
- target field
- prediction minus target

Behavior:

- accepts a file or directory
- writes `.png` outputs
- uses a shared color scale for prediction and target
- uses a symmetric color scale for the error map
- uses bbox as geographic extent if present
- selects slices with `--time-index` and `--channel-index`


### `src/pytorch_losses.py`

Important fix:

- `torch_masked_mse(...)` returns a graph-connected zero when all targets are masked:
  ```python
  return y_pred.sum() * 0.0
  ```

Why:

- A plain detached zero caused:
  `RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn`


### `src/simvp_model.py`

Important direction:

- A configurable SimVP variant was added earlier to support current OceanTACO input/target layouts.

Relevant model class:

- `SimVP_Model_no_skip_configurable`

Reason:

- Needed flexible input and output channel counts.


### `src/mlflow_utils.py`

Purpose:

- Optional MLflow wrapper
- flatten config for param logging
- artifact logging
- metric logging

Key pieces:

- `flatten_config()`
- `MLflowTracker`

Behavior:

- MLflow is optional.
- If enabled but package missing, raises a clear `ImportError`.
- Relative tracking URIs are resolved relative to repo root.


### `src/logging_utils.py`

Purpose:

- central logging setup

Behavior:

- Reads `logging.level` from config
- configures standard logging format:
  `%(asctime)s | %(levelname)s | %(name)s | %(message)s`


## Config files

### `configs/oceantaco.yaml`

Main experiment config.

Important sections:

- `paths`
- `logging`
- `tracking.mlflow`
- `oceantaco`
- `regions`
- `queries`
- `data`
- `splits`
- `model`
- `training`
- `prediction`

Current notable defaults:

- `logging.level: INFO`
- `prediction.allow_empty_inputs: true`


### `configs/oceantaco_smoke_test.yaml`

Fast validation config.

Characteristics:

- North Pacific regional bbox matching the main config
- moderate patch size for quicker but still meaningful coverage
- short split windows with a little more temporal coverage
- low but nontrivial query count
- `64x64` grid
- lighter model
- `epochs: 1`
- `amp: false`
- `prediction.allow_empty_inputs: true`


## .gitignore decisions

Added ignore rules for:

- `__pycache__/`
- `**/__pycache__/`
- `*.py[cod]`
- `**/*.py[cod]`
- `*.pyc.*`
- `.pytest_cache/`
- `.coverage`
- `.coverage.*`
- `htmlcov/`
- `queries/`
- `model_weights/`
- `loss_logs/`
- `predictions/`
- `mlruns/`
- `*.pt`
- `*.pth`
- `*.npz`
- `*.npy`
- `*.parquet`
- `.DS_Store`

Additional Git decision:

- Already-tracked `__pycache__` / `.pyc` files were removed from Git index with `git rm --cached`.


## Runtime issues encountered and how they were handled

### 1. Deprecated AMP warnings

Observed:

- `torch.cuda.amp.GradScaler(...)` deprecated
- `torch.cuda.amp.autocast(...)` deprecated

Fix:

- Switched to `torch.amp.GradScaler(...)`
- Switched to `torch.amp.autocast(...)`


### 2. BatchNorm mismatch in SimVP

Observed error:

- `RuntimeError: running_mean should contain 8 elements not 40`

Interpretation:

- OceanTACO sometimes returned fewer temporal slices than the model expected.
- Model expected `sequence_length=5`, but batch could effectively behave like `T=1`.

Fix:

- Temporal padding/trimming was added in `src/oceantaco.py`.


### 3. Backward pass failure due to detached zero loss

Observed error:

- `RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn`

Cause:

- Masked loss returned a fresh standalone zero tensor when no valid targets were present.

Fix:

- Return graph-connected zero with `y_pred.sum() * 0.0`.


### 4. Empty-input training batches

Observed error:

- `RuntimeError: Received a batch without any input tensors.`

Interpretation:

- OceanTACO can return windows where all configured inputs are missing.

Decision and fix:

- For training and validation, skip those batches instead of failing.
- Training logs `skipped_train_batches` and `skipped_val_batches`.


### 5. OceanTACO merge crash

Observed error:

- `IndexError: tuple index out of range`

Location:

- inside OceanTACO dataset merging / `GridMerger`

Interpretation:

- Some fragments are malformed / not shaped like expected gridded arrays.

Fix:

- Patched OceanTACO dataset wrapper in `src/oceantaco.py`
- Skip malformed fragments instead of crashing


### 6. Prediction skipping every batch

Observed behavior:

- `Skipping empty prediction batch` for all batches over a 21-day test period

Interpretation:

- All configured input variables for those windows were missing.
- Inference path treated that as an unusable batch and skipped everything.

Decision and fix:

- Keep training strict
- allow prediction to use zero-filled fallback inputs
- add `prediction.allow_empty_inputs: true`
- log `zero_input_prediction_batches`


## Bbox / coordinate handling

Important clarification given to the user:

- The bbox is already interpreted as lat-lon bounds.
- There is no projection transform from another CRS inside the current code.

Relevant code:

- `_resolve_bbox(...)` returns `(lon_min, lon_max, lat_min, lat_max)`
- query generation passes bbox directly to OceanTACO
- `_load_variable(...)` constructs coordinate arrays using `np.linspace(...)`

If the user asks where bbox becomes per-pixel coordinates:

- It is effectively in `src/oceantaco.py` inside `_load_variable(...)`, where `lats_out` and `lons_out` are generated from bbox and patch size.


## Tests added / maintained

### `tests/test_config_utils.py`

Covers:

- config loading adds internal paths
- resolve_path behavior
- ensure_dir behavior
- split lookup
- config flattening for MLflow


### `tests/test_oceantaco_adapter.py`

Covers:

- named bbox preset resolution
- `global` bbox handling
- temporal padding of singleton time dimension
- trimming overlong time dimension
- prediction record target date selection from `target_index`
- query generation wiring with fake OceanTACO modules
- zero-filled prediction fallback when all inputs are missing


### `tests/test_training_script.py`

Covers:

- `build_model(...)` output shape
- `evaluate(...)` on CPU
- skipping empty-input batches in evaluation
- masked MSE keeps gradients for all-zero target batch
- MLflow tracker is safe when disabled


## Verification and commands actually run in this environment

The following checks were run during development.

### Static checks / compile checks

Ran successfully multiple times:

- `python3 -m compileall simvp_ddp_training.py simvp_predict_ssh.py src/mlflow_utils.py tests`
- `python3 -m compileall simvp_predict_ssh.py src/oceantaco.py tests/test_oceantaco_adapter.py`
- `python3 -m compileall simvp_ddp_training.py simvp_predict_ssh.py src/logging_utils.py src/oceantaco.py`
- `python3 -m compileall simvp_ddp_training.py simvp_predict_ssh.py src/mlflow_utils.py src/logging_utils.py src/oceantaco.py tests`
- `python3 -m compileall analyze_queries.py`
- `python3 -m compileall plot_predictions.py`

These passed.


### YAML validation

Ran successfully:

- YAML parse check for:
  - `configs/oceantaco.yaml`
  - `configs/oceantaco_smoke_test.yaml`


### Git inspection / cleanup

Ran:

- `git status --short`
- `git ls-files | rg "__pycache__|\\.pyc$|\\.pyo$|\\.pyd$"`
- `git check-ignore -v ...`

Also ran:

- `git rm --cached` on tracked bytecode / `__pycache__` files


### Repo inspection and searches

Used repeatedly:

- `rg`
- `sed`
- `nl`
- `git diff`


### Query debugging utility checks

Ran:

- recursive inspection of `queries/`

Result in this workspace:

- no saved query files were present at the time of inspection

Consequence:

- `analyze_queries.py` was written to be tolerant of multiple file formats and schemas rather than assuming one observed local export layout


### Prediction plotting utility checks

Ran:

- `python3 -m compileall plot_predictions.py`

This passed.

Note:

- No local `.npz` prediction files were available in this workspace to perform a real plotting run against saved predictions.


## Things not fully verified in this environment

These were not fully exercised end-to-end here:

- full OceanTACO dataset retrieval against real remote data
- end-to-end model training on real data
- end-to-end prediction on real data
- full pytest execution against the complete runtime stack in every turn
- Docker image build

Reason:

- local environment constraints and package/runtime availability were not always guaranteed


## User-facing workflow changes introduced

### Training

Run with:

- `python3 simvp_ddp_training.py --config configs/oceantaco.yaml`

Smoke test:

- `python3 simvp_ddp_training.py --config configs/oceantaco_smoke_test.yaml`


### Prediction

Run with:

- `python3 simvp_predict_ssh.py --config configs/oceantaco.yaml --checkpoint <checkpoint>`

Smoke test:

- `python3 simvp_predict_ssh.py --config configs/oceantaco_smoke_test.yaml --checkpoint model_weights/simvp_oceantaco_smoke_epoch0.pt`


## Important caveats and assumptions

- Zero-input fallback in prediction is a pragmatic choice for sparse satellite windows. It allows the pipeline to run, but those outputs may represent model prior behavior more than observation-driven inference.
- Training still skips no-input batches by design.
- Query generation is bbox-based and not projection-based.
- The current pipeline assumes OceanTACO samples can be normalized and resampled onto the configured grid size.
- A later debugging hypothesis was that an explicit raw Hugging Face
  `.../resolve/main/` URL for `taco_path` was not the right kind of dataset root
  for OceanTACO’s `tacoreader.load(...)`. The adapter was updated to fall back
  to OceanTACO’s `HF_DEFAULT_URL` when it detects that raw URL pattern.


## Documentation changes made

`README.md` was updated to document:

- OceanTACO-based workflow
- MLflow usage
- smoke-test config
- runtime logging config
- current training / prediction commands


## If resuming work later, important files to inspect first

- `configs/oceantaco.yaml`
- `configs/oceantaco_smoke_test.yaml`
- `analyze_queries.py`
- `plot_predictions.py`
- `src/oceantaco.py`
- `simvp_ddp_training.py`
- `simvp_predict_ssh.py`
- `src/mlflow_utils.py`
- `src/logging_utils.py`
- `tests/test_oceantaco_adapter.py`
- `tests/test_training_script.py`


## Summary

The repo has been reshaped into a config-driven OceanTACO + SimVP pipeline with:

- bbox-driven split selection
- optional global runs
- YAML configuration
- pytest coverage
- Docker support
- MLflow integration
- structured runtime logging
- smoke-test config
- robust handling of malformed OceanTACO fragments
- robust handling of sparse / missing satellite inputs

The most important design split is:

- training stays strict and skips unusable batches
- prediction is more permissive and can zero-fill entirely missing inputs
