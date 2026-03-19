# Global_DL_SSH refactor scaffold

This package is a clean-room refactor scaffold for the Python scripts changed in PR #1 of `ainagaya/Global_DL_SSH`.

It does **not** try to rewrite the scientific kernels in the legacy `src/` directory. Instead, it:

- moves runtime parameters into YAML config files
- removes hard-coded paths and magic constants from executable scripts
- introduces reusable modules for IO, TFRecord parsing, plotting, prediction and orchestration
- keeps the option to call the existing scientific/model code through explicit adapter imports
- replaces top-level side effects with `main()` entrypoints and structured logging

## Suggested integration

Place this scaffold at the repository root, then:

1. keep the existing legacy scientific modules in `src/`
2. migrate scripts one by one to the wrappers in `scripts/`
3. point the config files to your real data, checkpoints, OceanTACO URL and model/loss callables
4. once stable, move the remaining legacy logic out of `src/` into this package

## Main design decisions

- No `sys.path.append(...)`
- No wildcard imports
- No hard-coded filesystem layout in code
- No global execution at import time
- Shared preprocessing and normalization code lives in one place
- Long-running scripts are configuration-driven and testable
- External scientific/model functions are injected through import strings such as `src.simvp_model:SimVP_Model_no_skip_sst`

## Coverage

The scaffold provides clean replacements for the PR-touched scripts:

- `inspect_nc.py`
- `load_files.py`
- `merge_maps.py`
- `plot_losses.py`
- `plot_maps.py`
- `pre_process_testing.py`
- `pre_process_training.py`
- `simvp_ddp_training.py`
- `simvp_predict_ssh.py`
- `test.py`
