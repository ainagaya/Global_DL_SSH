#!/usr/bin/env bash
set -euo pipefail

# Minimal experiment manager for reproducible training runs.
#
# What it does:
# - refuses to run from a dirty git worktree
# - assigns the next experiment id under ./experiments as a000, a001, ...
# - records the exact git commit used for the run
# - stores both an exact copy of the user config and a runtime config snapshot
# - rewrites all output paths so weights, logs, predictions, queries, and mlflow
#   artifacts land inside the experiment directory
# - downloads the OceanTACO files required by the frozen runtime config
# - launches training, inference, and all available analysis plots

usage() {
  cat <<'EOF'
Usage:
  ./run_experiment.sh <base-config.yaml> [additional training args...]

Example:
  ./run_experiment.sh configs/oceantaco_smoke_test.yaml
  ./run_experiment.sh configs/oceantaco_gulf_stream.yaml
EOF
}

require_clean_git_tree() {
  if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "Refusing to run: git worktree is dirty. Commit or stash changes first." >&2
    exit 1
  fi
}

next_experiment_id() {
  local experiments_root="$1"
  local last_id=-1
  local path

  mkdir -p "$experiments_root"
  for path in "$experiments_root"/a[0-9][0-9][0-9]; do
    if [[ ! -d "$path" ]]; then
      continue
    fi
    local name
    name="$(basename "$path")"
    local numeric="${name#a}"
    if [[ "$numeric" =~ ^[0-9]{3}$ ]] && (( 10#$numeric > last_id )); then
      last_id=$((10#$numeric))
    fi
  done

  printf "a%03d" "$((last_id + 1))"
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

BASE_CONFIG="$1"
shift

if [[ ! -f "$BASE_CONFIG" ]]; then
  echo "Config file not found: $BASE_CONFIG" >&2
  exit 1
fi

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

require_clean_git_tree

EXPERIMENTS_ROOT="$REPO_ROOT/experiments"
EXPERIMENT_ID="$(next_experiment_id "$EXPERIMENTS_ROOT")"
EXPERIMENT_DIR="$EXPERIMENTS_ROOT/$EXPERIMENT_ID"
WEIGHTS_DIR="$EXPERIMENT_DIR/weights"
LOGS_DIR="$EXPERIMENT_DIR/logs"
PREDICTIONS_DIR="$EXPERIMENT_DIR/predictions"
QUERIES_DIR="$EXPERIMENT_DIR/queries"
MLRUNS_DIR="$EXPERIMENT_DIR/mlruns"
ANALYSIS_DIR="$EXPERIMENT_DIR/analysis"
PREDICTION_FILE_PLOTS_DIR="$ANALYSIS_DIR/prediction_files"
REGIONAL_PLOTS_DIR="$ANALYSIS_DIR/prediction_regions"
MOSAIC_PLOTS_DIR="$ANALYSIS_DIR/prediction_mosaics"
METADATA_FILE="$EXPERIMENT_DIR/${EXPERIMENT_ID}_metadata.txt"
BASE_CONFIG_COPY="$EXPERIMENT_DIR/${EXPERIMENT_ID}_base_config.yaml"
RUNTIME_CONFIG="$EXPERIMENT_DIR/${EXPERIMENT_ID}_runtime_config.yaml"

mkdir -p \
  "$WEIGHTS_DIR" \
  "$LOGS_DIR" \
  "$PREDICTIONS_DIR" \
  "$QUERIES_DIR" \
  "$MLRUNS_DIR" \
  "$ANALYSIS_DIR" \
  "$PREDICTION_FILE_PLOTS_DIR" \
  "$REGIONAL_PLOTS_DIR" \
  "$MOSAIC_PLOTS_DIR"

GIT_COMMIT="$(git rev-parse HEAD)"
GIT_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
CREATED_AT_UTC="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

cp "$BASE_CONFIG" "$BASE_CONFIG_COPY"

# We rewrite the runtime config instead of mutating the source config so the
# experiment remains self-contained and reproducible even if the original YAML
# changes later on.
python3 - "$BASE_CONFIG" "$RUNTIME_CONFIG" "$EXPERIMENT_ID" "$EXPERIMENT_DIR" "$GIT_COMMIT" "$CREATED_AT_UTC" "$MLRUNS_DIR" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

import yaml

base_config_path = Path(sys.argv[1])
runtime_config_path = Path(sys.argv[2])
experiment_id = sys.argv[3]
experiment_dir = Path(sys.argv[4])
git_commit = sys.argv[5]
created_at_utc = sys.argv[6]
mlruns_dir = Path(sys.argv[7])

with base_config_path.open("r", encoding="utf-8") as handle:
    config = yaml.safe_load(handle)

config.setdefault("paths", {})
config["paths"]["weights_dir"] = str(experiment_dir / "weights")
config["paths"]["logs_dir"] = str(experiment_dir / "logs")
config["paths"]["predictions_dir"] = str(experiment_dir / "predictions")
config["paths"]["queries_dir"] = str(experiment_dir / "queries")

config.setdefault("training", {})
config["training"]["checkpoint_name"] = experiment_id

config.setdefault("tracking", {})
config["tracking"].setdefault("mlflow", {})
config["tracking"]["mlflow"]["tracking_uri"] = str(mlruns_dir)

config["experiment"] = {
    "id": experiment_id,
    "git_commit": git_commit,
    "created_at_utc": created_at_utc,
    "base_config_path": str(base_config_path.resolve()),
}

with runtime_config_path.open("w", encoding="utf-8") as handle:
    yaml.safe_dump(config, handle, sort_keys=False)
PY

cat > "$METADATA_FILE" <<EOF
experiment_id: $EXPERIMENT_ID
git_commit: $GIT_COMMIT
git_branch: $GIT_BRANCH
created_at_utc: $CREATED_AT_UTC
base_config: $(realpath "$BASE_CONFIG")
base_config_copy: $BASE_CONFIG_COPY
runtime_config: $RUNTIME_CONFIG
experiment_dir: $EXPERIMENT_DIR
EOF

echo "Created experiment $EXPERIMENT_ID"
echo "Commit: $GIT_COMMIT"
echo "Experiment directory: $EXPERIMENT_DIR"
echo "Frozen runtime config: $RUNTIME_CONFIG"
echo "Downloading required OceanTACO data..."
python3 download_oceantaco_local.py --config "$RUNTIME_CONFIG" 2>&1 | tee "$EXPERIMENT_DIR/logs/download.log"

echo "Launching training..."
python3 simvp_ddp_training.py --config "$RUNTIME_CONFIG" "$@" 2>&1 | tee "$EXPERIMENT_DIR/logs/training.log"

LATEST_CHECKPOINT="$(ls -1 "$WEIGHTS_DIR"/"${EXPERIMENT_ID}"_epoch*.pt 2>/dev/null | sort -V | tail -n 1)"
if [[ -z "$LATEST_CHECKPOINT" ]]; then
  echo "No checkpoint was produced in $WEIGHTS_DIR" >&2
  exit 1
fi

LOSS_CSV="$LOGS_DIR/${EXPERIMENT_ID}_losses.csv"
LOSS_PLOT="$ANALYSIS_DIR/${EXPERIMENT_ID}_losses.png"

echo "Launching inference with checkpoint: $LATEST_CHECKPOINT"
python3 simvp_predict_ssh.py --config "$RUNTIME_CONFIG" --checkpoint "$LATEST_CHECKPOINT" 2>&1 | tee "$EXPERIMENT_DIR/logs/prediction.log"

if [[ -f "$LOSS_CSV" ]]; then
  echo "Plotting training curves from $LOSS_CSV"
  python3 plot_training_curves.py "$LOSS_CSV" --output "$LOSS_PLOT"
else
  echo "Loss CSV not found, skipping training-curve plot: $LOSS_CSV"
fi

if compgen -G "$PREDICTIONS_DIR/*.npz" > /dev/null; then
  echo "Generating per-file prediction plots"
  python3 plot_predictions.py "$PREDICTIONS_DIR" --output-dir "$PREDICTION_FILE_PLOTS_DIR"

  echo "Generating regional prediction plots for all target dates"
  python3 plot_prediction_regions.py "$PREDICTIONS_DIR" --all-dates --output-dir "$REGIONAL_PLOTS_DIR"

  echo "Generating merged prediction mosaic plots for all target dates"
  python3 plot_prediction_mosaics.py "$PREDICTIONS_DIR" --all-dates --output-dir "$MOSAIC_PLOTS_DIR"
else
  echo "No prediction .npz files found, skipping prediction plots"
fi

echo "Experiment $EXPERIMENT_ID finished"
echo "Artifacts:"
echo "  weights: $WEIGHTS_DIR"
echo "  logs: $LOGS_DIR"
echo "  predictions: $PREDICTIONS_DIR"
echo "  analysis: $ANALYSIS_DIR"
echo "    prediction files: $PREDICTION_FILE_PLOTS_DIR"
echo "    prediction regions: $REGIONAL_PLOTS_DIR"
echo "    prediction mosaics: $MOSAIC_PLOTS_DIR"
