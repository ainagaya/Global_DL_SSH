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
  ./run_experiment.sh --resume=aXXX [additional training args...]

Example:
  ./run_experiment.sh configs/oceantaco_smoke_test.yaml
  ./run_experiment.sh configs/oceantaco_gulf_stream.yaml
  ./run_experiment.sh --resume=a002
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

latest_experiment_checkpoint() {
  local weights_dir="$1"
  local checkpoint_name="$2"

  ls -1 "$weights_dir"/"${checkpoint_name}"_epoch*.pt 2>/dev/null | sort -V | tail -n 1
}

RESUME_ID=""
BASE_CONFIG=""
TRAINING_ARGS=()
EXPECT_RESUME_VALUE=0

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

for arg in "$@"; do
  if (( EXPECT_RESUME_VALUE )); then
    RESUME_ID="$arg"
    EXPECT_RESUME_VALUE=0
    continue
  fi

  case "$arg" in
    -h|--help)
      usage
      exit 0
      ;;
    --resume=*)
      if [[ -n "$RESUME_ID" ]]; then
        echo "--resume can only be provided once." >&2
        exit 1
      fi
      RESUME_ID="${arg#--resume=}"
      ;;
    --resume)
      if [[ -n "$RESUME_ID" ]]; then
        echo "--resume can only be provided once." >&2
        exit 1
      fi
      EXPECT_RESUME_VALUE=1
      ;;
    *)
      if [[ -z "$BASE_CONFIG" && -z "$RESUME_ID" && "$arg" != -* ]]; then
        BASE_CONFIG="$arg"
      else
        TRAINING_ARGS+=("$arg")
      fi
      ;;
  esac
done

if (( EXPECT_RESUME_VALUE )); then
  echo "--resume requires an experiment id, for example --resume=a002." >&2
  exit 1
fi

if [[ -n "$BASE_CONFIG" && -n "$RESUME_ID" ]]; then
  echo "Choose either a new base config or --resume=aXXX, not both." >&2
  exit 1
fi

if [[ -z "$BASE_CONFIG" && -z "$RESUME_ID" ]]; then
  usage
  exit 1
fi

for arg in "${TRAINING_ARGS[@]}"; do
  case "$arg" in
    --config|--config=*|--checkpoint|--checkpoint=*)
      echo "run_experiment.sh manages --config and --checkpoint internally; remove $arg." >&2
      exit 1
      ;;
  esac
done

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

require_clean_git_tree

EXPERIMENTS_ROOT="$REPO_ROOT/experiments"

if [[ -n "$RESUME_ID" ]]; then
  if [[ ! "$RESUME_ID" =~ ^a[0-9]{3}$ ]]; then
    echo "Invalid experiment id for --resume: $RESUME_ID" >&2
    exit 1
  fi
  EXPERIMENT_ID="$RESUME_ID"
  echo "Resuming experiment with id: $EXPERIMENT_ID"
else
  EXPERIMENT_ID="$(next_experiment_id "$EXPERIMENTS_ROOT")"
fi

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
EXPERIMENTS_MD="$REPO_ROOT/EXPERIMENTS.md"

if [[ -n "$RESUME_ID" ]]; then
  if [[ ! -d "$EXPERIMENT_DIR" ]]; then
    echo "Experiment directory not found for --resume=$RESUME_ID: $EXPERIMENT_DIR" >&2
    exit 1
  fi
  if [[ ! -f "$RUNTIME_CONFIG" ]]; then
    echo "Runtime config not found for --resume=$RESUME_ID: $RUNTIME_CONFIG" >&2
    exit 1
  fi
else
  if [[ ! -f "$BASE_CONFIG" ]]; then
    echo "Config file not found: $BASE_CONFIG" >&2
    exit 1
  fi
fi

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

if [[ -z "$RESUME_ID" ]]; then
  cp "$BASE_CONFIG" "$BASE_CONFIG_COPY"
fi

# We rewrite the runtime config instead of mutating the source config so the
# experiment remains self-contained and reproducible even if the original YAML
# changes later on.
if [[ -z "$RESUME_ID" ]]; then
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
fi

if [[ -z "$RESUME_ID" ]]; then
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
fi

if [[ -n "$RESUME_ID" ]]; then
  RESUME_CHECKPOINT="$(latest_experiment_checkpoint "$WEIGHTS_DIR" "$EXPERIMENT_ID")"
  if [[ -z "$RESUME_CHECKPOINT" ]]; then
    echo "No checkpoint found to resume in $WEIGHTS_DIR" >&2
    exit 1
  fi
  echo "Resuming experiment $EXPERIMENT_ID"
  echo "Resume checkpoint: $RESUME_CHECKPOINT"
else
  RESUME_CHECKPOINT=""
  echo "Created experiment $EXPERIMENT_ID"
fi

append_experiments_md_start() {
  if [[ ! -f "$EXPERIMENTS_MD" ]]; then
    return
  fi

  cat >> "$EXPERIMENTS_MD" <<EOF

### $EXPERIMENT_ID

- Created: $CREATED_AT_UTC
- Git: \`$GIT_COMMIT\` on branch \`$GIT_BRANCH\`
- Base config: \`$(realpath "$BASE_CONFIG")\`
- Experiment dir: \`experiments/$EXPERIMENT_ID\`
- Status: started by \`run_experiment.sh\`; results pending.
- Scientific note: fill in after training and prediction complete.
EOF
}

append_experiments_md_finish() {
  if [[ ! -f "$EXPERIMENTS_MD" ]]; then
    return
  fi

  local finished_at_utc
  local prediction_count
  finished_at_utc="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  prediction_count="$(find "$PREDICTIONS_DIR" -maxdepth 1 -type f -name '*.npz' | wc -l | tr -d ' ')"

  cat >> "$EXPERIMENTS_MD" <<EOF
- Completed: $finished_at_utc
- Outputs: latest checkpoint \`experiments/$EXPERIMENT_ID/weights/$(basename "$LATEST_CHECKPOINT")\`, loss CSV \`experiments/$EXPERIMENT_ID/logs/${EXPERIMENT_ID}_losses.csv\`, prediction files: $prediction_count
EOF

  if [[ -f "$PREDICTIONS_DIR/reserved_l3_ssh_metrics_summary.json" ]]; then
    echo "- Metrics: \`experiments/$EXPERIMENT_ID/predictions/reserved_l3_ssh_metrics_summary.json\`" >> "$EXPERIMENTS_MD"
  fi

  echo "- Follow-up: replace this generated ledger note with a short result interpretation." >> "$EXPERIMENTS_MD"
}

append_experiments_md_start

echo "Created experiment $EXPERIMENT_ID"
echo "Commit: $GIT_COMMIT"
echo "Experiment directory: $EXPERIMENT_DIR"
echo "Frozen runtime config: $RUNTIME_CONFIG"
echo "Downloading required OceanTACO data..."
python3 download_oceantaco_local.py --config "$RUNTIME_CONFIG" 2>&1 | tee -a "$EXPERIMENT_DIR/logs/download.log"

echo "Launching training..."
if [[ -n "$RESUME_CHECKPOINT" ]]; then
  python3 simvp_ddp_training.py --config "$RUNTIME_CONFIG" --checkpoint "$RESUME_CHECKPOINT" "${TRAINING_ARGS[@]}" 2>&1 | tee -a "$EXPERIMENT_DIR/logs/training.log"
else
  python3 simvp_ddp_training.py --config "$RUNTIME_CONFIG" "${TRAINING_ARGS[@]}" 2>&1 | tee -a "$EXPERIMENT_DIR/logs/training.log"
fi

LATEST_CHECKPOINT="$(latest_experiment_checkpoint "$WEIGHTS_DIR" "$EXPERIMENT_ID")"
if [[ -z "$LATEST_CHECKPOINT" ]]; then
  echo "No checkpoint was produced in $WEIGHTS_DIR" >&2
  exit 1
fi

LOSS_CSV="$LOGS_DIR/${EXPERIMENT_ID}_losses.csv"
LOSS_PLOT="$ANALYSIS_DIR/${EXPERIMENT_ID}_losses.png"

echo "Launching inference with checkpoint: $LATEST_CHECKPOINT"
python3 simvp_predict_ssh.py --config "$RUNTIME_CONFIG" --checkpoint "$LATEST_CHECKPOINT" 2>&1 | tee -a "$EXPERIMENT_DIR/logs/prediction.log"

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
append_experiments_md_finish
echo "Artifacts:"
echo "  weights: $WEIGHTS_DIR"
echo "  logs: $LOGS_DIR"
echo "  predictions: $PREDICTIONS_DIR"
echo "  analysis: $ANALYSIS_DIR"
echo "    prediction files: $PREDICTION_FILE_PLOTS_DIR"
echo "    prediction regions: $REGIONAL_PLOTS_DIR"
echo "    prediction mosaics: $MOSAIC_PLOTS_DIR"
