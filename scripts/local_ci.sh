#!/usr/bin/env bash

# Local runner for CI-like checks.
# Runs lint, unit, property, e2e, and coverage tests and writes logs to local_reports/.

set -u

show_help() {
  cat <<'EOF'
Usage: bash scripts/local_ci.sh [options]

Runs a subset or all CI checks locally and writes logs to local_reports/<timestamp>/.

Options:
  --no-lint           Skip lint checks
  --no-unit           Skip unit tests
  --no-property       Skip property tests
  --no-e2e            Skip e2e tests
  --with-no-nan       Also run e2e filtered with -k "no_nan" (skip if none)
  --no-coverage       Skip coverage run
  --with-torch        Also run framework tests filtered by "torch" (if torch imports)
  --with-jax          Also run framework tests filtered by "jax" (if jax imports)
  --only LINT|UNIT|PROPERTY|E2E|E2E_NONAN|COVERAGE
                       Run only the specified step (can be repeated)
  -h, --help          Show this help

Environment:
  CI=1                Set automatically to mimic CI timing tolerances in tests
  PYTEST_E2E_MARK     Extra -m expression for e2e (default: not slow)
  PYTEST_ADDOPTS      Extra pytest options appended to every run
EOF
}

ONLY_STEPS=()
DO_LINT=1
DO_UNIT=1
DO_PROPERTY=1
DO_E2E=1
DO_E2E_NONAN=0
DO_COVERAGE=1
DO_TORCH=0
DO_JAX=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-lint) DO_LINT=0; shift ;;
    --no-unit) DO_UNIT=0; shift ;;
    --no-property) DO_PROPERTY=0; shift ;;
    --no-e2e) DO_E2E=0; shift ;;
    --with-no-nan) DO_E2E_NONAN=1; shift ;;
    --no-coverage) DO_COVERAGE=0; shift ;;
    --with-torch) DO_TORCH=1; shift ;;
    --with-jax) DO_JAX=1; shift ;;
    --only) shift; ONLY_STEPS+=("$1"); shift ;;
    -h|--help) show_help; exit 0 ;;
    *) echo "Unknown option: $1" >&2; show_help; exit 2 ;;
  esac
done

timestamp() {
  date +%Y%m%d_%H%M%S
}

OUT_ROOT="local_reports"
OUT_DIR="$OUT_ROOT/$(timestamp)"
mkdir -p "$OUT_DIR"

export CI=1
E2E_MARK=${PYTEST_E2E_MARK:-"not slow"}

declare -A STATUS

run_step() {
  local name="$1"; shift
  local logfile="$OUT_DIR/${name}.log"
  echo "==> [$name] Running: $*" | tee "$logfile"
  local start_ts end_ts
  start_ts=$(date +%s)
  # shellcheck disable=SC2086
  bash -lc "$*" 2>&1 | tee -a "$logfile"
  local ec=${PIPESTATUS[0]}
  end_ts=$(date +%s)
  local dur=$(( end_ts - start_ts ))
  if [[ $ec -eq 0 ]]; then
    echo "==> [$name] PASS (${dur}s)" | tee -a "$logfile"
    STATUS["$name"]=PASS
  else
    echo "==> [$name] FAIL (${dur}s), exit=$ec" | tee -a "$logfile"
    STATUS["$name"]=FAIL
  fi
  return $ec
}

should_run() {
  local name="$1"
  if [[ ${#ONLY_STEPS[@]} -eq 0 ]]; then
    return 0
  fi
  local s
  for s in "${ONLY_STEPS[@]}"; do
    if [[ "$s" == "$name" ]]; then return 0; fi
  done
  return 1
}

overall_ec=0

# Lint
if [[ $DO_LINT -eq 1 ]] && should_run LINT; then
  run_step LINT "black --check --diff zeroproof && ruff check zeroproof && isort --check-only --diff zeroproof && mypy" || overall_ec=1
fi

# Unit tests
if [[ $DO_UNIT -eq 1 ]] && should_run UNIT; then
run_step UNIT "pytest tests/unit -v --tb=short ${PYTEST_ADDOPTS:-}" || overall_ec=1
fi

# Property tests
if [[ $DO_PROPERTY -eq 1 ]] && should_run PROPERTY; then
run_step PROPERTY "pytest tests/property -v --tb=short -m property --hypothesis-profile=ci ${PYTEST_ADDOPTS:-}" || overall_ec=1
fi

# E2E tests
if [[ $DO_E2E -eq 1 ]] && should_run E2E; then
run_step E2E "pytest tests/e2e -v --tb=short -m \"$E2E_MARK\" ${PYTEST_ADDOPTS:-}" || overall_ec=1
fi

# E2E filtered: no_nan (treat 'no tests collected' as success)
if [[ $DO_E2E_NONAN -eq 1 ]] && should_run E2E_NONAN; then
  set +e
bash -lc "pytest tests/e2e -v -k 'no_nan' --tb=short ${PYTEST_ADDOPTS:-}" 2>&1 | tee "$OUT_DIR/E2E_NONAN.log"
  ec=${PIPESTATUS[0]}
  if [[ $ec -eq 5 ]]; then
    echo "==> [E2E_NONAN] No tests matched -k 'no_nan' — skipping." | tee -a "$OUT_DIR/E2E_NONAN.log"
    STATUS["E2E_NONAN"]=SKIP
    ec=0
  fi
  if [[ $ec -eq 0 ]]; then STATUS["E2E_NONAN"]=PASS; else STATUS["E2E_NONAN"]=FAIL; overall_ec=1; fi
  set -e
fi

# Coverage
if [[ $DO_COVERAGE -eq 1 ]] && should_run COVERAGE; then
run_step COVERAGE "pytest --cov=zeroproof --cov-report=term-missing --cov-report=xml:$OUT_DIR/coverage.xml ${PYTEST_ADDOPTS:-}" || overall_ec=1
fi

# Framework-specific tests (optional)
if [[ $DO_TORCH -eq 1 ]] && should_run TORCH; then
  if python - <<'PY'
import sys
try:
    import torch  # type: ignore
    sys.exit(0)
except Exception:
    sys.exit(42)
PY
  then
    run_step TORCH "pytest tests -v --tb=short -k torch ${PYTEST_ADDOPTS:-}" || overall_ec=1
  else
    echo "==> [TORCH] Skipping: torch not importable" | tee "$OUT_DIR/TORCH.log"
    STATUS["TORCH"]=SKIP
  fi
fi

if [[ $DO_JAX -eq 1 ]] && should_run JAX; then
  if python - <<'PY'
import sys
try:
    import jax  # type: ignore
    sys.exit(0)
except Exception:
    sys.exit(42)
PY
  then
    run_step JAX "pytest tests -v --tb=short -k jax ${PYTEST_ADDOPTS:-}" || overall_ec=1
  else
    echo "==> [JAX] Skipping: jax not importable" | tee "$OUT_DIR/JAX.log"
    STATUS["JAX"]=SKIP
  fi
fi

# Summary
echo "\nSummary (logs in $OUT_DIR):"
for key in LINT UNIT PROPERTY E2E E2E_NONAN COVERAGE TORCH JAX; do
  if [[ -n "${STATUS[$key]:-}" ]]; then
    printf "  - %-10s %s\n" "$key" "${STATUS[$key]}"
  fi
done

exit $overall_ec
