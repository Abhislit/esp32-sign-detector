#!/usr/bin/env bash
# run_parity.sh - compile src/mhi.c natively and verify it matches pc/mhi.py.
#
# This is the cheapest insurance in the project. Preprocessing skew between
# training and inference is invisible at runtime and costs days.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="$ROOT/build/parity"
mkdir -p "$WORK"

echo "== compiling src/mhi.c for the host =="
gcc -std=c11 -O2 -Wall -Wextra -Wconversion \
    -o "$WORK/test_mhi_parity" \
    "$ROOT/tools/test_mhi_parity.c" \
    "$ROOT/src/mhi.c"

echo "== running parity check =="
python3 "$ROOT/pc/parity.py"
