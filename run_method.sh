#!/usr/bin/env bash
# The validation method, end to end:
#   1. build wasm-core and run every test module, recording per-module counts   (seconds)
#   2. run the operator-based mutation campaign over every source module        (~5 minutes on 2 cores)
#   3. regenerate BUILD_INDEX.md from the claim registry + both results files
# Exits nonzero if the suite fails or if any surviving mutant lacks a verdict in
# python/method/triage.json -- an untriaged survivor is an open question, and
# the index must not be regenerated with open questions hidden in it.
set -uo pipefail
cd "$(dirname "$0")"

echo "== 1/3 building wasm-core and running the suite =="
bash wasm-core/build.sh || { echo "wasm-core build failed"; exit 1; }
( cd python && python3 -m method.run_suite ) || { echo "SUITE FAILED"; exit 1; }

echo ""
echo "== 2/3 mutation campaign =="
( cd python && python3 -m method.mutate --workers "${MUTATE_WORKERS:-2}" ) || { echo "mutation campaign failed"; exit 1; }

echo ""
echo "== 3/3 build index =="
( cd python && python3 -m method.build_index --require-triage ) || { echo "BUILD INDEX: untriaged survivors (see method/triage.json)"; exit 1; }
echo "OK: BUILD_INDEX.md is current"
