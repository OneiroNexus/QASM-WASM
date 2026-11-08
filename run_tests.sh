#!/usr/bin/env bash
# Builds wasm-core and runs every test module in this project, in the same
# dependency order the layers were built and validated in (see AGENTS.md /
# VALIDATION.md). Exits nonzero if any test module fails.
set -uo pipefail
cd "$(dirname "$0")"

echo "== building wasm-core =="
bash wasm-core/build.sh || { echo "wasm-core build failed"; exit 1; }

cd python
FAILED=0
for t in test_qasm_sim test_qasm_gate_coverage test_qasm_parser_robustness test_lif_neuron test_gnn_message_passing test_shared_space test_pipeline_smoke test_composition_wiring test_wasm_core_bounds test_properties test_persistent_wasm; do
  echo ""
  echo "== python3 -m tests.$t =="
  python3 -m "tests.$t"
  if [ $? -ne 0 ]; then
    echo "!! tests.$t FAILED"
    FAILED=1
  fi
done

echo ""
if [ "$FAILED" -eq 0 ]; then
  echo "ALL TEST MODULES PASSED"
else
  echo "SOME TEST MODULES FAILED"
fi
exit $FAILED
