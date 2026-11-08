Experimental research software by **Aaron Michael Weis**  
ORCID: https://orcid.org/0009-0002-1906-148X

This release adds a persistent Node-hosted WebAssembly LIF runtime to the QASM-WASM reference implementation. One worker and one WASM instance remain alive across ticks; the original one-shot engine stays available for comparison.

### Included
- Persistent runtime with bounded JSON messages, input validation, timeouts and explicit cleanup.
- A runnable graph-feedback demo and reproducible host benchmark.
- The unchanged Revision 4 paper and original experiment records, with a separate M2 supplement.
- MIT-licensed code; the paper and its figures remain CC BY 4.0.

### Verification
All 11 local test modules passed: **1,536 PASS entries**, zero failures or skips. The 250 new M2 entries cover numerical agreement, feedback, input isolation, capacity and lifecycle/error behavior. Assertion counts are not independent experiments. The supplied WASM binary was used locally without rebuilding.

On this Windows machine, persistent calls were about **185x faster at 25 neurons** and **40x faster at 1,024 neurons** than launching Node per tick. Python/NumPy was still faster than both. These are LIF-only roundtrip measurements, with persistent startup reported separately; they are not quantum or energy advantages.

### Scope
This is an interface analogy and integration-testing study, not a QASM-to-WASM compiler. No training, real-QPU result, quantum advantage or production-scale deployment is claimed. Known frontend/seed/sampler limitations remain documented. The historical mutation ledger predates M2.

See README.md, REVIEW_STATUS.md and docs/M2_RUNTIME.md for commands, measured results and limitations. AI assistance is disclosed in the manuscript.
