// orchestrator_core.c -- classical real-time orchestration core, compiled
// to real wasm32 (clang --target=wasm32-unknown-unknown, no libc, no
// emscripten). This is the "QASM is to WASM" half of the project, taken
// literally rather than as a metaphor: it mirrors the actual production
// split used by Quantinuum's QEC Decoder Toolkit, where WASM hosts a
// sandboxed, low-latency classical control loop that runs continuously
// and co-located with the hardware, while the quantum circuit itself
// (here: qmodule/qasm_module.py's QasmModule, an OpenQASM 2.0 statevector
// simulator) is invoked only on demand, per event.
//
// Concretely: orc_tick() below re-implements snn/lif_neuron.py's step()
// update rule -- the per-timestep LIF neuron integration -- as the thing
// that would, in a real deployment, run every tick regardless of whether
// anything interesting is happening. A future orchestrator/ layer decides,
// from the spike buffer this writes, which (if any) nodes just crossed
// the fusion layer's trigger condition and therefore need their QGNN
// node-update circuit invoked this tick. That decision logic belongs one
// layer up (orchestrator/, Task 6); this file only owns the always-on
// numerical core.
//
// Deliberately re-implemented from scratch in a second language with no
// shared code path with lif_neuron.py, specifically so the two can be
// cross-validated against each other in tests/test_pipeline_smoke.py --
// agreement between them is evidence the *algorithm* is right, not that
// one copied the other's bug. See that test file for the actual numbers.
//
// No libc, no malloc, no libm (the LIF update needs only +,-,*,/, so there
// is no transcendental-function / libm-availability dependency to manage
// in a freestanding build). Neuron state lives in fixed-size static
// arrays, exposed to the host via pointer-returning export functions --
// the host (see run_tick_harness.mjs) writes/reads them directly through
// the module's linear memory rather than passing arrays through function
// arguments, which the wasm32 MVP ABI does not support for anything
// beyond i32/i64/f32/f64 scalars.

#define MAX_NEURONS 65536

static double V[MAX_NEURONS];
static double I[MAX_NEURONS];
static double SPIKED[MAX_NEURONS];

int orc_max_neurons(void) {
    return MAX_NEURONS;
}

double *orc_v_ptr(void) {
    return V;
}

double *orc_i_ptr(void) {
    return I;
}

double *orc_spiked_ptr(void) {
    return SPIKED;
}

// One vectorized-over-neurons Euler step, for the first `n` entries of the
// V/I buffers, in place. Returns the number of neurons that spiked this
// tick (a cheap, useful summary the host can check without re-scanning the
// SPIKED buffer itself).
//
// Update rule -- IDENTICAL in structure and operation order to
// snn/lif_neuron.py's step(), on purpose, so floating point evaluates the
// same associativity in both languages:
//     V_next = V + (dt/tau) * ( -(V - v_rest) + R*I )
//     if V_next >= v_th: spiked=1, V_next = v_reset
//     else:              spiked=0
//
// Returns -1 and touches nothing if n is outside [0, MAX_NEURONS]. The
// original core had no such guard: the independent review showed that
// calling orc_tick(MAX_NEURONS + 2) silently advanced I[0] and I[1] as if
// they were neurons (V[] and I[] are adjacent in linear memory), and only
// a much larger n reached the end of linear memory and trapped. The WASM
// sandbox protects the HOST from the module; it does not protect the
// module's own buffers from each other. The Node harness already refused
// over-capacity n before calling in; this guard makes the core safe on
// its own, for any host.
int orc_tick(int n, double dt, double tau, double v_rest, double v_reset,
             double v_th, double R) {
    if (n < 0 || n > MAX_NEURONS) {
        return -1;
    }
    int spike_count = 0;
    for (int k = 0; k < n; k++) {
        double v = V[k];
        double i_in = I[k];
        double v_next = v + (dt / tau) * (-(v - v_rest) + R * i_in);
        double spiked = 0.0;
        if (v_next >= v_th) {
            spiked = 1.0;
            v_next = v_reset;
            spike_count++;
        }
        V[k] = v_next;
        SPIKED[k] = spiked;
    }
    return spike_count;
}
