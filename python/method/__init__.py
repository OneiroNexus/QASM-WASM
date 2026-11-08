"""
method/ -- the validation method, made into code.

The scaffold's layers are validated against independent ground truth
(VALIDATION.md); this package validates the VALIDATION: it measures what
the test suite can actually see, and keeps the evidence indexed.

  method/mutate.py       operator-based mutation testing over every source
                         module (and the C core), with test selection by
                         import graph and one fresh copy of the tree per mutant
  method/reference_sim.py an independent Kronecker-product statevector
                         simulator, shared by the tests that need a reference
  method/props.py        a seeded, dependency-free property-testing harness
  method/claims.py       the registry: every validated property, which test
                         checks it, by which mode, since which version
  method/build_index.py  generates BUILD_INDEX.md from the registry, the last
                         suite run and the last mutation campaign
  method/triage.json     the human-written verdict on every surviving mutant

Run everything with `bash run_method.sh` from the repository root.
"""
