# Course integration acceptance — 2026-11-13

Adapter distribution: **0.3.0**. Core source baseline:
`41c28bc192478c70903e48c5d39d4577cd83bd78` (Core package 0.7.1, interface 1.1).
This integration does not modify Core source or the existing MLE-bench command behavior.

## Verified scope

| Check | Result |
| --- | --- |
| Windows complete Adapter tests | 98 passed, 4 skipped |
| Assignment 1 Linux complete Adapter tests | 101 passed, 1 skipped |
| Core 1.1 persistent offline smoke on both platforms | single, ensemble, partial, novalid, summaryconflict passed |
| Portable synthetic solution replay | Each member script and Core-generated final/combine.py reproduced saved prediction hashes |
| Allowed model artifact links | Linux materializes the documented member model link; copied files remain after original model removal |
| Out-of-bound artifact links | Linux rejects them |
| Built wheel | 24 Python files match source; isolated installation loads version 0.3.0 and course CLI |
| Released course input conversion | Both tasks match original rows, columns, labels and order; test target excluded from Core test CSV |
| Released task preflight | Both ready=true with explicit absolute Core source path; no run directory created |

All 102 tests were collected on both platforms. The shared skip is the existing integration test
requiring `MLEBENCH_PREPARED_DIR`; it was not set. The other three Windows skips require Linux-style
unprivileged symbolic links and passed on Linux. The reported tests include all migrated legacy
tests, existing repository Adapter tests, and the new 1.1 integration/regression tests.

The actual Core functions `bookkeeping.write_best`, `write_status`, `write_search_tree`,
`final.build_final` and `report.build_summary` were exercised with handcrafted nodes and synthetic
arrays. No gateway or search controller was run. Network access is denied by the 1.1 fixture.
The Report Agent used deterministic output or a mock writer; local HTTP provider tests did not
contact an external model. Synthetic label comparisons only test the evaluator and report lifecycle.

**Real model API calls: 0. Model training runs: 0. Official task scoring runs: 0.**
These checks do not prove real Core search quality, gateway access, GPU readiness, final benchmark
performance, or course-wide compliance.

## Corrections established by failing regressions

1. An incomplete voting format could pass collection but make `combine.py` fail. The collector now
   requires `format.normalize` to be exactly false for course class-label ensembles. Missing, true
   and non-boolean values are rejected.
2. Core votes original CSV strings, including distinct spellings such as `0` and `0.0`. The collector
   now uses the same literal-string accumulation and member-order tie rule, then validates the winning
   numeric class. The regression invokes the actual Core-generated combination script for comparison.
3. Summary checks now treat explicit null final metrics/scores as meaningful anchors and compare
   summary final members with the independently collected member identities. Unsupported claims
   cannot become report conclusions simply because best-node metadata is correct.

The first two failures were independently reproduced, fixed, and independently rechecked. Existing
input sealing, private-label separation, partial outputs, report fallback, stale-report detection,
source snapshots and old 0.1 behaviors remain covered.

## Input preparation

| Task | train | val | test |
| --- | ---: | ---: | ---: |
| headlines_ctr_clean | 8301 | 940 | 3232 |
| in_vehicle_coupon_recommendation | 8117 | 2030 | 2537 |

No original input was changed. Prepared 1.1 data and private labels are local deployment artifacts,
not repository contents. Validation remains the provisional Core default; test labels use separate
same-account paths, not operating-system-enforced isolation. Full-script delivery remains the default.

## Evidence and environment

`course-verification.json` records compact results, artifact hashes and the source baseline. Detailed
logs and smoke reports are retained under the local/Assignment 1 `evidence/` directories and ignored
by Git. The installable wheel is a build artifact, not checked-in source.

Windows: Python 3.13.5; Linux: Python 3.13.11. Core-writer test dependencies include NumPy 2.1.3,
pandas 2.2.3, scikit-learn 1.6.1 and pytest 8.3.4. The Linux `.venv` was created for this checkout
with access to the existing base packages; it is not an isolated full production environment lock.
Full Core training dependencies were not installed as part of this work.

Delegation followed `adaptive-task-routing`: Terra handled runtime migration and report adaptation;
Sol created offline integration fixtures and independently reviewed the collector. Parent integration
included both-platform checks, regression fixes and package verification. Worker usage totals were
not exposed and no cost claim is made.

## Remaining acceptance

Real Core search, configured model gateways, approved training/inference, official held-out scoring,
final PDF report and the team’s final validation/deployment policy are separate pending stages.
Static preflight success is not a substitute for those checks. Any new real-run report must preserve
the distinction between validation selection, held-out evaluation and unverified replay capability.
