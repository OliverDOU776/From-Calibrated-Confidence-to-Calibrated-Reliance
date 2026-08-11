# Reproducibility guide

## Scope

The public package reproduces the accepted manuscript's **offline HAIID analysis** from a pinned,
checksum-verified upstream dataset. It also ships a six-row, cited GRACE aggregate table so the
bounded literature comparison can be audited without presenting it as a raw-pipeline reproduction.

It does not claim to reproduce the prospective pilot because participant-level pilot data and its
trial-level analysis code are not present in the research workspace.

## Clean-room sequence

The exact release lock was verified on Python 3.12.3 and requires Python 3.11 or newer:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m pip install -e . --no-deps
python scripts/download_data.py
python scripts/reproduce.py --profile full
python scripts/verify_results.py --profile full
python -m unittest discover -s tests -v
```

Python 3.10 is supported through the broader constraints in `pyproject.toml`; on Python 3.10, use
`python -m pip install -e .` instead of the exact lock file.

All commands should be run from the repository root.

## Expected dataset audit

| Quantity | Expected |
|---|---:|
| Complete HAIID rows | 35,670 |
| Complete HAIID participants | 1,125 |
| Four main tasks | 34,783 rows |
| Main `perceived_accuracy=80` slice | 28,168 rows |
| Training partition | 19,717 rows |
| Validation partition | 8,451 rows |
| Main-slice participants | 881 |
| Main-slice task instances | 128 |

The downloader verifies the raw-file checksums before the analysis starts. The data loader then
validates required columns, ranges, task names, and nonmissing critical fields. Missing or malformed
inputs produce a nonzero exit rather than a synthetic fallback.

## Profiles

### Core

`python scripts/reproduce.py --profile core` generates the central policy and interpretation
package:

- `tab_revision_main_policy_eval.csv`
- `tab_revision_support_coverage.csv`
- `tab_revision_clipping_shift.csv`
- `tab_revision_rce_binning_sensitivity.csv`
- `tab_revision_jackknife_rce.csv`
- `tab_revision_self_confidence_noise.csv`
- `tab_revision_subgroup_definition_sensitivity.csv`
- `tab_revision_private_information_decomposition.csv`
- `tab_revision_bounded_frontier.csv`
- `tab_revision_simple_interfaces.csv`
- `tab_revision_pilot_power.csv` (planning calculation from reported aggregates only)
- `fig_revision_display_shift.png`

### Full

`python scripts/reproduce.py --profile full` adds:

- `tab_revision_reliance_model_sensitivity.csv`
- `tab_revision_split_sensitivity.csv`

These checks fit high-dimensional intercept and nonlinear model specifications and therefore take
longer than the core profile.

## Output-to-claim map

| Claim or boundary | Primary generated evidence |
|---|---|
| Direct confidence is not team-calibrated | RCE binning and jackknife tables |
| `g1` improves model-predicted MSE but creates extreme displays | Main policy, clipping, and display-shift outputs |
| `g2` gives strongest plug-in alignment but is governance-limited | RCE binning, main policy, subgroup-noise, and subgroup-definition outputs |
| `g3` avoids 0/1 displays while retaining simulated MSE gains | Main policy, clipping, and bounded-frontier outputs |
| Findings depend on the response model and split assumptions | Reliance-model and split-sensitivity outputs |
| Simpler interfaces do not reproduce every learned-policy trade-off | Simple-interface output |
| Pilot should be treated as underpowered feasibility evidence | Pilot power-planning output plus explicit non-reproduction note |

## Verification policy

`scripts/verify_results.py` compares generated CSVs with `results/reference/tables/`.

- Column names, row order, shapes, and string values must match.
- Numeric values use `rtol=1e-9` and `atol=1e-10` by default to permit harmless last-bit differences
  across BLAS/scikit-learn versions.
- Missing files, nonfinite unexpected values, schema changes, or values outside tolerance fail the
  command.
- The table verifier does not compare PNGs. The display-shift plot emitted by
  `scripts/reproduce.py` is an explanatory artifact; numerical regression is enforced through the
  corresponding CSV outputs rather than raster byte identity.

Reference files are read-only inputs to verification. Reproduction writes exclusively to
`results/generated/`.

## Determinism

- Analysis seed: `42`.
- Main split: task-stratified 70/30.
- Isotonic subgroup policy: fitted only on the training partition.
- Participant jackknife: deterministic leave-one-participant-out evaluation.
- Stochastic estimators receive explicit random states.

The release verification environment is recorded in `requirements-lock.txt`. The broader ranges in
`pyproject.toml` support reuse, while numerical regression should use the lock file.

## Data and privacy boundary

HAIID is a public third-party human-participant dataset containing hashed participant identifiers
and demographic fields. This repository downloads it from the source rather than creating a second
copy. Do not attempt re-identification or join the identifiers to external records.

GRACE raw data are not redistributed because the audited upstream snapshot did not expose an
explicit license. Only aggregate values already published in the ACL paper are included.
