<p align="center">
  <img src="docs/assets/architecture.svg" width="960" alt="Human-aware confidence communication and evaluation framework">
</p>

<h1 align="center">From Calibrated Confidence to Calibrated Reliance</h1>

<p align="center">
  <strong>Human-Aware Confidence Communication for AI-Assisted Decision Making</strong>
</p>

<p align="center">
  <a href="https://github.com/OliverDOU776/From-Calibrated-Confidence-to-Calibrated-Reliance/actions/workflows/tests.yml"><img src="https://github.com/OliverDOU776/From-Calibrated-Confidence-to-Calibrated-Reliance/actions/workflows/tests.yml/badge.svg" alt="tests"></a>
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python 3.10+">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="MIT License"></a>
  <img src="https://img.shields.io/badge/ACM%20TSC-accepted-1F6F43" alt="Accepted at ACM TSC">
  <img src="https://img.shields.io/badge/reproducibility-code%20%2B%20reference%20outputs-6F42C1" alt="Reproducibility package">
</p>

> **Accepted for publication in ACM Transactions on Social Computing (TSC).**

Most calibration work asks whether a model is correct 80% of the time when it reports 80%
confidence. We ask a different question: **after a person sees that confidence and makes the final
decision, is the human-AI team correct 80% of the time?**

This repository provides the accepted paper's CPU-only offline analysis, the implementation of
**Reliance Calibration Error (RCE)**, the `g0`-`g3` confidence-display policies, deterministic data
setup, compact reference outputs, and numerical regression checks.

## Why calibrated confidence is not enough

A statistically calibrated model can still induce poorly calibrated human reliance. People combine
AI advice with their own beliefs, confidence, and task context; the displayed number is therefore
filtered through human judgment before it affects the final decision.

We formalize **calibrated reliance** as a team-level property. For a display policy `g`, the shown
confidence is calibrated when

$$
\Pr(\text{final human-AI decision is correct} \mid D_g=d)=d.
$$

RCE summarizes deviations from that ideal:

$$
\operatorname{RCE}(g)=\mathbb{E}\left[\left|\eta_g(D_g)-D_g\right|\right].
$$

The distinction matters empirically: in the primary HAIID condition, conventional model ECE is
`0.018`, while direct-display plug-in RCE is `0.114`.

<p align="center">
  <img src="docs/assets/reliance_reliability.png" width="760" alt="Reliance reliability diagram across HAIID tasks">
</p>

## Contributions

- **A new team-level target.** Calibrated reliance evaluates the meaning of confidence after human
  interpretation, not just the model in isolation.
- **A measurable error.** RCE uses the familiar reliability-diagram template with final team
  correctness replacing model correctness.
- **A display-policy framework.** `g0`-`g3` separate direct, global, subgroup-aware, and bounded
  confidence communication without changing the underlying predictor.
- **Large-scale behavioral evidence.** The source HAIID dataset contains 35,670 interactions across
  visual, textual, and tabular tasks; the main policy cohort contains 28,168 records before the
  held-out 70/30 split.
- **A deployment warning.** Reliance varies with user state, while aggressive remapping can create
  semantically misleading boundary values. A population average is not a deployment guarantee.
- **Diagnostics beyond one split.** The package covers overlap, clipping, display shift, RCE
  binning, participant jackknife, reliance-model choice, participant/item-disjoint splits, and
  user-state perturbations.

## The confidence-display policies

| Policy | Definition | Interpretation |
|---|---|---|
| `g0` | `d = c` | Directly display the model confidence. |
| `g1` | `d = clip(2.4c - 0.5, 0, 1)` | Global human-aware diagnostic mapping. It is intentionally treated as unconstrained, not automatically deployment-ready. |
| `g2` | Isotonic mapping fitted within pre-advice self-confidence terciles | Captures user-state heterogeneity. It requires reliable elicitation, governance checks, and a safe fallback. |
| `g3` | `d = clip(g1(c), 0.15, 0.85)` | Bounded global mapping that avoids 0/1 displays. |

`g2` is **not** presented as a fairness guarantee. If self-confidence is missing, noisy, correlated
with protected attributes, or strategically reported, the system should fall back to `g0` or a
bounded population-level policy.

See [docs/METHOD.md](docs/METHOD.md) for equations, estimands, and implementation details.

## Key findings

The accepted paper deliberately distinguishes a held-out display-alignment diagnostic from
model-dependent off-policy simulations.

| Policy | Model-predicted MSE ↓ | Plug-in observed-outcome RCE ↓ | Model-predicted RCE ↓ | Displays at 0 or 1 |
|---|---:|---:|---:|---:|
| `g0` direct | 0.571 | 0.114 | **0.040** | 5.0% |
| `g1` global | **0.518** | 0.104 | 0.162 | 57.8% |
| `g2` subgroup-aware | 0.623 | **0.009** | 0.125 | 14.3% |
| `g3` bounded global | 0.530 | 0.047 | 0.070 | **0.0%** |

> **Metric note.** Plug-in observed-outcome RCE keeps held-out team outcomes fixed and evaluates
> numerical display-outcome alignment. Model-predicted MSE and counterfactual RCE re-simulate
> outcomes under a fitted reliance model. They are off-policy estimates, not randomized causal
> effects.

Three patterns are especially important:

1. `g1` produces the lowest simulated MSE, but 57.8% of its displays lie at 0 or 1. It is best read
   as a diagnostic estimate of the correction required under the fitted model.
2. `g2` produces the strongest plug-in RCE alignment, but its model-predicted MSE and RCE are worse
   than direct display. It exposes heterogeneity; it does not justify immediate personalized
   deployment.
3. `g3` preserves much of the simulated MSE gain while eliminating 0/1 displays, making bounded
   policies the more defensible design direction.

The verified release intentionally centers the final revision diagnostic matrix. It does not mix in
legacy task- or subgroup-effect plots generated under earlier all-condition/simple-model pipelines,
because those use a different analysis slice and response-model specification.

## Reproduce the accepted-paper analysis

### 1. Create the environment

```bash
git clone https://github.com/OliverDOU776/From-Calibrated-Confidence-to-Calibrated-Reliance.git
cd From-Calibrated-Confidence-to-Calibrated-Reliance

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Python 3.10+ is supported. The accepted-paper offline pipeline is CPU-only; it does not require a
GPU, an API key, or a language-model checkpoint.

For the exact dependency versions used in the release verification run (Python 3.12.3):

```bash
python -m pip install -r requirements-lock.txt
python -m pip install -e . --no-deps
```

The lock file requires Python 3.11 or newer. On Python 3.10, use the compatible dependency
resolution from `python -m pip install -e .`; that path is covered by the test matrix.

### 2. Download and verify HAIID

```bash
python scripts/download_data.py
```

The downloader is pinned to upstream commit `24881cc7586180a9c9742a7dd838aea97d008235` and
fails if either SHA-256 checksum differs. Raw data remain Git-ignored. See
[data/README.md](data/README.md) for provenance and licensing.

### 3. Run the core analysis

```bash
python scripts/reproduce.py --profile core
python scripts/verify_results.py --profile core
```

Generated files are written to `results/generated/`. Reference outputs are never overwritten.

### 4. Run the full diagnostic matrix

```bash
python scripts/reproduce.py --profile full
python scripts/verify_results.py --profile full
```

The full profile additionally fits the alternative reliance-model specifications and runs random,
participant-disjoint, and item-disjoint split sensitivity analyses.

Equivalent shortcuts are available:

```bash
make data
make reproduce       # core
make verify
make reproduce-full  # full diagnostic matrix
make test            # data-free unit tests
```

## What is reproduced

| Scope | Public command | Status |
|---|---|---|
| HAIID preprocessing and accepted 70/30 task-stratified split | `scripts/reproduce.py` | Reproduced from pinned public data |
| `g0`-`g3`, model-predicted MSE/RCE, plug-in RCE | `scripts/reproduce.py` | Reproduced |
| Support, clipping, and display-shift diagnostics | `scripts/reproduce.py` | Reproduced |
| Binning, jackknife, user-state, subgroup, and bounded-policy checks | `--profile core` | Reproduced |
| Reliance-model and disjoint-split sensitivity | `--profile full` | Reproduced |
| Published GRACE comparison values | `data/external/grace_verbalized_results.csv` | Auditable context only; not a reproduction of the GRACE raw pipeline |
| 20-participant prospective pilot | — | Not claimed: participant-level data are not in the research workspace |

For the complete output-to-claim map and numerical tolerance policy, see
[docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## Repository structure

```text
.
├── src/calibrated_reliance/   # Reusable data, metric, policy, and reliance-model code
├── scripts/
│   ├── download_data.py       # Pinned, checksum-verified HAIID setup
│   ├── reproduce.py           # Accepted-paper offline analysis
│   └── verify_results.py      # Generated-vs-reference numerical checks
├── data/
│   ├── external/              # Small cited aggregate table for GRACE
│   └── raw/                   # Downloaded data; ignored by Git
├── results/
│   ├── reference/             # Compact accepted-paper reference outputs
│   └── generated/             # Local reproduction outputs; ignored by Git
├── docs/                      # Method and reproducibility documentation
├── tests/                     # Data-free unit tests
├── CITATION.cff
├── pyproject.toml
└── Makefile
```

## Evidence boundaries and responsible use

This repository is designed to make the paper inspectable, including its uncomfortable results.

- The HAIID study is observational. Self-confidence moderation is consistent with anchoring, but it
  does not identify anchoring as the unique causal mechanism.
- Off-policy estimates depend on conditional response modeling, overlap, and the absence of major
  unobserved confounding.
- A confidence display can improve simulated decisions while becoming less semantically honest
  about the model's epistemic state. Do not deploy an undisclosed remapping as if it were raw model
  confidence.
- Subgroup-conditioned policies require measurement-quality, privacy, disparate-impact, and gaming
  audits.
- The small prospective pilot in the paper is feasibility evidence, not confirmatory validation.
- Nothing here establishes safety or effectiveness in high-stakes clinical, legal, financial, or
  public-sector deployment.

<p align="center">
  <img src="docs/assets/display_shift.png" width="780" alt="Raw versus displayed confidence distributions under g0 through g3">
</p>

## Citation

The DOI and final volume/issue metadata were not available when this repository was prepared. Until
they are assigned, please cite the accepted paper as:

```bibtex
@article{wang2026calibratedreliance,
  title   = {From Calibrated Confidence to Calibrated Reliance:
             Human-Aware Confidence Communication for AI-Assisted Decision Making},
  author  = {Wang, Zijia and Hu, Kejia},
  journal = {ACM Transactions on Social Computing},
  year    = {2026},
  note    = {Accepted for publication}
}
```

GitHub can also read [CITATION.cff](CITATION.cff) directly.

## Data and third-party acknowledgements

HAIID remains governed by its upstream MIT License and citation requirements. GRACE raw files are
not redistributed. See [data/README.md](data/README.md) and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for exact versions, checksums, sources, and reuse
boundaries.

## License

The original code in this repository is released under the [MIT License](LICENSE). Third-party
datasets and aggregate results remain governed by their own licenses and citation requirements;
the project license does not override those terms.

## Questions and contributions

Reproducibility reports are welcome through GitHub Issues. When reporting a discrepancy, include
your Python version, dependency versions, command, and the relevant generated/reference filenames.
