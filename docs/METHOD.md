# Method reference

This document describes the implementation that corresponds to the accepted manuscript. It also
separates quantities that are easy to conflate: model calibration, observed-outcome display
alignment, and model-predicted off-policy evaluation.

## 1. Setup

For each interaction:

- `Y` is the ground-truth label;
- `Y_hat` is the AI recommendation;
- `C in [0, 1]` is the confidence that the recommendation is correct;
- `H0` is the human's pre-advice judgment;
- `U` contains observable pre-display user/context features;
- `D_g = g(C, U)` is the confidence shown under policy `g`;
- `Y_team^(g)` is the final human-AI decision under that display.

Changing `g` leaves the underlying predictor and recommendation unchanged. It changes only the
confidence interface.

## 2. Model calibration and calibrated reliance

Conventional model calibration asks whether

```text
P(Y_hat = Y | C = c) = c.
```

Calibrated reliance instead asks whether

```text
P(Y_team^(g) = Y | D_g = d) = d.
```

This is a semantic property of the final team confidence, not a requirement that a person always
accept the AI advice.

With confidence bins `I_b`, empirical Reliance Calibration Error is

```text
RCE(g) = sum_b (n_b / N) * |team_accuracy(I_b) - mean_display(I_b)|.
```

The implementation uses the actual mean displayed value in each nonempty bin. A value of exactly
`1.0` is assigned to the final equal-width bin rather than discarded.

## 3. Accepted-paper policies

### g0: direct display

```python
d = c
```

This is the identity baseline.

### g1: global human-aware mapping

```python
d = clip(2.4 * c - 0.5, 0.0, 1.0)
```

The accepted analysis uses `alpha=2.4` and `beta=-0.5`, estimated on the training partition. The
mapping is deliberately unconstrained and places many values at the boundaries. It should be
interpreted as a diagnostic policy, not as a recommendation to present undisclosed 0/1 values to
users.

### g2: subgroup-aware isotonic mapping

1. Convert the pre-advice response to the normalized self-confidence signal used by the paper.
2. Divide the training observations into low, mid, and high self-confidence terciles.
3. Within each tercile, fit isotonic regression from model confidence to observed final team
   correctness.
4. At evaluation time, apply the mapping for the elicited pre-advice self-confidence group.
5. Use the global isotonic mapping as a software fallback for an unseen label; in a real interface,
   missing or unreliable elicitation should trigger `g0` or a bounded population policy.

Isotonic regression preserves monotonicity but does not by itself establish fairness, causal
personalization, or deployment safety.

### g3: robustness guard

The reported default guards `g1`:

```python
d = clip(g1(c), 0.15, 0.85)
```

The package also exposes bounded-shift variants that constrain `|d - c| <= delta` and restrict the
display range to `[0.05, 0.95]`.

## 4. HAIID preprocessing

The public HAIID responses are encoded on `[-1, 1]`, with positive values aligned to the correct
label. The analysis derives:

```text
advice_prob       = (advice + 1) / 2
prob_correct_1    = (response_1 + 1) / 2
correct_pre       = 1[response_1 > 0]
correct_post      = 1[response_2 > 0]
ai_correct        = 1[advice > 0]
shifted_toward    = 1[(response_2-response_1)(advice-response_1) > 0]
abs_gap_pre_advice = |advice - response_1|
```

The main analysis keeps Art, Sarcasm, Cities, and Census trials with
`perceived_accuracy=80`. It then performs a 70/30 task-stratified split with seed 42, producing
19,717 training rows and 8,451 validation rows.

## 5. Reliance model and counterfactual response

The baseline reliance model is logistic regression. Its predictors are:

- displayed confidence;
- normalized pre-advice self-confidence;
- AI-source indicator;
- absolute pre-advice/advice gap;
- displayed-confidence by self-confidence interaction;
- one-hot task and advice-source indicators.

Numeric features are standardized on the training partition. The model estimates the probability
of moving toward the advice, `pi(d, x, u)`. The counterfactual response is

```text
r2^(g) = r1 + pi(g(c, u), x, u) * (advice - r1).
```

The model-predicted task loss is

```text
MSE(g) = mean((r2^(g) - 1)^2),
```

because the correct endpoint is encoded as `+1`.

## 6. Two different RCE estimands

The release keeps these names explicit.

### Plug-in observed-outcome RCE

This holds `correct_post` fixed on the held-out logged outcomes and changes only the displayed
value. It asks whether a proposed display is numerically aligned with the outcomes that were
actually observed.

It is a **display-alignment diagnostic**, not the causal value of deploying the new display.

### Model-predicted counterfactual RCE

This uses the fitted reliance model to obtain a counterfactual continuous team score under the new
display, then compares that score with the display.

It is closer to off-policy evaluation but inherits reliance-model assumptions and becomes
especially sensitive under aggressive display shifts.

## 7. Diagnostic assumptions

Interpret model-predicted quantities only with the accompanying checks:

- empirical support and nearest-neighbor distance;
- boundary mass and mean absolute display shift;
- alternative reliance-model specifications;
- participant- and item-disjoint splits;
- equal-width/equal-mass RCE bins;
- participant jackknife;
- subgroup-definition and self-report perturbations;
- bounded-policy honesty/performance frontier.

These checks constrain the claim. They do not turn observational off-policy simulation into a
randomized causal experiment.
