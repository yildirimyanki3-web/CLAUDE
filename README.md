# DNLSSM Research Platform

A reproducible academic research platform for estimating **Dynamic
Nonlinear State Space Models (DNLSSM)** on macroeconomic observation
panels via Maximum Likelihood Estimation and a from-scratch **Bootstrap
Particle Filter / Particle Smoother**.

This is research software, not an application. Its purpose is to turn a
17-variable macroeconomic observation panel into a statistically validated,
numerically stable, fully reproducible estimate of a low-dimensional latent
dynamical system — nothing more. The platform is **strictly theory-neutral**:

- Latent states are always labelled neutrally (`State 1`, `State 2`, ...
  `State k`). No code path names, infers, or asserts an economic regime,
  historical period, or policy narrative.
- Every generated report presents numbers and their direct statistical
  meaning only ("did not converge," "ARCH effects detected in variable
  X," "AIC favors latent_dim=5"). Interpretation is left entirely to the
  researcher working from these numbers afterward.
- Synthetic data is used only inside the test suite, and never presented
  as, or silently substituted for, a real research result. If real data
  cannot be resolved, the pipeline stops and states exactly which series
  are missing and how to supply them.

## Contents

- [Architecture](#architecture)
- [Installation](#installation)
- [Configuration](#configuration)
- [The observation space](#the-observation-space)
- [Running the pipeline](#running-the-pipeline)
- [What one run produces](#what-one-run-produces)
- [The model](#the-model)
- [The Bootstrap Particle Filter and Smoother](#the-bootstrap-particle-filter-and-smoother)
- [Parameter estimation](#parameter-estimation)
- [Model selection](#model-selection)
- [Diagnostics and robustness](#diagnostics-and-robustness)
- [Docker](#docker)
- [Testing](#testing)
- [Extending the platform](#extending-the-platform)
- [Reproducibility](#reproducibility)
- [Repository layout](#repository-layout)

## Architecture

The package is organized so that each module has exactly one
responsibility, and every hyperparameter flows from one place — there are
no hardcoded magic numbers scattered through the business logic.

```
config/           Central configuration: experiment_config.yaml (all
                   hyperparameters) + variables.yaml (the 17-variable
                   observation-space registry). Validated by pydantic
                   schemas in src/dnlssm/config/schema.py.

src/dnlssm/
  config/          Pydantic schema + YAML loader (single source of truth).
  data/            Provider connectors (FRED, World Bank, IMF IFS, OECD
                   SDMX, BIS, TCMB EVDS, manual CSV/Excel upload) with a
                   uniform interface, orchestrated by DataManager with
                   full per-variable provenance logging.
  preprocessing/   Frequency alignment, missing-data reporting/imputation,
                   transforms (log/diff/log_diff/pct_change), seasonal
                   adjustment (STL), standardization, observation-matrix
                   assembly with explicit shape validation.
  models/          AbstractStateSpaceModel interface, modular nonlinear
                   function families (neural/polynomial/generalized-
                   logistic/linear), numerically stable noise-covariance
                   parameterization, the concrete DNLSSM, and the neutral
                   LatentManifold/LatentTrajectory data structures.
  filters/         Bootstrap Particle Filter and Particle Smoother
                   (backward-simulation + fixed-lag), built from scratch:
                   resampling schemes, ESS/entropy diagnostics, one-step-
                   ahead forecasting.
  optimization/    Multi-start Maximum Likelihood Estimation: the common-
                   random-numbers objective, a uniform optimizer wrapper
                   with per-iteration logging, automated non-convergence
                   diagnosis, Hessian-based confidence intervals.
  model_selection/ AIC/BIC/HQIC + genuine out-of-sample RMSE/MAE across a
                   swept latent dimension, selected by weighted rank
                   aggregation across all criteria.
  diagnostics/     Per-variable residual statistics, normality
                   (Shapiro-Wilk/Jarque-Bera), autocorrelation (ACF/PACF/
                   Ljung-Box), heteroskedasticity (ARCH-LM), and
                   robustness analysis (particle-count and noise-structure
                   sensitivity, both genuinely re-fit).
  visualization/   Publication-quality (300 DPI) matplotlib figures.
  experiments/     Unique experiment IDs, reproducible directory layout,
                   metadata.
  reports/         Numeric-only Markdown report generation, CSV/LaTeX
                   table export.
  cli.py           `dnlssm-run run` — the single command that runs
                   everything above, in order.

tests/
  unit/            One test module per source module (~270 tests).
  integration/     Full pipeline smoke tests (offline, synthetic data).
  fixtures/        A from-scratch Kalman filter + RTS smoother reference,
                   used to validate the particle filter/smoother against
                   the exact closed-form answer on a linear-Gaussian
                   special case of the DNLSSM.
```

### Designed for a multi-layer research program

The current release runs exactly one latent system (the domestic DNLSSM).
The data model does not assume that will always be true:

- `LatentManifold` identifies a latent coordinate system by an opaque id
  and a dimension, not by any economic label — a second manifold (e.g. a
  future international/global latent system) is just another instance,
  not an architectural change.
- Observation space and latent dimension are both plain integers derived
  from config, never assumed fixed anywhere in the codebase — adding an
  18th observation variable or changing the latent dimension requires no
  code change.
- `AbstractStateSpaceModel` is a narrow interface (transition/observation
  moments, noise covariances, initial-state moments, parameter
  count/init). The filtering, optimization, model-selection, and
  diagnostics stack are written entirely against this interface, so a
  future Dynamic Factor Model, Hidden Markov Model, or Switching Linear
  State Space Model becomes usable by the same infrastructure by
  implementing this one class.
- Every module cleanly separates the computation layer from
  interpretation: nothing here produces or requires an economic
  narrative. Fiber-bundle morphisms between manifolds, a state
  optimization theory layer, and a praxeological interpretation layer are
  future additions that operate on this platform's outputs — not
  something this codebase needs to be rewritten to accommodate.

## Installation

Requires Python 3.11 or 3.12.

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .                 # installs the `dnlssm-run` command
```

For development (tests, linter, type checker):

```bash
pip install -r requirements-dev.txt
```

Copy `.env.example` to `.env` and fill in any API keys you have available
(TCMB EVDS, FRED). World Bank, IMF IFS, OECD, and BIS do not require a key.
Series with no configured or reachable provider fall back to a manual
CSV/Excel upload — see `data/manual_uploads/README.md`.

```bash
cp .env.example .env
# edit .env
```

## Configuration

**Every hyperparameter of a run lives in exactly two files:**

- `config/experiment_config.yaml` — runtime, preprocessing, model
  architecture, optimization, particle filter, model selection, and
  diagnostics settings.
- `config/variables.yaml` — the observation-space variable registry: the
  17 series described below, their transform/standardization/seasonal-
  adjustment policy, and their provider priority list.

Nothing in `src/dnlssm/**` hardcodes a hyperparameter; changing an
experiment means editing these two files (or passing `--config` /
`--variables` to point at alternate copies), never touching source code.

## The observation space

Seventeen variables, aligned onto a common monthly grid regardless of
native frequency:

| # | Variable | Native frequency | Transform |
|---|----------|-------------------|-----------|
| 1 | Policy interest rate | monthly | none |
| 2 | Credit volume / broad money supply | monthly | log |
| 3 | CPI index (→ inflation via log-diff) | monthly | log_diff |
| 4 | USD/TRY exchange rate | daily → monthly avg | log |
| 5 | Current account balance | monthly | none |
| 6 | Central government budget balance | monthly | none |
| 7 | Public debt / GDP | quarterly → monthly | none |
| 8 | Private sector debt / GDP | quarterly → monthly | none |
| 9 | Industrial production index | monthly | log |
| 10 | Unemployment rate | monthly | none |
| 11 | Exports | monthly | log |
| 12 | Imports | monthly | log |
| 13 | Tax revenue / GDP | annual → monthly | none |
| 14 | Real wage index | monthly | log |
| 15 | Manufacturing capacity utilization rate | monthly | none |
| 16 | Economic Complexity Index (ECI) | *placeholder* | — |
| 17 | Structural Orientation (Specialization) Index | *placeholder* | — |

Series 16-17 have no configured provider yet; they are carried through the
full pipeline as declared, all-`NaN` columns — the observation matrix
always has all 17 columns regardless of data availability, so adding a
real source for either later requires editing `variables.yaml` only, never
the model or filtering code.

**On provider series codes:** connector code (HTTP requests, response
parsing, retry/error handling) is fully implemented for every provider.
The exact TCMB EVDS series identifiers, however, are deliberately left
`null` in the shipped `variables.yaml` — EVDS codes are periodically
renumbered, and shipping a guessed code that looks authoritative would be
worse than an explicit, auditable gap. Fill them in from the EVDS
interactive query system before a production run, or supply the series via
manual upload. The pipeline reports, itemized, exactly which series this
affects on every run.

## Running the pipeline

```bash
dnlssm-run run --config config/experiment_config.yaml --variables config/variables.yaml
```

or, without installing the package:

```bash
python scripts/run_pipeline.py run --config config/experiment_config.yaml --variables config/variables.yaml
```

This single command:

1. downloads every resolvable observation series and reports provenance;
2. aligns, imputes, transforms, and standardizes them into an observation
   matrix with fully validated dimensions;
3. sweeps latent dimensions 3-8 (configurable), fitting a DNLSSM at each
   by multi-start MLE;
4. runs the Bootstrap Particle Filter and Particle Smoother on the
   selected model;
5. computes residual, normality, autocorrelation, and heteroskedasticity
   diagnostics;
6. runs particle-count and noise-structure robustness analyses;
7. generates every figure and a full numeric-only Markdown report;
8. writes reproducibility metadata (seed, software version, hardware,
   exact resolved config).

**By default the pipeline refuses to proceed past data acquisition** if
any non-placeholder variable could not be resolved from any provider or
manual upload — it prints exactly which variables and how to fix each.
Pass `--allow-partial-data` to proceed anyway (the report still states
which columns are missing). Useful flags:

```
--allow-partial-data      Proceed even with unresolved observation variables.
--skip-model-selection    Fit only model.latent_dim instead of sweeping.
--skip-robustness         Skip the particle-count / noise-structure sensitivity stage.
--output-root PATH        Override runtime.output_root.
--manual-upload-dir PATH  Directory scanned for manual CSV/Excel fallback series.
--log-level LEVEL         Override runtime.log_level (DEBUG for full optimizer traces).
```

## What one run produces

Each run gets a unique, timestamped experiment directory under
`experiments_output/<experiment_name>_<timestamp>_<id>/`:

```
config.yaml              Exact resolved configuration for this run.
metadata.json            Software version, seed, hardware, runtime, summary metrics.
run.log                  Full log, including every optimizer iteration.
results/                 Raw JSON numeric outputs.
figures/                 Latent trajectories, ESS curve, observation fit,
                          per-variable residual histograms/QQ-plots and
                          ACF/PACF, optimization convergence traces,
                          AIC/BIC/HQIC comparison, particle-count
                          sensitivity — all at 300 DPI.
reports/
  report.md              The full numeric-only experiment report.
  tables/                Every result table as both CSV and LaTeX
                          (booktabs-style, ready for a manuscript).
```

Re-running with the same `config.yaml` and the same `global_seed`
reproduces the same numerical results (see [Reproducibility](#reproducibility)).

## The model

```
x_t = f(x_{t-1}; theta_f) + w_t,   w_t ~ N(0, Q(theta_Q))
y_t = g(x_t;     theta_g) + v_t,   v_t ~ N(0, R(theta_R))
x_0 ~ N(mu_0(theta_0), Sigma_0(theta_0))
```

`f` and `g` are members of a configurable nonlinear function family
(`neural` by default — a genuine single-hidden-layer MLP, so the model
does not degenerate to a linear-Gaussian Kalman-filter model unless
`family: linear` is explicitly selected; `polynomial` and
`generalized_logistic` are also available). Both are dimension-agnostic:
neither the latent dimension nor the observation dimension is assumed
fixed anywhere.

Noise covariances are parameterized to be symmetric positive definite **by
construction**, never patched after the fact: variances go through
`softplus` (always positive), and the `full` structure uses a Cholesky
factor (`Q = L L^T`, PSD by construction, positive diagonal via
`softplus`). A configurable diagonal jitter (`regularization_epsilon`) and
an adaptive-escalation `safe_cholesky` guard every factorization used in
filtering against near-singular matrices.

## The Bootstrap Particle Filter and Smoother

Implemented from scratch — no external particle-filtering library is used.

- **Filter**: initialization, prediction, importance weighting with
  correct missing-dimension masking, log-likelihood evaluation via the
  general (non-uniform-prior) sequential-importance-weight recursion,
  Effective-Sample-Size computation, adaptive ESS-triggered resampling
  (systematic/stratified/multinomial/residual), filtered posterior
  estimation. Default particle count: 4000 (`particle_filter.n_particles`,
  freely configurable).
- **Smoother**: FFBSi backward-simulation (Godsill, Doucet & West, 2004),
  vectorized via the Gumbel-max trick, plus a cheaper fixed-lag smoother
  using the filter's resampling genealogy. Both produce a *smoothed*
  trajectory distinct from the filtered one.
- **One-step-ahead forecasting**: at every time step the filter also
  records `E[y_t | y_1:t-1]` and its covariance — computed from the
  pre-update particle weights, so it never conditions on `y_t` itself.
  This is what model-selection out-of-sample scoring and innovation-
  residual diagnostics use; using the posterior (`y_1:t`-informed)
  estimate instead would leak information and overstate fit quality.

**Correctness validation**: on a linear-Gaussian special case of the
DNLSSM (`family: linear` for both maps), the filter's filtered mean and
log-likelihood, and the smoother's smoothed mean, are checked against a
from-scratch Kalman filter / RTS smoother reference implementation
(`tests/fixtures/linear_gaussian_reference.py`) — the strongest
correctness check available for a Monte Carlo filtering algorithm.

## Parameter estimation

Maximum Likelihood via `scipy.optimize`, with:

- **Multi-start**: every configured method (`L-BFGS-B`, `BFGS`,
  `Nelder-Mead`, ...) is run from `optimization.n_multistarts`
  independent random initializations; the best converged run is selected,
  and a cross-method comparison table (convergence rate, best/mean
  log-likelihood, wall time) reports which optimizer is more reliable for
  the given model.
- **Common random numbers**: the particle filter's random seed is fixed
  once per optimization attempt and reused across every evaluation within
  it, so the Monte Carlo objective is a deterministic function of the
  parameters for that attempt — standard practice for optimizing a
  particle-filter likelihood, and essential for a stable convergence
  trace.
- **Full iteration logging**: every accepted iteration records
  log-likelihood, parameter norm, parameter step size, and (for
  gradient-based methods) the finite-difference gradient norm.
- **Automated non-convergence diagnosis**: a `converged = False` result is
  never reported as a bare fact. Rule-based checks identify iteration-
  budget exhaustion, a high rate of numerically failed evaluations, a
  stalled gradient norm, or divergent parameter magnitude, each paired
  with an actionable recommendation.
- **Confidence intervals**: a numerical Hessian at the MLE (via
  `numdifftools`) gives asymptotic standard errors and confidence
  intervals. A Hessian that is numerically flat is detected explicitly and
  reported as *unavailable* rather than silently regularized into a
  finite-but-meaningless interval.

## Model selection

`model_selection.latent_dim_min .. latent_dim_max` (3-8 by default) are
each fit independently via MLE on a training split
(`model_selection.train_fraction`); AIC, BIC, and HQIC are computed
together (never a single criterion in isolation), and out-of-sample RMSE/
MAE (overall and per observation variable) are scored on the held-out tail
using the leak-free one-step-ahead forecast. The final dimension is chosen
by weighted rank aggregation across all criteria
(`model_selection.criteria_weights`) — not by any single metric.

## Diagnostics and robustness

- **Per-variable residual statistics** (RMSE, MAE, bias, std) from
  one-step-ahead innovation residuals, standardized by the predicted
  marginal standard deviation.
- **Normality**: Shapiro-Wilk and Jarque-Bera, per variable.
- **Autocorrelation**: ACF/PACF and the Ljung-Box test, per variable.
- **Heteroskedasticity**: the ARCH-LM test, per variable — reported with
  an explicit statement that no static (scalar/diagonal/full) noise
  covariance structure resolves time-varying variance if found; that
  would require a stochastic-volatility model family, a stated candidate
  for future work, not a claimed fix.
- **Particle-count sensitivity**: the fitted model is re-run through the
  filter at each of `particle_filter.robustness_particle_counts`
  (default `[2000, 3000, 4000, 5000]`), comparing filtered-mean and
  log-likelihood stability.
- **Noise-structure sensitivity**: independent MLE refits under each of
  `diagnostics.observation_noise_alternatives`, compared by AIC/BIC —
  answering whether the data actually supports a more complex covariance,
  rather than assuming one structure a priori.

## Docker

```bash
docker compose build
docker compose run dnlssm run --config config/experiment_config.yaml
```

The image pins the Python version and every dependency, so results are
reproducible across Linux, macOS, and Windows (via Docker Desktop/WSL2)
identically.

## Testing

```bash
pytest                          # full suite: ~270 unit tests + integration tests, ~80s
pytest -m "not integration"     # unit tests only, faster
pytest --cov=dnlssm             # with coverage (requires pytest-cov)
ruff check src/ tests/          # lint
mypy src/dnlssm                 # type check
```

The most important tests validate the from-scratch particle filter and
smoother against exact closed-form references (a Kalman filter and an RTS
smoother) on a linear-Gaussian special case of the model — not just that
the code runs, but that it computes the mathematically correct answer.
`tests/integration/test_pipeline_smoke.py` runs the entire CLI pipeline
end to end against synthetic, offline data, exercising the cross-module
wiring that per-module unit tests cannot.

## Extending the platform

- **New data provider**: implement `DataConnector`
  (`src/dnlssm/data/base.py`) and register it in
  `dnlssm.data.manager.build_default_connectors`.
- **New nonlinear function family**: implement `NonlinearFunction`
  (`src/dnlssm/models/functions.py`) and add one branch to
  `build_nonlinear_function`.
- **New model family** (Dynamic Factor Model, HMM, Switching LSS, ...):
  implement `AbstractStateSpaceModel`
  (`src/dnlssm/models/base.py`) — the filtering, optimization,
  model-selection, and diagnostics stack work with it unchanged.
- **New observation variable**: add an entry to `config/variables.yaml`;
  no source code changes are required anywhere in the pipeline.

## Reproducibility

A single `runtime.global_seed` drives everything. Every stochastic
component (multi-start initial points, particle initialization/resampling,
robustness-analysis draws, ...) receives an independent
`numpy.random.Generator` spawned deterministically from that one seed via
`dnlssm.utils.random_state.SeedSequence`, keyed by a human-readable label
plus contextual tags — so results are reproducible bit-for-bit given the
same seed and config, regardless of call order, and independent components
never accidentally correlate through a shared global RNG state.

Every experiment directory contains the exact resolved `config.yaml` used
to produce it; re-running the CLI against that file with the same seed
reproduces the same numerical results.

## Repository layout

```
config/                 experiment_config.yaml, variables.yaml
data/
  manual_uploads/        Researcher-supplied CSV/Excel fallback series.
src/dnlssm/              The package (see Architecture above).
scripts/run_pipeline.py  Convenience entry point (no install required).
tests/
  unit/                  ~270 unit tests, one module per source module.
  integration/            Full-pipeline smoke tests.
  fixtures/               Kalman filter / RTS smoother reference implementation.
experiments_output/      Default output root (one directory per run).
Dockerfile, docker-compose.yml
pyproject.toml, requirements.txt, requirements-dev.txt
```
