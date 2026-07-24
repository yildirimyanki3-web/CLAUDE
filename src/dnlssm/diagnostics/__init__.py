"""Statistical validation, residual diagnostics, and robustness analysis.

``residuals.py``
    Per-variable standardized (innovation) residuals, computed from the
    particle filter's genuinely out-of-sample one-step-ahead predictions,
    plus per-variable RMSE/MAE/bias -- never a single aggregate error
    number.
``normality.py``
    Shapiro-Wilk and Jarque-Bera normality tests on standardized residuals,
    per variable.
``autocorrelation.py``
    ACF/PACF and the Ljung-Box test for residual autocorrelation, per
    variable.
``heteroskedasticity.py``
    The ARCH-LM test for residual conditional heteroskedasticity, per
    variable, with an explicit statement of what a static (scalar/diagonal/
    full) observation-noise covariance can and cannot address.
``robustness.py``
    Particle-count sensitivity (does the fitted result change materially
    with more/fewer particles?) and observation-noise-structure sensitivity
    (does the data support a more complex noise covariance?), both fit
    directly rather than asserted.

Every function here reports numeric findings only; none produces economic
interpretation.
"""

from dnlssm.diagnostics.autocorrelation import test_autocorrelation
from dnlssm.diagnostics.heteroskedasticity import test_heteroskedasticity
from dnlssm.diagnostics.normality import test_normality
from dnlssm.diagnostics.residuals import ResidualDiagnostics, compute_residual_diagnostics
from dnlssm.diagnostics.robustness import (
    analyze_noise_structure_sensitivity,
    analyze_particle_count_sensitivity,
)

__all__ = [
    "test_autocorrelation",
    "test_heteroskedasticity",
    "test_normality",
    "ResidualDiagnostics",
    "compute_residual_diagnostics",
    "analyze_noise_structure_sensitivity",
    "analyze_particle_count_sensitivity",
]
