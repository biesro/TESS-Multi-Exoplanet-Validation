"""
=============================================================================
TITLE: Base Configuration and Type Schema (configs/base_config.py)
=============================================================================
Description:
    Defines type contracts, data structures, and default configuration
    schemas for the Bayesian transit modeling and physical characterization
    pipeline.

Author: Biel Escolà-Rodrigo
=============================================================================
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class StellarParams:
    """Host star physical parameters with symmetric Gaussian uncertainties."""
    teff: float          # Effective temperature [K]
    teff_err: float
    m_star: float        # Stellar mass [M_sun]
    m_star_err: float
    r_star: float        # Stellar radius [R_sun]
    r_star_err: float
    logg: float          # Surface gravity [cgs]
    logg_err: float
    feh: float           # Metallicity [Fe/H]
    feh_err: float
    mag_J: float         # 2MASS J-band apparent magnitude (for TSM calculation)


@dataclass
class OrbitalPriors:
    """Initial orbital ephemerides and transit geometric priors."""
    period: float        # Orbital period [days]
    period_err: float
    epoch_bjd: float     # Mid-transit epoch [BJD / BTJD]
    epoch_err: float
    depth_ppm: float     # Estimated transit depth [ppm]
    duration_hours: float# Total transit duration T_14 [hours]


@dataclass
class PreprocessingConfig:
    """Photometric data preprocessing and windowing settings."""
    binning_min: float = 3.0       # Light curve binning interval [minutes]
    use_windows: bool = True       # If True, retains data only around transit windows
    window_factor: float = 1.5     # Window half-width = window_factor * duration


@dataclass
class SamplerConfig:
    """Nested sampling execution parameters for Bayesian model fitting."""
    nlive: int = 1000              # Number of active live points
    dlogz: float = 0.1             # Convergence stopping threshold (evidence tolerance)
    nthreads: int = 4              # CPU threads for parallel sampling
    n_samples_mc: int = 10000      # Monte Carlo samples for physical parameter propagation
    use_cached_fit: bool = True    # If True and cached posteriors exist, reuse sampling run


@dataclass
class PlanetConfig:
    """Master target specification encapsulating system data and model priors."""
    # System identifiers
    target_name: str               # Target identifier (e.g., "TOI_1277_02")
    tic_id: str                    # TESS Input Catalog ID (e.g., "153949511")
    toi: str                       # TESS Object of Interest ID (e.g., "1277.02")

    # Modular parameter blocks
    stellar: StellarParams
    orbit: OrbitalPriors
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    sampler: SamplerConfig = field(default_factory=SamplerConfig)

    # Physical modeling flags
    is_circular: bool = True       # If True, enforces circular orbit (e = 0); otherwise fits (e, omega)
    has_rv: bool = False           # If True, incorporates spectroscopic radial velocity dataset
    rv_file_path: Optional[str] = None # File path to radial velocity data if has_rv=True
    use_gp: bool = False           # If True, includes Gaussian Process for stellar activity
    out_folder: str = "results"    # Base output directory