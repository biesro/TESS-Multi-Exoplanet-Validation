"""
=============================================================================
TITLE: Target Configuration: TOI-1277.02 (configs/TOI_1277_02.py)
=============================================================================
Description:
    Stellar, orbital, and sampling parameters for the candidate TOI-1277.02,
    including ground-based photometric follow-up data configuration.

Author: Biel Escolà-Rodrigo
=============================================================================
"""

from configs.base_config import (
    PlanetConfig, 
    StellarParams, 
    OrbitalPriors, 
    PreprocessingConfig, 
    SamplerConfig
)

config = PlanetConfig(
    target_name="TOI_1277_02_nodata",
    tic_id="153949511",
    toi="1277.02",
    
    stellar=StellarParams(
        teff=5544.0,
        teff_err=120.311,
        m_star=0.976,
        m_star_err=0.122166,
        r_star=0.857648,
        r_star_err=0.0429648,
        logg=4.5609,
        logg_err=0.1,      # Nominal conservative uncertainty
        feh=-0.072,        # Representative mean of spectroscopic estimates (-0.072, -0.09, -0.061)
        feh_err=0.08,
        mag_J=9.14       
    ),
    
    orbit=OrbitalPriors(
        period=14.855896,
        period_err=0.001,                  # Standard BLS detection uncertainty
        epoch_bjd=1684.9758 + 2457000.0,   # Linear conversion: BTJD -> BJD_TDB
        epoch_err=0.005,
        depth_ppm=1287.0,
        duration_hours=3.84
    ),
    
    # Extended transit window to guarantee out-of-transit baseline stability
    preprocessing=PreprocessingConfig(
        binning_min=3.0,
        use_windows=True,
        window_factor=2.0  # Expanded from 1.5 to 2.0 to sample stable baseline for a 3.84h transit
    ),
    
    sampler=SamplerConfig(
        nlive=1000,
        dlogz=0.1,
        nthreads=4,
        use_cached_fit=True
    ),
    
    # Physical modeling options
    is_circular=True,  
    has_rv=False,       # Planetary mass inferred via mr_forecast (Chen & Kipping 2017)
    
    out_folder="results"
)