"""
=============================================================================
TITLE: Target Configuration: TOI-4450.01 (configs/TOI_4450_01.py)
=============================================================================
Description:
    Physical and orbital configuration for candidate TOI-4450.01.

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
    target_name="TOI_4450_01",
    tic_id="373017346",
    toi="4450.01",
    
    stellar=StellarParams(
        teff=5844.0,
        teff_err=134.577,
        m_star=1.054,
        m_star_err=0.128222,
        r_star=1.15,
        r_star_err=0.06,
        logg=4.34318,
        logg_err=0.0769439,
        feh=0.26,        # Mean of spectroscopic measurements: (0.283 + 0.236) / 2
        feh_err=0.08,
        mag_J=9.963      
    ),
    
    orbit=OrbitalPriors(
        period=10.692429,
        period_err=0.001,                  # Standard BLS detection uncertainty
        epoch_bjd=2396.3315 + 2457000.0,   # Linear conversion: BTJD -> BJD_TDB
        epoch_err=0.005,
        depth_ppm=1224.0,
        duration_hours=3.12
    ),
    
    preprocessing=PreprocessingConfig(
        binning_min=3.0,  
        use_windows=True,
        window_factor=2.0
    ),
    
    sampler=SamplerConfig(
        nlive=1000,
        dlogz=0.1,
        nthreads=4,
        use_cached_fit=True
    ),
    
    is_circular=True,  
    has_rv=False,       
    
    out_folder="results"
)