"""
=============================================================================
TITLE: Target Configuration: TOI-2392.03 (configs/TOI_2392_03.py)
=============================================================================
Description:
    Physical and orbital configuration for candidate TOI-2392.03.

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
    target_name="TOI_2392_03",
    tic_id="149349867",
    toi="2392.03",
    
    stellar=StellarParams(
        teff=6244.5,
        teff_err=133.9,
        m_star=1.21,
        m_star_err=0.18,
        r_star=1.76,
        r_star_err=0.09,
        logg=4.03,
        logg_err=0.09,      
        feh=-0.05,
        feh_err=0.12,
        mag_J=9.879      
    ),
    
    orbit=OrbitalPriors(
        period=9.169467,
        period_err=0.001,                  # Standard BLS detection uncertainty
        epoch_bjd=2038.6311 + 2457000.0,   # Linear conversion: BTJD -> BJD_TDB
        epoch_err=0.005,
        depth_ppm=216.0,
        duration_hours=5.16
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