"""
=============================================================================
TITLE: Target Configuration: TOI-7721.01 (configs/TOI_7721_01.py)
=============================================================================
Description:
    Physical and orbital configuration for candidate TOI-7721.01.

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
    target_name="TOI_7721_01",
    tic_id="163029327",
    toi="7721.01",
    
    stellar=StellarParams(
        teff=5415.0,
        teff_err=126.137,
        m_star=0.94,
        m_star_err=0.112493,
        r_star=0.9493,       # Rounded for code cleanliness
        r_star_err=0.0548,
        logg=4.45636,
        logg_err=0.0817619,
        feh=0.0,             # Assumed solar metallicity (unconstrained in literature)
        feh_err=0.5,         # Broad prior uncertainty reflecting the assumed metallicity
        mag_J=11.614      
    ),
    
    orbit=OrbitalPriors(
        period=2.564226,
        period_err=0.001,                  # Standard BLS detection uncertainty
        epoch_bjd=4046.3536 + 2457000.0,   # Linear conversion: BTJD -> BJD_TDB
        epoch_err=0.005,
        depth_ppm=5537.0,
        duration_hours=1.68
    ),
    
    preprocessing=PreprocessingConfig(
        binning_min=3.0,     # Short binning cadence to preserve the morphology of a 1.68h transit
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