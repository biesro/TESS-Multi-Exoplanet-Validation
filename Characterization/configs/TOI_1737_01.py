"""
=============================================================================
TITLE: Target Configuration: TOI-1737.01 (configs/TOI_1737_01.py)
=============================================================================
Description:
    Physical and orbital configuration for candidate TOI-1737.01 orbiting a
    rapidly rotating F-type dwarf star.

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
    target_name="TOI_1737_01",
    tic_id="233009109",
    toi="1737.01",
    
    stellar=StellarParams(
        teff=6669.0,
        teff_err=127.5,
        m_star=1.408,
        m_star_err=0.237152,
        r_star=1.95,
        r_star_err=0.08,
        logg=4.01,
        logg_err=0.1,      # Nominal conservative uncertainty of 0.1 cgs
        feh=0.081,
        feh_err=0.08,
        mag_J=8.133
    ),
    
    orbit=OrbitalPriors(
        # Parameters derived directly from multi-sector BLS search (624,959 cadences)
        period=14.447995,
        period_err=0.001,                  # Constrained prior width for nested sampler
        epoch_bjd=1717.0140 + 2457000.0,   # Linear conversion: BTJD -> BJD_TDB
        epoch_err=0.005,
        depth_ppm=245.0,                   # Shallow transit signal (0.245 ppt)
        duration_hours=5.76
    ),
    
    # Preprocessing: Long transit duration (5.76h); maintain window_factor at 2.0 
    # to guarantee robust out-of-transit baseline continuum coverage
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
    
    # Key physical modeling choices for TOI-1737.01
    is_circular=True,  
    has_rv=False,       # Rapidly rotating F-dwarf (v*sin(i) ~ 16 km/s); RV semi-amplitude unconstrained; mass inferred via mr_forecast
    
    out_folder="results"
)

# Note: Ground-based photometry is omitted (ground_photometry left undefined).
# The shallow transit depth (245 ppm) is beyond ground-based photometric precision limits.
# Archival LCO-McD observations serve exclusively for nearby eclipsing binary (NEB) spatial vetting, not light curve fitting.