"""
=============================================================================
TITLE: Derived Physics and Monte Carlo Error Propagation (core/physics_models.py)
=============================================================================
Description:
    Derived physics engine and Monte Carlo error propagation module for the
    exofit-framework. Cross-matches Dynesty posterior distributions with
    Gaussian host star parameter priors to derive planetary physical dimensions,
    bulk densities, atmospheric metrics, and taxonomic classifications.

Author: Biel Escolà-Rodrigo
=============================================================================
"""

import os
from pathlib import Path
from typing import Dict, Tuple, Any

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import gaussian_kde
from astropy import constants as const
import astropy.units as u

from configs.base_config import PlanetConfig
from core import mr_forecast as mr


# ============================================================================
# CLASSIFICATION AND ATMOSPHERIC METRICS
# ============================================================================

def calc_tsm(
    Rp_Rearth: float, 
    Mp_Mearth: float, 
    R_star_Rsol: float, 
    Teq: float, 
    mag_J: float
) -> float:
    """
    Computes the Transmission Spectroscopy Metric (TSM).
    Reference: Kempton et al. (2018), PASP 130, 114401.
    """
    if Rp_Rearth < 1.5:
        scale = 0.190
    elif Rp_Rearth < 2.75:
        scale = 1.260
    elif Rp_Rearth < 4.0:
        scale = 1.280
    else:
        scale = 1.150
        
    tsm = scale * (Rp_Rearth**3 * Teq) / (Mp_Mearth * R_star_Rsol**2)
    return float(tsm * 10.0**(-mag_J / 5.0))


def classify_planet(Rp_Rjup: float, Teq: float, Mp_Mjup: float = None) -> Dict[str, str]:
    """
    Classifies the companion across radius, thermal regime, and mass domain
    with high taxonomic resolution.
    """
    result = {}
    
    # Radius conversion to Earth radii for terrestrial/volatile regimes
    Rp_Rearth = Rp_Rjup * 11.2089

    # ====================================================================
    # 1. PHYSICAL SIZE (Reflecting the Fulton radius valley & Kepler demographics)
    # ====================================================================
    if Rp_Rearth < 0.8:
        result['size'] = 'Sub-Earth'
        result['size_emoji'] = '🌑'
    elif Rp_Rearth < 1.25:
        result['size'] = 'Earth-sized'
        result['size_emoji'] = '🌍'
    elif Rp_Rearth < 2.0:
        # Below the Fulton radius gap (predominantly rocky composition)
        result['size'] = 'Super-Earth'
        result['size_emoji'] = '🪨'
    elif Rp_Rearth < 4.0:
        # Above the Fulton radius gap (volatile-rich H/He envelopes or water-worlds)
        result['size'] = 'Sub-Neptune / Mini-Neptune'
        result['size_emoji'] = '💧'
    elif Rp_Rearth < 7.0:
        result['size'] = 'Neptune-sized'
        result['size_emoji'] = '🧊'
    elif Rp_Rearth < 10.5:
        result['size'] = 'Sub-Saturn'
        result['size_emoji'] = '🪐'
    elif Rp_Rjup < 1.3:
        result['size'] = 'Jovian'
        result['size_emoji'] = '♃'
    else:
        # Highly irradiated or inflated gas giant
        result['size'] = 'Inflated Super-Jovian'
        result['size_emoji'] = '🎈'

    # ====================================================================
    # 2. EQUILIBRIUM TEMPERATURE (Independent of mass/radius boundaries)
    # Note: Earth has Teq ~255 K assuming zero albedo and greenhouse baseline.
    # ====================================================================
    if Teq > 2500:
        result['temp'] = 'Extreme Ultra-Hot'
        result['temp_emoji'] = '🌋'
    elif Teq > 1500:
        result['temp'] = 'Ultra-Hot'
        result['temp_emoji'] = '🔥'
    elif Teq > 1000:
        result['temp'] = 'Hot'
        result['temp_emoji'] = '🌡️'
    elif Teq > 500:
        result['temp'] = 'Warm'
        result['temp_emoji'] = '☀️'
    elif Teq > 350:
        result['temp'] = 'Mild'
        result['temp_emoji'] = '🌤️'
    elif Teq > 200:
        result['temp'] = 'Temperate (Habitable Zone Teq)'
        result['temp_emoji'] = '🌿'
    else:
        result['temp'] = 'Cold / Frozen'
        result['temp_emoji'] = '❄️'

    # ====================================================================
    # 3. COMPANION MASS (Dynamical regimes from planetesimals to VLMS)
    # ====================================================================
    if Mp_Mjup is not None:
        Mp_Mearth = Mp_Mjup * 317.8
        
        if Mp_Mearth < 0.5:
            result['mass'] = 'Sub-Earth mass'
        elif Mp_Mearth < 2.0:
            result['mass'] = 'Earth mass'
        elif Mp_Mearth < 10.0:
            result['mass'] = 'Super-Earth mass'
        elif Mp_Mearth < 30.0:
            result['mass'] = 'Neptune mass'
        elif Mp_Mearth < 100.0:
            result['mass'] = 'Sub-Saturn mass'
        elif Mp_Mjup < 0.8:
            result['mass'] = 'Saturn mass'
        elif Mp_Mjup < 2.5:
            result['mass'] = 'Jovian mass'
        elif Mp_Mjup < 13.0:
            # Deuterium burning boundary (~13 M_Jup per IAU definition)
            result['mass'] = 'Super-Jovian mass'
        elif Mp_Mjup < 80.0:
            # Sub-stellar regime between deuterium and hydrogen burning limits
            result['mass'] = 'Brown Dwarf regime'
        else:
            # Hydrogen burning threshold (>80 M_Jup: Red Dwarf regime)
            result['mass'] = 'Very Low-Mass Star (VLMS)'

    # Combine thermal regime and physical size (e.g., "Temperate Super-Earth")
    result['overall'] = f"{result['temp']} {result['size']}"
    return result


# ============================================================================
# POSTERIOR STATISTICAL HELPER FUNCTIONS
# ============================================================================

def _map(array: np.ndarray, name: str, unit: str, decimals: int = 3) -> float:
    """Computes the Maximum A Posteriori (MAP) via Gaussian KDE estimation."""
    clean = array[np.isfinite(array)]
    kde = gaussian_kde(clean)
    x = np.linspace(np.percentile(clean, 1), np.percentile(clean, 99), 1000)
    map_val = float(x[np.argmax(kde(x))])
    fmt = f"{{:.{decimals}f}}"
    print(f"  {name:32s} = {fmt.format(map_val)} {unit}  [MAP]")
    return map_val


def _pct(array: np.ndarray, name: str, unit: str, decimals: int = 3) -> Tuple[float, float, float]:
    """Computes the median and asymmetric 1-sigma percentiles (16th, 84th)."""
    p16, p50, p84 = np.nanpercentile(array, [16, 50, 84])
    ep, em = p84 - p50, p50 - p16
    fmt = f"{{:.{decimals}f}}"
    print(f"  {name:32s} = {fmt.format(p50)} +{fmt.format(ep)} -{fmt.format(em)} {unit}")
    return float(p50), float(ep), float(em)


# ============================================================================
# MONTE CARLO ERROR PROPAGATION
# ============================================================================

def propagate_uncertainties(results: Any, config: PlanetConfig) -> Dict[str, np.ndarray]:
    """
    Propagates physical parameter uncertainties by combining Gaussian sampling 
    of host star properties with correlated nested sampling posterior draws.
    """
    print("\n" + "=" * 70)
    print("MONTE CARLO ERROR PROPAGATION")
    print("=" * 70)
    n = config.sampler.n_samples_mc
    print(f"\n  Drawing {n} correlated Monte Carlo samples...")

    # 1. Host star parameter Gaussian draws
    M_star_s = np.random.normal(config.stellar.m_star, config.stellar.m_star_err, n)
    R_star_s = np.random.normal(config.stellar.r_star, config.stellar.r_star_err, n)
    Teff_s   = np.random.normal(config.stellar.teff,   config.stellar.teff_err,   n)

    # 2. Resampling from Dynesty nested sampling posterior distributions
    post = results.posteriors['posterior_samples']
    total_samples = len(post['p_p1'])
    idx = np.random.randint(0, total_samples, n)

    P_s   = post['P_p1'][idx]
    T0_s  = post['t0_p1'][idx] + 2457000.0  # Linear conversion: BTJD -> BJD_TDB
    p_s   = post['p_p1'][idx]
    b_s   = post['b_p1'][idx]
    rho_s = np.clip(post['rho'][idx], 10.0, None)

    # 3. Orbital geometry and scaled semi-major axis
    G_SI   = const.G.value
    P_sec  = P_s * 86400.0
    a_rs_s = (G_SI * rho_s * P_sec**2 / (3.0 * np.pi))**(1.0 / 3.0)
    a_au   = a_rs_s * R_star_s * 0.00465047  # Solar radii to Astronomical Units

    # Orbital eccentricity
    if not config.is_circular and 'ecc_p1' in post:
        ecc_s = post['ecc_p1'][idx]
    else:
        ecc_s = np.zeros(n)

    # 4. Planetary radius derivation (IAU 2015 nominal standards)
    R_sun_to_Rjup = (const.R_sun / const.R_jup).decompose().value
    Rp_Rjup   = p_s * R_star_s * R_sun_to_Rjup
    Rp_Rearth = Rp_Rjup * (const.R_jup / const.R_earth).decompose().value

    # 5. Equilibrium temperature and incident stellar insolation
    R_sun_to_au = (const.R_sun / const.au).decompose().value
    Teq = Teff_s * np.sqrt(R_star_s * R_sun_to_au / (2.0 * a_au)) * (0.7)**0.25

    sigma_cgs = const.sigma_sb.cgs.value
    R_sun_cm  = const.R_sun.to(u.cm).value
    au_cm     = const.au.to(u.cm).value
    F_rel     = sigma_cgs * Teff_s**4 * (R_star_s * R_sun_cm / (a_au * au_cm))**2 / 1.36e6

    # 6. Mass determination: Radial velocity semi-amplitude vs. empirical forecaster
    if config.has_rv and any(k.startswith('K_') for k in post.keys()):
        k_key = [k for k in post.keys() if k.startswith('K_')][0]
        K_s = post[k_key][idx]  # Semi-amplitude in m/s
        print(f"  → Solving dynamical companion mass from spectroscopic RVs ({k_key})...")
        
        # Orbital inclination (bounded against grazing geometry singularities)
        cos_i = np.clip(b_s / a_rs_s, 0.0, 1.0)
        sin_i = np.clip(np.sqrt(1.0 - cos_i**2), 1e-5, 1.0)
        
        M_star_kg = M_star_s * const.M_sun.value
        
        # Iterative solution of the binary mass function (accurate for brown dwarf companions)
        Mp_kg = np.zeros(n)
        term1 = (K_s * np.sqrt(1.0 - ecc_s**2) / sin_i) * (P_sec / (2.0 * np.pi * G_SI))**(1.0 / 3.0)
        
        # Initial guess assuming test particle limit (Mp << M_star)
        Mp_guess = term1 * (M_star_kg)**(2.0 / 3.0)
        
        # 5 fixed-point iterations ensure numerical convergence across all mass ratios
        for _ in range(5):
            Mp_guess = term1 * (M_star_kg + Mp_guess)**(2.0 / 3.0)
            
        Mp_kg = Mp_guess
        Mp_Mjup = Mp_kg / const.M_jup.value
        Mp_Mearth = Mp_kg / const.M_earth.value
    else:
        print("  → Inferring probabilistic mass via mr_forecast (Chen & Kipping 2017)...")
        Rp_safe = np.clip(Rp_Rearth, 0.11, 99.0)
        Mp_Mearth = np.array(mr.Rpost2M(Rp_safe, unit='Earth'), dtype=float)
        Mp_Mearth = np.where(Mp_Mearth > 0, Mp_Mearth, np.nan)
        Mp_Mjup = Mp_Mearth / (const.M_jup / const.M_earth).decompose().value

    # 7. Derived bulk physical parameters
    print("  → Computing bulk density, surface gravity, and atmospheric scale height...")
    M_jup_kg = const.M_jup.value
    R_jup_m  = const.R_jup.value
    k_B      = const.k_B.value
    m_H      = const.m_p.value

    rho_p  = (Mp_Mjup * M_jup_kg) / ((4.0 / 3.0) * np.pi * (Rp_Rjup * R_jup_m)**3)
    g_p    = G_SI * (Mp_Mjup * M_jup_kg) / (Rp_Rjup * R_jup_m)**2
    logg_p = np.log10(g_p * 100.0)  # Convert to cgs units
    v_esc  = np.sqrt(2.0 * G_SI * (Mp_Mjup * M_jup_kg) / (Rp_Rjup * R_jup_m)) / 1000.0  # km/s
    
    # Atmospheric mean molecular weight (mu) approximation based on radius thresholds
    # < 1.5 R_earth: Rocky (mu = 29); 1.5 - 2.75 R_earth: Water/Sub-Neptune (mu = 18); > 2.75: H2/He (mu = 2.3)
    mu_array = np.where(Rp_Rearth < 1.5, 29.0, 
                        np.where(Rp_Rearth < 2.75, 18.0, 2.3))
    
    H_atm = (k_B * Teq) / (mu_array * m_H * g_p) / 1000.0  # Atmospheric scale height [km]

    print("  ✓ Uncertainty propagation completed successfully")
    return {
        'Rp_Rjup':       Rp_Rjup,   'Rp_Rearth':   Rp_Rearth,
        'a_au':          a_au,      'Teq':         Teq,
        'Mp_Mjup':       Mp_Mjup,   'Mp_Mearth':   Mp_Mearth,
        'rho_planet':    rho_p,     'g_planet':    g_p,
        'logg_planet':   logg_p,    'v_esc':       v_esc,
        'H_atm':         H_atm,     'F_incident':  F_rel,
        'b_impact':      b_s,       'P_orbital':   P_s,
        'T0_bjd':        T0_s,      'eccentricity': ecc_s,
    }


# ============================================================================
# CONSOLE REPORTING AND SUMMARY EXPORT
# ============================================================================

def print_results(mc: Dict[str, np.ndarray], config: PlanetConfig) -> Tuple[Dict, float, float, float, float]:
    """Prints a structured physical characterization summary to the console."""
    print("\n" + "=" * 70)
    print(f"FULL PHYSICAL CHARACTERIZATION RESULTS — {config.target_name}")
    print("=" * 70)

    print("\n[1] STELLAR HOST PARAMETERS:")
    print(f"  TIC {config.tic_id} / TOI {config.toi}")
    print(f"  Teff   = {config.stellar.teff:.0f} ± {config.stellar.teff_err:.0f} K")
    print(f"  M*     = {config.stellar.m_star:.3f} ± {config.stellar.m_star_err:.3f} M_sun")
    print(f"  R*     = {config.stellar.r_star:.4f} ± {config.stellar.r_star_err:.4f} R_sun")
    print(f"  log g  = {config.stellar.logg:.2f} ± {config.stellar.logg_err:.2f} (cgs)")
    print(f"  [Fe/H] = {config.stellar.feh:.2f} ± {config.stellar.feh_err:.2f}")

    print("\n[2] ORBITAL PARAMETERS:")
    _pct(mc['P_orbital'], "P", "days", decimals=7)
    _pct(mc['T0_bjd'],    "T0", "BJD_TDB", decimals=6)
    if config.is_circular:
        print("  e                                = 0.0 (fixed circular)")
    else:
        _pct(mc['eccentricity'], "e", "", decimals=4)
    _pct(mc['a_au'],     "a (median)", "AU", 4)
    _map(mc['a_au'],     "a (MAP)",    "AU", 4)
    _pct(mc['b_impact'], "b",          "",   3)

    print("\n[3] PLANETARY PHYSICAL PROPERTIES:")
    Rp_med, Rp_ep, Rp_em = _pct(mc['Rp_Rjup'],   "Rp (median)", "R_Jup",   3)
    Rp_map               = _map(mc['Rp_Rjup'],   "Rp (MAP)",    "R_Jup",   3)
    Re_med, *_           = _pct(mc['Rp_Rearth'], "Rp (median)", "R_Earth", 2)
    _map(mc['Rp_Rearth'], "Rp (MAP)", "R_Earth", 2)
    
    label_mass = "Mp (Dynamical RV)" if config.has_rv else "Mp (Forecaster Prior)"
    Mp_med, Mp_ep, Mp_em = _pct(mc['Mp_Mjup'],   f"{label_mass} (median)", "M_Jup",   3)
    Mp_map               = _map(mc['Mp_Mjup'],   f"{label_mass} (MAP)",    "M_Jup",   3)
    Me_med, *_           = _pct(mc['Mp_Mearth'], f"{label_mass} (median)", "M_Earth", 1)
    _map(mc['Mp_Mearth'], f"{label_mass} (MAP)", "M_Earth", 1)

    rho_med, *_ = _pct(mc['rho_planet'], "rho_p (median)", "kg/m³", 1)
    _map(mc['rho_planet'], "rho_p (MAP)", "kg/m³", 1)

    print("\n[4] SURFACE GRAVITY AND ESCAPE VELOCITY:")
    g_med, *_    = _pct(mc['g_planet'],    "g_p",    "m/s²", 1)
    _pct(mc['logg_planet'], "log g_p", "(cgs)", 2)
    _pct(mc['v_esc'],       "v_esc",   "km/s",  1)

    print("\n[5] EQUILIBRIUM TEMPERATURE AND INSOLATION:")
    Teq_med, *_ = _pct(mc['Teq'],        "T_eq (median)", "K", 0)
    Teq_map     = _map(mc['Teq'],        "T_eq (MAP)",    "K", 0)
    _pct(mc['F_incident'], "F_p", "F_Earth", 0)

    print("\n[6] ATMOSPHERE AND OBSERVABILITY:")
    _pct(mc['H_atm'], "H (median)", "km", 0)
    _map(mc['H_atm'], "H (MAP)",    "km", 0)

    tsm = calc_tsm(Re_med, Me_med, config.stellar.r_star, Teq_med, config.stellar.mag_J)
    tsm_1mjup = calc_tsm(Re_med, 318.0, config.stellar.r_star, Teq_med, config.stellar.mag_J)
    print(f"\n  TSM (Derived Companion Mass)       ≈ {tsm:.1f}  (mJ = {config.stellar.mag_J})")
    print(f"  TSM (Nominal 1.0 M_Jup Reference)  ≈ {tsm_1mjup:.1f}")

    print("\n[7] TAXONOMIC CLASSIFICATION:")
    cls = classify_planet(Rp_med, Teq_med, Mp_med)
    print(f"  {cls['size_emoji']}  Size Category  : {cls['size']} ({Rp_med:.2f} R_Jup)")
    print(f"  {cls['temp_emoji']}  Thermal Regime : {cls['temp']}")
    if 'mass' in cls:
        print(f"  ⚖️   Mass Regime    : {cls['mass']} ({Mp_med:.2f} M_Jup)")

    return cls, Rp_med, Mp_med, Teq_med, rho_med


def save_summary(mc: Dict[str, np.ndarray], config: PlanetConfig, cls: Dict, out_dir: Path):
    """Writes a comprehensive numerical synthesis report to a formatted text file."""
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / f"{config.target_name}_summary.txt"

    def wp(f, array, name, unit, dec=4):
        p16, p50, p84 = np.nanpercentile(array, [16, 50, 84])
        f.write(f"  {name:30s}: {p50:.{dec}f} +{p84-p50:.{dec}f} -{p50-p16:.{dec}f} {unit}\n")

    def wmap(f, array, name, unit, dec=4):
        clean = array[np.isfinite(array)]
        kde = gaussian_kde(clean)
        x = np.linspace(np.percentile(clean, 1), np.percentile(clean, 99), 1000)
        map_val = float(x[np.argmax(kde(x))])
        f.write(f"  {name:30s}: {map_val:.{dec}f} {unit}  [MAP]\n")

    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write(f"FULL CHARACTERIZATION REPORT — {config.target_name}\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"TARGET         : TIC {config.tic_id} / TOI {config.toi}\n")
        f.write(f"Classification : {cls['overall']}\n\n")

        f.write("STELLAR HOST PARAMETERS:\n")
        f.write(f"  Teff   = {config.stellar.teff} ± {config.stellar.teff_err} K\n")
        f.write(f"  M*     = {config.stellar.m_star} ± {config.stellar.m_star_err} M_sun\n")
        f.write(f"  R*     = {config.stellar.r_star} ± {config.stellar.r_star_err} R_sun\n")
        f.write(f"  log g  = {config.stellar.logg} ± {config.stellar.logg_err} (cgs)\n")
        f.write(f"  [Fe/H] = {config.stellar.feh} ± {config.stellar.feh_err}\n")
        f.write(f"  mag_J  = {config.stellar.mag_J} (2MASS)\n")

        f.write("\nORBITAL PARAMETERS:\n")
        wp(f,  mc['P_orbital'], "P", "days", dec=7)
        wp(f,  mc['T0_bjd'],    "T0", "BJD_TDB", dec=6)
        if config.is_circular:
            f.write("  e                             : 0.0 (fixed circular)\n")
        else:
            wp(f, mc['eccentricity'], "e", "", dec=4)
        wp(f,   mc['a_au'],     "a (median)", "AU")
        wmap(f, mc['a_au'],     "a (MAP)",    "AU")
        wp(f,   mc['b_impact'], "b",          "")

        f.write("\nPLANETARY PHYSICAL PROPERTIES:\n")
        wp(f,   mc['Rp_Rjup'],   "Rp (median)", "R_Jup")
        wmap(f, mc['Rp_Rjup'],   "Rp (MAP)",    "R_Jup")
        wp(f,   mc['Rp_Rearth'], "Rp (median)", "R_Earth", dec=2)
        wp(f,   mc['Mp_Mjup'],   "Mp (median)", "M_Jup")
        wmap(f, mc['Mp_Mjup'],   "Mp (MAP)",    "M_Jup")
        wp(f,   mc['Mp_Mearth'], "Mp (median)", "M_Earth", dec=1)
        wp(f,   mc['rho_planet'],"rho_p (median)", "kg/m³", dec=1)
        wmap(f, mc['rho_planet'],"rho_p (MAP)",    "kg/m³", dec=1)

        f.write("\nTEMPERATURE AND ATMOSPHERE:\n")
        wp(f,   mc['Teq'],        "T_eq (median)", "K", dec=0)
        wmap(f, mc['Teq'],        "T_eq (MAP)",    "K", dec=0)
        wp(f,   mc['H_atm'],      "H (median)",    "km", dec=0)
        wp(f,   mc['F_incident'], "F_p",            "F_Earth", dec=0)

        Re = float(np.nanmedian(mc['Rp_Rearth']))
        Me = float(np.nanmedian(mc['Mp_Mearth']))
        Tq = float(np.nanmedian(mc['Teq']))
        tsm = calc_tsm(Re, Me, config.stellar.r_star, Tq, config.stellar.mag_J)
        f.write("\nATMOSPHERIC OBSERVABILITY:\n")
        f.write(f"  TSM = {tsm:.1f}\n")

    print(f"  → Numerical report saved to: {summary_path}")


# ============================================================================
# PLOTTING AND DIAGNOSTIC VISUALIZATIONS
# ============================================================================

def plot_posterior_distributions(mc: Dict[str, np.ndarray], config: PlanetConfig, out_dir: Path):
    """Generates diagnostic histograms of derived physical posterior distributions."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle(f'{config.target_name} — Derived Posterior Distributions', fontsize=14, fontweight='bold')

    params = [
        (mc['Rp_Rjup'],    r'$R_p$ [$R_{Jup}$]',  'steelblue'),
        (mc['Mp_Mjup'],    r'$M_p$ [$M_{Jup}$]',  'darkorange'),
        (mc['rho_planet'], r'$\rho_p$ [kg/m³]',    'seagreen'),
        (mc['Teq'],        r'$T_{eq}$ [K]',         'firebrick'),
        (mc['a_au'],       r'$a$ [AU]',             'purple'),
        (mc['H_atm'],      r'$H$ [km]',             'brown'),
    ]

    for ax, (data, label, color) in zip(axes.flat, params):
        clean = data[np.isfinite(data)]
        p16, p50, p84 = np.nanpercentile(clean, [16, 50, 84])
        ax.hist(clean, bins=60, color=color, alpha=0.75, edgecolor='none')
        ax.axvline(p50, color='black', linewidth=1.5, label='median')
        ax.axvline(p16, color='black', linewidth=1.0, linestyle='--')
        ax.axvline(p84, color='black', linewidth=1.0, linestyle='--')
        ax.set_xlabel(label, fontsize=11)
        ax.set_ylabel('N', fontsize=10)
        ax.set_title(f'{p50:.3g} $^{{+{p84-p50:.3g}}}_{{-{p50-p16:.3g}}}$', fontsize=10)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plot_path = out_dir / 'posterior_distributions.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  → Posterior distribution figure saved to: {plot_path}")


def plot_population_comparison(Rp_Rjup: float, Teq: float, config: PlanetConfig, out_dir: Path):
    """Places the target companion within the broader NASA Exoplanet Archive demographic context."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_file = Path('nasa_exoplanet_cache_universal.csv')

    if cache_file.exists():
        df = pd.read_csv(cache_file)
    else:
        try:
            from astroquery.nasa_exoplanet_archive import NasaExoplanetArchive
            # Query the confirmed companion table without radius boundaries
            raw = NasaExoplanetArchive.query_criteria(
                table="pscomppars",
                select="pl_name, pl_rade, pl_eqt, pl_orbper, pl_bmassj",
                where="pl_orbper < 100 AND pl_eqt > 0" 
            )
            df = raw.to_pandas()
            df.to_csv(cache_file, index=False)
        except Exception:
            # Fallback population dataset representing diverse demographic regimes
            df = pd.DataFrame({
                'pl_rade': [2.0, 4.0, 8.9, 10.7, 11.8, 12.7, 15.2, 2.5, 6.0],
                'pl_eqt':  [400, 800, 1240, 1620, 1850, 1450, 2180, 500, 1000]
            })

    df['pl_eqt'] = pd.to_numeric(df['pl_eqt'], errors='coerce')
    df['pl_rade_jup'] = pd.to_numeric(df.get('pl_rade_jup', df.get('pl_rade', pd.Series()) / 11.2089), errors='coerce')

    # Remove incomplete records
    mask = df['pl_eqt'].notna() & df['pl_rade_jup'].notna()
    df_clean = df[mask]

    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Render background confirmed exoplanet population
    ax.scatter(df_clean['pl_eqt'], df_clean['pl_rade_jup'], c='lightgray', s=30, alpha=0.4,
               edgecolors='darkgray', linewidths=0.3, label=f'NASA Archive (N={len(df_clean)})', zorder=2)
    
    # Highlight current target candidate
    ax.scatter(Teq, Rp_Rjup, c='red', s=300, marker='*', edgecolors='darkred', linewidths=1.5,
               label=config.target_name, zorder=5)

    # Reference demarcation lines for hot / inflated regimes
    if Rp_Rjup > 0.8:
        ax.axhline(1.3, color='orange', linestyle='--', alpha=0.5, linewidth=1.2)
        ax.text(ax.get_xlim()[0] + 50, 1.35, r'Inflated ($R_p > 1.3\ R_J$)', color='darkorange', fontsize=9)
    if Teq > 1200:
        ax.axvline(1800, color='red', linestyle='--', alpha=0.5, linewidth=1.2)
        ax.text(1850, ax.get_ylim()[0] + 0.1, 'Ultra-Hot', color='red', fontsize=9, rotation=90)

    ax.set_xlabel('Equilibrium Temperature [K]', fontsize=13)
    ax.set_ylabel(r'Planetary Radius [$R_{Jup}$]', fontsize=13)
    ax.set_title('Demographic Context — NASA Exoplanet Archive', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.25, linestyle='--')
    
    # Restrict Y-axis to physical companion boundaries
    ax.set_ylim(bottom=0.0, top=min(3.5, max(Rp_Rjup * 1.5, 2.5)))
    
    # Zoom X-axis around the target's thermal regime
    teq_margin = max(500, Teq * 0.5)
    ax.set_xlim(max(0, Teq - teq_margin), Teq + teq_margin)

    plt.tight_layout()
    plot_path = out_dir / 'population_comparison.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  → Population comparison figure saved to: {plot_path}")