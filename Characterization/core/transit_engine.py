"""
=============================================================================
TITLE: Bayesian Transit Engine and Photometric Pipeline (core/transit_engine.py)
=============================================================================
Description:
    Inference and photometric analysis engine for the exofit-framework.
    Handles TESS SPOC light curve retrieval, automated windowing and binning,
    optional multi-format ground-based photometric ingestion (ExoFOP/AstroImageJ),
    spectroscopic radial velocity data handling, prior construction, and
    Bayesian nested sampling via juliet and Dynesty.

Author: Biel Escolà-Rodrigo
=============================================================================
"""

import os
import shutil
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import corner
import juliet
import lightkurve as lk
import astropy.units as u

from configs.base_config import PlanetConfig


# ============================================================================
# DATA ACQUISITION AND PREPROCESSING (TESS + GROUND-BASED PHOTOMETRY)
# ============================================================================

def fetch_and_preprocess_tess(config: PlanetConfig, out_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, lk.LightCurve]:
    """
    Downloads the TESS SPOC light curve and applies configured transit windowing
    and temporal binning.
    """
    print("\n[1] Photometric data acquisition (TESS SPOC)...")
    try:
        search = lk.search_lightcurve(f"TIC {config.tic_id}", mission="TESS", author="SPOC", exptime=120)
        if len(search) == 0:
            search = lk.search_lightcurve(f"TIC {config.tic_id}", mission="TESS")
        if len(search) == 0:
            raise RuntimeError(f"No photometric data found for TIC {config.tic_id}")
            
        lc_raw = search.download_all().stitch().remove_nans()
        print(f"  → Downloaded {len(lc_raw)} cadences (TESS)")
    except Exception as e:
        print(f"  ✗ Error downloading TESS data: {e}")
        raise

    epoch_btjd = config.orbit.epoch_bjd - 2457000.0

    if config.preprocessing.use_windows:
        phase = ((lc_raw.time.value - epoch_btjd) % config.orbit.period) / config.orbit.period
        phase = np.where(phase > 0.5, phase - 1, phase)
        dur_d = config.orbit.duration_hours / 24.0
        mask  = np.abs(phase) < (dur_d * config.preprocessing.window_factor / config.orbit.period)
        lc_windowed = lc_raw[mask]
    else:
        lc_windowed = lc_raw

    lc_clean = lc_windowed.remove_nans()
    
    # Temporal binning with analytical error propagation
    if config.preprocessing.binning_min > 0:
        bin_time = config.preprocessing.binning_min * u.minute
        lc_final = lc_clean.bin(time_bin_size=bin_time).remove_nans()
    else:
        lc_final = lc_clean

    times    = lc_final.time.value
    flux     = lc_final.flux.value
    flux_err = lc_final.flux_err.value
    
    print(f"  → Retained {len(lc_final)} TESS cadences (reduced from {len(lc_clean)} with {config.preprocessing.binning_min}-min binning)")
    return times, flux, flux_err, lc_raw


def load_ground_photometry(file_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Ingests ground-based follow-up photometric files (.tbl, .csv, .txt, .dat)
    originating from ExoFOP or AstroImageJ with flexible column schema mapping.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"[ERROR] Ground-based follow-up file not found: {file_path}")
    
    # Automated delimiter detection
    if path.suffix.lower() == '.csv':
        df = pd.read_csv(path)
    else:
        df = pd.read_csv(path, sep=r'\s+')
        
    # Standard column header matching
    time_cols = ['BJD_TDB', 'BJD', 'HJD', 'Time', 'time', 't']
    flux_cols = ['rel_flux_T1', 'rel_flux', 'Flux', 'flux', 'f']
    err_cols  = ['rel_flux_err_T1', 'rel_flux_err', 'Flux_err', 'flux_err', 'err', 'e']
    
    time_col = next((c for c in time_cols if c in df.columns), df.columns[0])
    flux_col = next((c for c in flux_cols if c in df.columns), df.columns[1])
    err_col  = next((c for c in err_cols if c in df.columns), df.columns[2])
    
    times = df[time_col].values.astype(float)
    flux = df[flux_col].values.astype(float)
    flux_err = df[err_col].values.astype(float)
    
    # Clean missing / invalid values
    mask = ~(np.isnan(times) | np.isnan(flux) | np.isnan(flux_err))
    times, flux, flux_err = times[mask], flux[mask], flux_err[mask]
    
    # Temporal alignment: convert absolute BJD (> 2.4e6) to BTJD (- 2457000.0) for TESS consistency
    if np.median(times) > 2400000:
        times = times - 2457000.0
        
    return times, flux, flux_err


# ============================================================================
# PRE-FIT DIAGNOSTICS AND VALIDATION
# ============================================================================

def plot_full_lc_with_transits(lc_raw: lk.LightCurve, config: PlanetConfig, out_dir: Path):
    """
    Generates diagnostic light curve indicating expected transit ephemerides.
    Applies an intelligent zoom to the longest, most recent continuous sector block.
    """
    epoch_btjd = config.orbit.epoch_bjd - 2457000.0
    t_all = lc_raw.time.value
    f_all = lc_raw.flux.value

    # 1. Segment observations by gaps > 10 days to identify distinct observing blocks
    gaps = np.where(np.diff(t_all) > 10.0)[0]
    start_idx = 0
    blocks = []
    
    for end_idx in gaps:
        blocks.append((start_idx, end_idx))
        start_idx = end_idx + 1
    blocks.append((start_idx, len(t_all) - 1))

    # 2. Evaluate blocks to prioritize largest continuous duration and recency
    best_block = None
    best_score = (-1, -1)  # (n_sectors, end_time)
    
    for (start, end) in blocks:
        duration = t_all[end] - t_all[start]
        n_sectors = max(1, round(duration / 27.0))
        score = (n_sectors, t_all[end])
        
        if score > best_score:
            best_score = score
            best_block = (start, end)

    # 3. Restrict data to selected primary sector block
    t = t_all[best_block[0]:best_block[1]+1]
    f = f_all[best_block[0]:best_block[1]+1]

    # 4. Render diagnostic figure
    fig, ax = plt.subplots(figsize=(20, 5))
    ax.scatter(t, f, s=0.8, alpha=0.5, c='steelblue')

    # Demarcate predicted transit centers across visible block range
    n_start = int((t.min() - epoch_btjd) / config.orbit.period) - 1
    n_end   = int((t.max() - epoch_btjd) / config.orbit.period) + 2
    for n in range(n_start, n_end):
        t_tr = epoch_btjd + n * config.orbit.period
        if t.min() <= t_tr <= t.max():
            ax.axvline(t_tr, color='red', alpha=0.7, linewidth=1.5, linestyle='--')

    ax.set_xlabel('BTJD [days]', fontsize=11)
    ax.set_ylabel('Normalized Flux', fontsize=11)
    
    n_sectors_plot = best_score[0]
    ax.set_title(f'{config.target_name} — Diagnostic Light Curve (Zoom on ~{n_sectors_plot} consecutive sector(s))', fontsize=12)
    
    # Set X-axis bounds with 1-day padding
    ax.set_xlim(t.min() - 1.0, t.max() + 1.0)
    
    # Establish robust Y-axis limits using percentiles
    q_low, q_high = np.percentile(f, [0.1, 99.9])
    ax.set_ylim(q_low, q_high)

    ax.grid(alpha=0.3)
    plt.tight_layout()
    
    plot_path = out_dir / 'diagnostic_full_lc.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → {plot_path.name} created (Filtered to primary block: ~{n_sectors_plot} sector(s))")


def plot_phased_check(times: np.ndarray, flux: np.ndarray, config: PlanetConfig, out_dir: Path):
    """
    Generates phase-folded pre-fit inspection plots with 10-minute binning
    and robust percentile-based scaling.
    """
    epoch_btjd = config.orbit.epoch_bjd - 2457000.0
    
    # Compute phase in hours
    phase = ((times - epoch_btjd) % config.orbit.period) / config.orbit.period
    phase = np.where(phase > 0.5, phase - 1, phase)
    phase_h = phase * config.orbit.period * 24.0

    depth_val = 1.0 - config.orbit.depth_ppm / 1e6
    
    # 10-minute binning grid
    bin_width_h = 10.0 / 60.0
    bins = np.arange(phase_h.min(), phase_h.max() + bin_width_h, bin_width_h)
    bin_centers, bin_flux, bin_err = [], [], []
    
    for i in range(len(bins) - 1):
        mask = (phase_h >= bins[i]) & (phase_h < bins[i+1])
        if mask.sum() >= 3:
            bin_centers.append((bins[i] + bins[i+1]) / 2.0)
            bin_flux.append(np.median(flux[mask]))
            bin_err.append(np.std(flux[mask]) / np.sqrt(mask.sum()))
            
    bin_centers = np.array(bin_centers)
    bin_flux = np.array(bin_flux)
    bin_err = np.array(bin_err)

    # Compute robust vertical limits
    q_low, q_high = np.percentile(flux, [0.1, 99.9])
    y_min = min(depth_val - (1.0 - depth_val) * 0.8, q_low)
    y_max = max(1.0 + (1.0 - depth_val) * 0.8, q_high)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False)
    fig.suptitle(f'{config.target_name} — Pre-fit Verification\nExpected Depth = {config.orbit.depth_ppm:.0f} ppm', fontsize=12)

    # ---------- PANEL 1: TRANSIT ZOOM ----------
    ax1 = axes[0]
    ax1.set_title('Transit Window Zoom', fontsize=10)
    ax1.scatter(phase_h, flux, s=2, alpha=0.15, c='steelblue', label='Individual cadences')
    
    if len(bin_centers) > 0:
        ax1.errorbar(bin_centers, bin_flux, yerr=bin_err, fmt='o', color='black', 
                     markersize=4, linewidth=1.2, capsize=2, label='10-min bins', zorder=5)
                     
    ax1.axvspan(-config.orbit.duration_hours / 2.0, config.orbit.duration_hours / 2.0, 
                alpha=0.12, color='red', label=f'Expected duration ({config.orbit.duration_hours:.1f} h)')
    ax1.axhline(depth_val, color='red', linestyle='--', alpha=0.8, label=f'Expected depth ({config.orbit.depth_ppm:.0f} ppm)')
    ax1.axhline(1.0, color='gray', linestyle=':', alpha=0.5)
    
    ax1.set_xlim(-config.orbit.duration_hours * 2.5, config.orbit.duration_hours * 2.5)
    ax1.set_ylim(y_min, y_max)
    ax1.set_ylabel('Normalized Flux', fontsize=11)
    ax1.legend(fontsize=9, loc='lower right')
    ax1.grid(alpha=0.25)

    # ---------- PANEL 2: FULL PHASE BASELINE ----------
    ax2 = axes[1]
    ax2.set_title('Full Orbital Phase Context', fontsize=10)
    ax2.scatter(phase_h, flux, s=2, alpha=0.15, c='steelblue')
    
    if len(bin_centers) > 0:
        ax2.errorbar(bin_centers, bin_flux, yerr=bin_err, fmt='o', color='black', 
                     markersize=4, linewidth=1.2, capsize=2, zorder=5)
                     
    ax2.axvspan(-config.orbit.duration_hours / 2.0, config.orbit.duration_hours / 2.0, alpha=0.12, color='red')
    ax2.axhline(depth_val, color='red', linestyle='--', alpha=0.8)
    ax2.axhline(1.0, color='gray', linestyle=':', alpha=0.5)
    
    # Confine lower panel to +/- 12 hours for clear baseline inspection
    ax2.set_xlim(max(-12.0, phase_h.min()), min(12.0, phase_h.max()))
    ax2.set_ylim(y_min, y_max)
    ax2.set_xlabel('Phase × Period [hours]', fontsize=11)
    ax2.set_ylabel('Normalized Flux', fontsize=11)
    ax2.grid(alpha=0.25)

    plt.tight_layout()
    plot_path = out_dir / 'phased_check.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → {plot_path.name} created")


# ============================================================================
# BAYESIAN INFERENCE ENGINE (JULIET + DYNESTY)
# ============================================================================

def run_juliet_fit(times: np.ndarray, flux: np.ndarray, flux_err: np.ndarray, config: PlanetConfig, out_dir: Path):
    """
    Executes the Bayesian joint fit using juliet and Dynesty nested sampling.
    Constructs analytical stellar density and limb-darkening priors, incorporates
    optional ground-based photometry and radial velocities, and generates posterior corner plots.
    """
    print("\n[2] Bayesian Inference (juliet + Dynesty)...")
    
    epoch_btjd = config.orbit.epoch_bjd - 2457000.0
    p_expected = np.sqrt(config.orbit.depth_ppm / 1e6)
    
    p_lo = max(0.001, p_expected * 0.2)
    p_hi = min(1.0,   p_expected * 2.5) 

    # Stellar bulk density derivation
    M_sun_kg = 1.9884e30
    R_sun_m  = 6.957e8
    m_star_kg = config.stellar.m_star * M_sun_kg
    r_star_m  = config.stellar.r_star * R_sun_m
    rho_star  = m_star_kg / ((4.0 / 3.0) * np.pi * r_star_m**3)
    
    rel_err_m = config.stellar.m_star_err / config.stellar.m_star
    rel_err_r = config.stellar.r_star_err / config.stellar.r_star
    rho_err = rho_star * np.sqrt(rel_err_m**2 + (3.0 * rel_err_r)**2)

    # Base dataset dictionary initialized with TESS
    tim = {'TESS': np.array(times)}
    fl  = {'TESS': np.array(flux)}
    fle = {'TESS': np.array(flux_err)}
    
    # ---------------------------------------------------------
    # PRIOR FACTORY
    # ---------------------------------------------------------
    priors = {}
    
    # Global stellar and orbital parameters
    priors['rho']   = {'distribution': 'normal',  'hyperparameters': [rho_star, rho_err]}
    priors['P_p1']  = {'distribution': 'normal',  'hyperparameters': [config.orbit.period, config.orbit.period_err]}
    priors['t0_p1'] = {'distribution': 'normal',  'hyperparameters': [epoch_btjd, config.orbit.epoch_err]}
    priors['p_p1']  = {'distribution': 'uniform', 'hyperparameters': [p_lo, p_hi]}
    priors['b_p1']  = {'distribution': 'uniform', 'hyperparameters': [0.0, 0.99]}
    
    if config.is_circular:
        priors['ecc_p1']   = {'distribution': 'fixed', 'hyperparameters': 0.0}
        priors['omega_p1'] = {'distribution': 'fixed', 'hyperparameters': 90.0}
    else:
        priors['ecc_p1']   = {'distribution': 'uniform', 'hyperparameters': [0.0, 0.95]}
        priors['omega_p1'] = {'distribution': 'uniform', 'hyperparameters': [0.0, 360.0]}

    # TESS instrumental parameters
    priors['ld_law_TESS']    = {'distribution': 'fixed',      'hyperparameters': 'quadratic'}
    priors['q1_TESS']        = {'distribution': 'normal',     'hyperparameters': [0.348, 0.05]}
    priors['q2_TESS']        = {'distribution': 'normal',     'hyperparameters': [0.246, 0.05]}
    priors['mdilution_TESS'] = {'distribution': 'fixed',      'hyperparameters': 1.0}
    priors['mflux_TESS']     = {'distribution': 'normal',     'hyperparameters': [0.0, 0.1]}
    priors['sigma_w_TESS']   = {'distribution': 'loguniform', 'hyperparameters': [0.1, 10000.0]}

    # Ground-based follow-up photometry handling (Multi-instrument joint fit)
    ground_dict = getattr(config, 'ground_photometry', None)
    if ground_dict and isinstance(ground_dict, dict):
        for inst_name, file_path in ground_dict.items():
            try:
                g_t, g_f, g_fe = load_ground_photometry(file_path)
                tim[inst_name] = g_t
                fl[inst_name]  = g_f
                fle[inst_name] = g_fe
                
                # Dynamic limb-darkening priors per optical passband
                ground_filters_dict = getattr(config, 'ground_filters', {})
                band = ground_filters_dict.get(inst_name, 'zs')  # Default to Sloan zs if unspecified

                filter_ld_priors = {
                    'TESS': [0.348, 0.246],
                    'zs':   [0.285, 0.231],  # Sloan z-short band (LCO)
                    'z':    [0.310, 0.240]   # Standard z-band (TRAPPIST, etc.)
                }

                q1_val, q2_val = filter_ld_priors.get(band, [0.30, 0.24])

                priors[f'ld_law_{inst_name}']    = {'distribution': 'fixed',      'hyperparameters': 'quadratic'}
                priors[f'q1_{inst_name}']        = {'distribution': 'normal',     'hyperparameters': [q1_val, 0.05]}
                priors[f'q2_{inst_name}']        = {'distribution': 'normal',     'hyperparameters': [q2_val, 0.05]}
                priors[f'mdilution_{inst_name}'] = {'distribution': 'fixed',      'hyperparameters': 1.0}
                priors[f'mflux_{inst_name}']     = {'distribution': 'normal',     'hyperparameters': [0.0, 0.1]}
                priors[f'sigma_w_{inst_name}']   = {'distribution': 'loguniform', 'hyperparameters': [0.1, 10000.0]}
                
                print(f"  → Loaded ground photometry [{inst_name}, band {band}]: {len(g_t)} points from '{file_path}'")
            except Exception as e:
                print(f"  ⚠️ Error loading ground photometry for '{inst_name}': {e}")

    # Radial velocity dataset handling
    if config.has_rv and config.rv_file_path is not None:
        try:
            rv_data = pd.read_csv(config.rv_file_path)
            inst_rv = 'CORALIE'
            tim[inst_rv] = rv_data.iloc[:, 0].values - 2457000.0 
            fl[inst_rv]  = rv_data.iloc[:, 1].values
            fle[inst_rv] = rv_data.iloc[:, 2].values
            
            mean_rv = float(np.mean(fl[inst_rv]))
            ptp_rv  = float(np.ptp(fl[inst_rv]))
            
            priors[f'mu_{inst_rv}']      = {'distribution': 'uniform',    'hyperparameters': [mean_rv - 2.0*ptp_rv, mean_rv + 2.0*ptp_rv]}
            priors[f'sigma_w_{inst_rv}'] = {'distribution': 'loguniform', 'hyperparameters': [0.1, 1000.0]}
            priors['K_p1']               = {'distribution': 'loguniform', 'hyperparameters': [0.1, 100000.0]}
            
            print(f"  → Loaded RV dataset: {len(tim[inst_rv])} measurements.")
        except Exception as e:
            print(f"  ⚠️ Error loading RV data: {e}")

    # ---------------------------------------------------------
    # SAMPLER EXECUTION OR CACHE RETRIEVAL
    # ---------------------------------------------------------
    juliet_out = str(out_dir / 'juliet_fit')
    cached = (Path(juliet_out) / 'posteriors.hdf5').exists()

    if config.sampler.use_cached_fit and cached:
        print(f"  ✓ CACHE HIT — Loading existing fit from '{juliet_out}'")
    else:
        if cached:
            print(f"  ⚠️ Clearing previous fit directory and re-running sampler...")
            shutil.rmtree(juliet_out)
        print(f"  → Initializing Dynesty nested sampling (nlive={config.sampler.nlive})...")

    # Separate photometric channels from spectroscopic channels
    t_lc_dict    = {k: tim[k] for k in tim if k != 'CORALIE'}
    y_lc_dict    = {k: fl[k]  for k in fl  if k != 'CORALIE'}
    yerr_lc_dict = {k: fle[k] for k in fle if k != 'CORALIE'}

    t_rv_dict    = {k: tim[k] for k in tim if k == 'CORALIE'} if config.has_rv and 'CORALIE' in tim else None
    y_rv_dict    = {k: fl[k]  for k in fl  if k == 'CORALIE'} if config.has_rv and 'CORALIE' in fl else None
    yerr_rv_dict = {k: fle[k] for k in fle if k == 'CORALIE'} if config.has_rv and 'CORALIE' in fle else None

    dataset = juliet.load(
        priors=priors,
        t_lc=t_lc_dict,
        y_lc=y_lc_dict,
        yerr_lc=yerr_lc_dict,
        t_rv=t_rv_dict,
        y_rv=y_rv_dict,
        yerr_rv=yerr_rv_dict,
        out_folder=juliet_out
    )
    
    results = dataset.fit(
        sampler='dynesty',
        nthreads=config.sampler.nthreads,
        n_live_points=config.sampler.nlive,
        dlogz=config.sampler.dlogz
    )

    if not (config.sampler.use_cached_fit and cached):
        print("  ✓ Mathematical nested sampling completed with Dynesty")

    # POSTERIOR CORNER PLOT
    try:
        post = results.posteriors['posterior_samples']
        params_to_plot = [p for p in ['p_p1', 'b_p1', 'rho', 't0_p1', 'P_p1', 'ecc_p1', 'K_p1'] if p in post]
        labels_map = {
            'p_p1': r'$R_p/R_*$', 'b_p1': r'$b$', 'rho': r'$\rho_*$', 
            't0_p1': r'$T_0$', 'P_p1': r'$P$ [d]', 'ecc_p1': r'$e$', 'K_p1': r'$K$ [m/s]'
        }
        
        data_c = np.column_stack([post[p] for p in params_to_plot])
        lbls   = [labels_map.get(p, p) for p in params_to_plot]
        
        fig = corner.corner(data_c, labels=lbls, quantiles=[0.16, 0.5, 0.84], 
                            show_titles=True, title_kwargs={"fontsize": 10})
        corner_path = out_dir / 'corner_characterization.png'
        fig.savefig(corner_path, dpi=120, bbox_inches='tight')
        plt.close(fig)
        print(f"  → {corner_path.name} created")
    except Exception as e:
        print(f"  ⚠️ Corner plot generation failed: {e}")

    return results