"""
=============================================================================
TITLE: Dynamic Post-Fit Transit Model Generator (juliet/Dynesty)
=============================================================================
Description:
    Generates publication-quality, dynamically scaled transit light curve
    figures from converged `juliet` nested sampling posteriors.
    
    Features:
    - Custom robust parsing for photometric data and priors.
    - Automated limb-darkening string bypass for `juliet` initialization.
    - Non-linear adaptive Y-axis scaling based on physical transit depth 
      and out-of-transit dispersion.
    - Dynamic X-axis boundary detection to eliminate blank margins.
    - 10-minute phase-folded binning with underlying raw cadences.

Author: Biel Escolà-Rodrigo
=============================================================================
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# WINDOWS COMPATIBILITY PATCH
# Ensure 'HOME' environment variable exists for radvel/juliet OS level calls.
# ============================================================================
if 'HOME' not in os.environ:
    os.environ['HOME'] = os.environ.get('USERPROFILE', 'C:\\')

import juliet

# Main directory containing all TOI target folders
base_path = r"C:\Users\Usuari\Desktop\Estudi_Espai\NIVELL_1_El_Rellotger_(Sèries_Temporals_Estàtiques)\Auditor\Super_analisi\results"

def bin_data(time, flux, bin_width_hours):
    """
    Groups raw photometric data points into temporal bins to increase 
    the signal-to-noise ratio for visual diagnostics.
    """
    bin_width_days = bin_width_hours / 24.0
    bins = np.arange(np.min(time), np.max(time) + bin_width_days, bin_width_days)
    
    binned_t, binned_f, binned_err = [], [], []
    for i in range(len(bins)-1):
        mask = (time >= bins[i]) & (time < bins[i+1])
        if np.sum(mask) > 0:
            binned_t.append(np.mean(time[mask]))
            binned_f.append(np.mean(flux[mask]))
            binned_err.append(np.std(flux[mask]) / np.sqrt(np.sum(mask)))
            
    return np.array(binned_t), np.array(binned_f), np.array(binned_err)

# Locate all TOI target directories
toi_folders = glob.glob(os.path.join(base_path, "TOI_*"))

for folder in toi_folders:
    juliet_path = os.path.join(folder, "juliet_fit")
    toi_name = os.path.basename(folder)
    
    if not os.path.exists(juliet_path):
        continue
        
    print(f"Processing physical model for {toi_name}...")

    # ------------------------------------------------------------------------
    # 1. MANUAL DATA INGESTION
    # Robust parsing to handle instrument tags regardless of column order.
    # ------------------------------------------------------------------------
    lc_file = os.path.join(juliet_path, "lc.dat")
    t_lc, y_lc, yerr_lc = {}, {}, {}
    
    if not os.path.exists(lc_file):
        print(f"  -> Error: 'lc.dat' not found for {toi_name}.")
        continue
        
    with open(lc_file, 'r') as f:
        for line in f:
            if line.strip() == '' or line.startswith('#'): continue
            parts = line.strip().split()
            
            try:
                # Format: Time, Flux, Error, Instrument
                time_val = float(parts[0])
                flux_val = float(parts[1])
                err_val = float(parts[2])
                inst = parts[3]
            except ValueError:
                # Format: Instrument, Time, Flux, Error
                inst = parts[0]
                time_val = float(parts[1])
                flux_val = float(parts[2])
                err_val = float(parts[3])
                
            if inst not in t_lc:
                t_lc[inst], y_lc[inst], yerr_lc[inst] = [], [], []
            t_lc[inst].append(time_val)
            y_lc[inst].append(flux_val)
            yerr_lc[inst].append(err_val)
            
    for inst in t_lc:
        t_lc[inst] = np.array(t_lc[inst])
        y_lc[inst] = np.array(y_lc[inst])
        yerr_lc[inst] = np.array(yerr_lc[inst])

    # ------------------------------------------------------------------------
    # 2. LIMB-DARKENING PRIOR SANITIZATION
    # Strips string-based LD laws from the prior file to prevent float conversion 
    # exceptions during Juliet's internal readpriors() execution.
    # ------------------------------------------------------------------------
    original_prior = os.path.join(juliet_path, "priors.dat")
    fixed_prior = os.path.join(juliet_path, "priors_fixed.dat")
    
    with open(original_prior, 'r') as f_in, open(fixed_prior, 'w') as f_out:
        for line in f_in:
            if 'ld_law' in line or 'quadratic' in line:
                continue 
            f_out.write(line)

    # ------------------------------------------------------------------------
    # 3. INITIALIZE JULIET ENVIRONMENT
    # Loads the corrected prior file and explicitly passes the LD law.
    # ------------------------------------------------------------------------
    dataset = juliet.load(priors=fixed_prior, 
                          t_lc=t_lc, y_lc=y_lc, yerr_lc=yerr_lc, 
                          ld_laws="quadratic", 
                          out_folder=juliet_path)
                          
    # Retrieve posteriors without re-running the sampler (relies on cached .pkl)
    res = dataset.fit(sampler='dynesty')
    
    # Clean up temporary prior file
    if os.path.exists(fixed_prior):
        os.remove(fixed_prior)
        
    instrument = list(t_lc.keys())[0]
    planets = [key.split('_')[1] for key in dataset.priors.keys() if key.startswith('P_')]
    if not planets:
        print(f"  -> Error: No planetary priors found for {toi_name}.")
        continue
    planet = planets[0]
    
    # Extract median posterior parameters for phase-folding
    P_med = np.median(res.posteriors['posterior_samples'][f'P_{planet}'])
    t0_med = np.median(res.posteriors['posterior_samples'][f't0_{planet}'])
    
    t = t_lc[instrument]
    y_data = y_lc[instrument]
    y_model = res.lc.evaluate(instrument)
    
    # Fold time arrays and shift transit center to phase 0 (in hours)
    phases_days = (t - t0_med + 0.5 * P_med) % P_med - 0.5 * P_med
    phases_hours = phases_days * 24.0
    
    # Sort arrays for continuous theoretical model line rendering
    sort_idx = np.argsort(phases_hours)
    phases_sort = phases_hours[sort_idx]
    y_data_sort = y_data[sort_idx]
    y_model_sort = y_model[sort_idx]
    
    bin_t, bin_f, bin_err = bin_data(phases_sort, y_data_sort, bin_width_hours=10/60)
    
    # ------------------------------------------------------------------------
    # 4. PUBLICATION-QUALITY PLOT WITH ADAPTIVE SCALING
    # ------------------------------------------------------------------------
    plt.figure(figsize=(10, 5))
    
    # Baseline and data rendering
    plt.axhline(1.0, color='grey', linestyle='--', linewidth=1, zorder=0)
    plt.scatter(phases_sort, y_data_sort, color='lightsteelblue', alpha=0.3, s=2, zorder=1, label='Individual cadences')
    plt.errorbar(bin_t, bin_f, yerr=bin_err, fmt='o', color='black', markersize=4, 
                 capsize=2, elinewidth=1, zorder=2, label='10-min bins')
    
    # Physical Bayesian Model
    plt.plot(phases_sort, y_model_sort, color='firebrick', linewidth=2.5, zorder=3, label='Juliet Model (Post-fit)')
    
    # --- NON-LINEAR ADAPTIVE Y-AXIS SCALING ---
    model_depth = 1.0 - np.min(y_model_sort)  
    noise_std = np.std(y_data_sort)           
    
    # Autonomous logic: assign zoom factor based on transit depth
    if model_depth < 0.0015:      # Shallow transits (< 1500 ppm) -> Aggressive zoom
        zoom_factor = 1.5
    elif model_depth < 0.0050:    # Medium transits (< 5000 ppm) -> Standard zoom
        zoom_factor = 2.5
    else:                         # Deep transits (> 5000 ppm) -> Panoramic view
        zoom_factor = 3.5
        
    y_pad = max(noise_std * zoom_factor, model_depth * 0.5)
    
    # --- DYNAMIC X-AXIS SCALING ---
    # Constrains the plot to +/- 8 hours, but shrinks if actual data ends earlier
    x_min = max(-8.0, np.min(phases_sort))
    x_max = min(8.0, np.max(phases_sort))
    
    plt.xlim(x_min, x_max)
    plt.ylim(1.0 - model_depth - y_pad, 1.0 + y_pad)
    # ------------------------------------------
    
    plt.title(f"{toi_name} — Post-fit Transit Model", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Time from mid-transit [hours]", fontsize=12)
    plt.ylabel("Normalized Flux", fontsize=12)
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    
    # High-resolution export
    out_plot = os.path.join(folder, f"{toi_name}_post_fit_juliet.png")
    plt.savefig(out_plot, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  -> Plot successfully generated: {out_plot}\n")