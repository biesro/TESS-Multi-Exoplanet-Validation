"""
=============================================================================
SUB-PIXEL CENTROID ANALYSIS: TOI 2392.02
=============================================================================

Measures phase-folded sub-pixel centroid shifts during a transit 
to test for a background eclipsing binary (BEB) false positive scenario. 
A genuine on-target transit produces no significant centroid displacement; 
a BEB produces a systematic shift correlated with transit phase.

Method:
    1. Extract pixel-level centroid time series from TESS TPF using
       flux-weighted moment estimation.
    2. Remove spacecraft drift using a dynamically calculated median filter 
       kernel. Kernel size is automatically chosen to exceed the transit 
       duration (~3x) to avoid self-subtraction of the centroid signal.
    3. Phase-fold the residual (drift-corrected) centroid shifts and bin
       at 30-minute resolution.
    4. Assess peak shift amplitude against a ±0.005-pixel threshold.
"""

import lightkurve as lk
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import medfilt
import warnings

warnings.simplefilter('ignore')

# ---------------------------------------------------------------------------
# CONFIGURATION (EDIT FOR EACH TARGET)
# ---------------------------------------------------------------------------
TIC_ID                 = "TIC 149349867"
TOI_ID                 = "TOI 2392.02"
SECTOR                 = 98          # Target Sector for TPF extraction
KNOWN_PERIOD           = 3.407231   # Orbital period in days (extret del BLS)
KNOWN_T0               = 2039.4186   # Transit epoch in BTJD (extret del BLS)
TRANSIT_DURATION_HOURS = 3.84        # Critical for dynamic kernel sizing (extret del BLS)

# Visualization phase window
PHASE_XLIM = (TRANSIT_DURATION_HOURS / 24) * 2  

# Centroid displacement threshold for BEB suspicion (pixels)
CENTROID_THRESHOLD = 0.005  

# Binning resolution (30 minutes converted to days)
BIN_PHASE = 30 / (24 * 60)  

# Transit half-duration in days for in-transit masking
TRANSIT_HALF_DURATION_DAYS = (TRANSIT_DURATION_HOURS / 2) / 24  

# ---------------------------------------------------------------------------
# TPF RETRIEVAL AND CENTROID EXTRACTION
# ---------------------------------------------------------------------------
print(f"[INFO] Initiating centroid analysis for {TOI_ID} ({TIC_ID})")
print("-" * 70)

try:
    print(f"[INFO] Querying MAST for TESS Target Pixel Files (Sector {SECTOR})...")
    search = lk.search_targetpixelfile(TIC_ID, author="SPOC", sector=SECTOR)
    if len(search) == 0:
        search = lk.search_targetpixelfile(TIC_ID, author="TESS-SPOC", sector=SECTOR)
    if len(search) == 0:
        raise ValueError("No TPF found for the specified target and sector.")

    tpf = search.download()
    print(f"[INFO] TPF successfully retrieved. Total cadences: {len(tpf.time)}")

    # Flux-weighted moment centroid estimation
    print("[INFO] Calculating flux-weighted moment centroids...")
    centroids = tpf.estimate_centroids(method='moments')
    x_raw = centroids[0].value
    y_raw = centroids[1].value
    time  = tpf.time.value

    # Quality mask: filter NaN values
    valid_mask = ~np.isnan(x_raw) & ~np.isnan(y_raw)
    time_clean = time[valid_mask]
    x_clean    = x_raw[valid_mask]
    y_clean    = y_raw[valid_mask]
    print(f"[INFO] Valid cadences after NaN filtering: {valid_mask.sum()}")

    # ---------------------------------------------------------------------------
    # SYSTEMATIC DRIFT REMOVAL (Dynamic Median Filter)
    # ---------------------------------------------------------------------------
    cadence_days = np.nanmedian(np.diff(time_clean))
    cadence_minutes = cadence_days * 24 * 60
    
    # Kernel window must be ~3x the transit duration to avoid self-subtraction
    window_hours = TRANSIT_DURATION_HOURS * 3
    
    calculated_kernel = int((window_hours * 60) / cadence_minutes)
    
    if calculated_kernel < 11:
        calculated_kernel = 11
        
    if calculated_kernel % 2 == 0:
        calculated_kernel += 1

    print(f"[INFO] Data cadence detected: {cadence_minutes:.1f} minutes.")
    print(f"[INFO] Applying median filter (dynamic kernel = {calculated_kernel} cadences, ~{window_hours:.1f}h)...")

    trend_x = medfilt(x_clean, kernel_size=calculated_kernel)
    trend_y = medfilt(y_clean, kernel_size=calculated_kernel)
    shift_x = x_clean - trend_x
    shift_y = y_clean - trend_y

    # ---------------------------------------------------------------------------
    # PHASE FOLDING AND BINNING
    # ---------------------------------------------------------------------------
    print("[INFO] Folding and binning centroid shift time-series...")

    lc_x = lk.LightCurve(time=time_clean, flux=shift_x)
    lc_y = lk.LightCurve(time=time_clean, flux=shift_y)

    folded_x = lc_x.fold(period=KNOWN_PERIOD, epoch_time=KNOWN_T0)
    folded_y = lc_y.fold(period=KNOWN_PERIOD, epoch_time=KNOWN_T0)

    bin_x = folded_x.bin(time_bin_size=BIN_PHASE)
    bin_y = folded_y.bin(time_bin_size=BIN_PHASE)

    # Statistical assessment of peak in-transit amplitude
    transit_mask_x = np.abs(bin_x.time.value) < TRANSIT_HALF_DURATION_DAYS
    transit_mask_y = np.abs(bin_y.time.value) < TRANSIT_HALF_DURATION_DAYS
    
    peak_x = np.nanmax(np.abs(bin_x.flux.value[transit_mask_x])) if transit_mask_x.any() else float('nan')
    peak_y = np.nanmax(np.abs(bin_y.flux.value[transit_mask_y])) if transit_mask_y.any() else float('nan')

    print(f"[RESULT] Maximum in-transit centroid displacement:")
    print(f"         Peak ΔX (columns): {peak_x:.4f} pixels")
    print(f"         Peak ΔY (rows)   : {peak_y:.4f} pixels")
    print(f"         Decision limit   : ±{CENTROID_THRESHOLD:.3f} pixels")

    if peak_x < CENTROID_THRESHOLD and peak_y < CENTROID_THRESHOLD:
        print("[RESULT] Centroid stability confirmed. Origin is consistent with the target star.")
    else:
        print("[WARNING] Significant centroid shift detected. Potential background eclipsing binary (BEB).")

    # ---------------------------------------------------------------------------
    # DIAGNOSTIC VISUALIZATION
    # ---------------------------------------------------------------------------
    print("[INFO] Rendering diagnostic figures...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Panel X: Column-axis centroid shift
    folded_x.scatter(ax=ax1, s=2, alpha=0.15, c='gray', label='Drift-corrected shift')
    bin_x.plot(ax=ax1, c='red', lw=2.5, label='30-min binned trend')
    ax1.axhline( CENTROID_THRESHOLD, color='orange', ls=':', alpha=0.85, label=f'±{CENTROID_THRESHOLD}" threshold')
    ax1.axhline(-CENTROID_THRESHOLD, color='orange', ls=':', alpha=0.85)
    ax1.axvline(0, color='white', ls='--', alpha=0.3, lw=1)
    
    # Highlight full transit duration window
    ax1.axvspan(-TRANSIT_HALF_DURATION_DAYS, TRANSIT_HALF_DURATION_DAYS, color='red', alpha=0.1, label='Transit Window')

    ax1.set_title(r'Centroid Shift — X ($\Delta$ Columns)', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Phase [days]', fontsize=11)
    ax1.set_ylabel('Shift [pixels]', fontsize=11)
    ax1.set_xlim(-PHASE_XLIM, PHASE_XLIM)
    ax1.set_ylim(-0.02, 0.02)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Panel Y: Row-axis centroid shift
    folded_y.scatter(ax=ax2, s=2, alpha=0.15, c='gray')
    bin_y.plot(ax=ax2, c='steelblue', lw=2.5, label='30-min binned trend')
    ax2.axhline( CENTROID_THRESHOLD, color='orange', ls=':', alpha=0.85, label=f'±{CENTROID_THRESHOLD}" threshold')
    ax2.axhline(-CENTROID_THRESHOLD, color='orange', ls=':', alpha=0.85)
    ax2.axvline(0, color='white', ls='--', alpha=0.3, lw=1)
    
    # Highlight full transit duration window
    ax2.axvspan(-TRANSIT_HALF_DURATION_DAYS, TRANSIT_HALF_DURATION_DAYS, color='steelblue', alpha=0.1, label='Transit Window')

    ax2.set_title(r'Centroid Shift — Y ($\Delta$ Rows)', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Phase [days]', fontsize=11)
    ax2.set_ylabel('Shift [pixels]', fontsize=11)
    ax2.set_xlim(-PHASE_XLIM, PHASE_XLIM)
    ax2.set_ylim(-0.02, 0.02)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.suptitle(f"Sub-pixel Centroid Analysis — {TOI_ID} (Sector {SECTOR})", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

except Exception as e:
    print(f"[ERROR] Centroid analysis pipeline failure: {e}")
    import traceback
    traceback.print_exc()

print("[INFO] Analysis concluded.")