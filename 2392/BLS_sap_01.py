"""
=============================================================================
TITLE: Two-Stage Photometric Detection and Characterization Pipeline
=============================================================================
Description:
    Performs a two-stage Box Least Squares (BLS) transit detection:
      1. Period search on a dynamically detrended light curve to maximize SNR.
      2. Depth and morphology characterization on the raw (un-detrended) data.
    Exports phase-folded light curve data for downstream statistical validation.

Inputs:
    - TIC ID and TOI identifier.
    - TESS photometric data retrieved via Lightkurve (SPOC, TESS-SPOC, or QLP).

Outputs:
    - Diagnostic figures (Raw light curve, BLS periodogram, Phase-folded transit).
    - Exported CSV files for downstream statistical validation pipelines.

Author: Biel Escolà-Rodrigo
=============================================================================
"""

import lightkurve as lk
import numpy as np
import matplotlib.pyplot as plt
import gc
import warnings
import pandas as pd

warnings.simplefilter('ignore')

# ---------------------------------------------------------------------------
# CONFIGURATION (Target-specific parameters)
# ---------------------------------------------------------------------------
TIC_ID = "TIC 149349867"
TOI_ID = "TOI 2392.01"

SEARCH_PERIOD_MIN = 5.8
SEARCH_PERIOD_MAX = 6.0
DURATIONS_TO_TEST = np.linspace(0.05, 0.35, 12)

print(f"[INFO] Initializing Two-Stage BLS detection pipeline for {TOI_ID} ({TIC_ID})")
print("-" * 70)

# ---------------------------------------------------------------------------
# DATA RETRIEVAL AND INITIAL CLEANING
# ---------------------------------------------------------------------------
print("[INFO] Querying MAST archive for TESS light curves...")

try:
    search = lk.search_lightcurve(TIC_ID, author="SPOC", exptime=120)
    if len(search) == 0:
        search = lk.search_lightcurve(TIC_ID, author="TESS-SPOC")
    if len(search) == 0:
        search = lk.search_lightcurve(TIC_ID, author="QLP")
    if len(search) == 0:
        raise ValueError("No TESS light curve data found for this target.")

    print(f"[INFO] {len(search)} sector(s) successfully matched.")

    lc_collection = []
    for lc_item in search:
        try:
            print(f"[INFO] Downloading sector: {lc_item.mission[0]}")
            temp_lc = lc_item.download(flux_column='sap_flux')
            # Initial cleaning: remove NaNs, normalize flux, and clip extreme outliers
            temp_lc = temp_lc.remove_nans().normalize()
            temp_lc = temp_lc.remove_outliers(sigma_upper=4, sigma_lower=40)
            lc_collection.append(temp_lc)
            del temp_lc
            gc.collect()
        except Exception as e:
            print(f"[WARNING] Failed to process sector data: {e}")
            continue

    if not lc_collection:
        raise ValueError("No usable light curve data available after cleaning.")

    lc_combined = lk.LightCurveCollection(lc_collection).stitch()
    lc_combined.flux = lc_combined.flux.astype(np.float32)
    print(f"[INFO] Light curve successfully assembled. Total cadences: {len(lc_combined)}")

except Exception as e:
    print(f"[ERROR] Pipeline execution aborted: {e}")
    exit()

# ---------------------------------------------------------------------------
# STAGE 1: DYNAMIC DETRENDED BLS SEARCH
# ---------------------------------------------------------------------------
print("\n[INFO] Executing high-resolution multi-duration BLS search...")

# Calculate cadence in days to establish a dynamic flattening window
cadence_days = np.median(np.diff(lc_combined.time.value))
max_duration_days = np.max(DURATIONS_TO_TEST)

# Window must be at least 3x the max duration to prevent transit over-fitting
window_length = int((max_duration_days * 3) / cadence_days)
if window_length % 2 == 0:
    window_length += 1
if window_length < 11:  # Safety fallback for extreme cadence configurations
    window_length = 11

print(f"[INFO] Applied dynamic flattening window: {window_length} cadences")
lc_search = lc_combined.flatten(window_length=window_length)

period_grid = np.linspace(SEARCH_PERIOD_MIN, SEARCH_PERIOD_MAX, 10000)

bls = lc_search.to_periodogram(
    method='bls',
    period=period_grid,
    duration=DURATIONS_TO_TEST
)

best_period   = bls.period_at_max_power.value
best_t0       = bls.transit_time_at_max_power.value
best_duration = bls.duration_at_max_power.value

print(f"[RESULT] BLS optimal parameters:")
print(f"        Period   : {best_period:.6f} days")
print(f"        T0       : {best_t0:.4f} BTJD")
print(f"        Duration : {best_duration * 24:.2f} hours")

# ---------------------------------------------------------------------------
# STAGE 2: RAW CHARACTERIZATION (Unbiased Depth & SNR calculation)
# ---------------------------------------------------------------------------
print("\n[INFO] Computing transit signal-to-noise ratio from un-detrended photometry...")

lc_folded = lc_combined.fold(period=best_period, epoch_time=best_t0)

# Mask 1 (0.25): Clean central baseline for depth derivation (avoids ingress/egress distortions)
phase_mask_depth = (np.abs(lc_folded.time.value) < (best_duration * 0.25))

# Mask 2 (0.50): Full transit duration T14 to account for all in-transit data points
phase_mask_snr   = (np.abs(lc_folded.time.value) < (best_duration * 0.50))

# Mask 3: Out-of-Transit baseline windows for noise estimation
phase_mask_out   = (np.abs(lc_folded.time.value) > (best_duration * 0.75)) & \
                   (np.abs(lc_folded.time.value) < (best_duration * 2.25))

flux_in_depth = lc_folded.flux[phase_mask_depth]
flux_in_snr   = lc_folded.flux[phase_mask_snr]
flux_out      = lc_folded.flux[phase_mask_out]

if len(flux_in_depth) > 5 and len(flux_out) > 5:
    median_flux_out     = np.nanmedian(flux_out)
    median_flux_in      = np.nanmedian(flux_in_depth)
    depth_calc_natural  = median_flux_out - median_flux_in
    noise               = np.nanstd(flux_out)
    
    # Corrected SNR calculation incorporating 100% of in-transit cadence points
    snr_final           = (depth_calc_natural / noise) * np.sqrt(len(flux_in_snr))
    
    depth_ppm_natural   = depth_calc_natural * 1e6
    noise_ppm           = noise * 1e6
else:
    print("[WARNING] Insufficient in-transit cadences for reliable SNR computation.")
    snr_final = depth_ppm_natural = depth_calc_natural = noise = noise_ppm = 0.0

# Construct global mask for diagnostic data selection
global_epochs = np.floor((lc_combined.time.value - best_t0 + 0.5 * best_period) / best_period)
expected_transit_times = best_t0 + global_epochs * best_period
is_in_transit_global = np.abs(lc_combined.time.value - expected_transit_times) < (best_duration * 0.55)

# ---------------------------------------------------------------------------
# DIAGNOSTIC FIGURE GENERATION (3 Panels)
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(15, 12))
gs  = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.2])

lc_collection_sorted = sorted(
    lc_collection, 
    key=lambda x: x.sector if hasattr(x, 'sector') else x.meta.get('SECTOR', 0)
)

selected_sector_label = "T0 Window (Fallback)"
sector_mask = np.zeros(len(lc_combined), dtype=bool)

# Iterate backwards to locate the latest sector exhibiting active transits
for lc_sector in reversed(lc_collection_sorted):
    s_num = lc_sector.sector if hasattr(lc_sector, 'sector') else lc_sector.meta.get('SECTOR', 'Unknown')
    t_start = lc_sector.time.value[0]
    t_end = lc_sector.time.value[-1]
    
    current_mask = (lc_combined.time.value >= t_start) & (lc_combined.time.value <= t_end)
    
    if np.any(is_in_transit_global[current_mask]):
        sector_mask = current_mask
        selected_sector_label = f"Sector {s_num}"
        break
else:
    print("[WARNING] No active transits found within individual sectors. Centering plot on T0.")
    fallback_window = best_period * 0.8
    sector_mask = (np.abs(lc_combined.time.value - best_t0) <= fallback_window)

time_zoom = lc_combined.time.value[sector_mask]
flux_zoom = lc_combined.flux[sector_mask]
is_transit_zoom = is_in_transit_global[sector_mask]

# Panel A: Target Sector Window (Un-detrended raw flux)
ax0 = fig.add_subplot(gs[0, :])
ax0.scatter(time_zoom[~is_transit_zoom], flux_zoom[~is_transit_zoom], 
            color='black', s=5, alpha=0.6, label='Out-of-Transit')
ax0.scatter(time_zoom[is_transit_zoom], flux_zoom[is_transit_zoom], 
            color='red', s=20, alpha=0.9, zorder=5, label='In-Transit Mask')

ax0.set_title(f"Raw Light Curve ({selected_sector_label}) — {TOI_ID}", fontsize=13, fontweight='bold')
ax0.set_xlabel("Time [BTJD]", fontsize=11)
ax0.set_ylabel("Normalized Flux", fontsize=11)
ax0.legend(loc='upper right')
ax0.grid(True, alpha=0.3)
ax0.set_xlim(time_zoom[0], time_zoom[-1])

# Panel B: BLS Periodogram
ax1 = fig.add_subplot(gs[1, :])
ax1.plot(bls.period, bls.power, color='black', lw=1)
ax1.axvline(best_period, color='red', alpha=0.5, lw=6, label=f'Peak Period: {best_period:.6f} d')
ax1.set_title("BLS Periodogram (computed on flattened light curve)", fontsize=13, fontweight='bold')
ax1.set_xlabel("Period [days]", fontsize=11)
ax1.set_ylabel("BLS Power", fontsize=11)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Panel C: Phase-folded Transit (Dynamic scaling)
ax2     = fig.add_subplot(gs[2, :])
# Dynamic binning: target approximately 15 bins across the transit duration
bin_size = best_duration / 15 
lc_binned = lc_folded.bin(time_bin_size=bin_size)

lc_folded.scatter(ax=ax2, color='gray', alpha=0.3, s=2, label='Unbinned Flux')
lc_binned.scatter(ax=ax2, color='red', s=25, marker='o', label='Dynamically Binned', zorder=5)

ax2.set_title(f"Phase-folded Transit Profile — T0 = {best_t0:.4f} BTJD", fontsize=13, fontweight='bold')
ax2.set_xlabel("Orbital Phase [days]", fontsize=11)
ax2.set_ylabel("Normalized Flux", fontsize=11)

# Dynamic X-axis limits (3x transit duration on each side)
x_limit = best_duration * 3
ax2.set_xlim(-x_limit, x_limit)

# Dynamic Y-axis limits derived from computed depth and baseline noise
y_min = 1.0 - (depth_calc_natural * 1.5) - (noise * 3)
y_max = 1.0 + (noise * 3)

if y_min > 0.999: y_min = 0.999
if y_max < 1.001: y_max = 1.001
ax2.set_ylim(y_min, y_max)

ax2.legend(loc='lower right')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ---------------------------------------------------------------------------
# SUMMARY CONSOLE REPORT
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print(f"DETECTION SUMMARY REPORT — {TOI_ID} ({TIC_ID})")
print("=" * 70)
print(f"   Orbital Period     : {best_period:.6f} days")
print(f"   Epoch (T0)         : {best_t0:.4f} BTJD")
print(f"   Transit Duration   : {best_duration * 24:.2f} hours")
print(f"   Transit Depth      : {depth_ppm_natural:.0f} ppm (detrended baseline)")
print(f"   Signal-to-Noise    : {snr_final:.2f}")
print(f"   Baseline Noise     : {noise_ppm:.0f} ppm")
print("-" * 70)
if snr_final > 10:
    print("   Signal Status: High-significance transit-like signal (SNR > 10)")
elif snr_final > 7:
    print("   Signal Status: Marginal candidate signal (7 < SNR <= 10)")
else:
    print("   Signal Status: Low significance (SNR <= 7) — interpret with caution")
print("=" * 70)

# ---------------------------------------------------------------------------
# DATA EXPORT: STANDARDIZED FORMAT FOR DOWNSTREAM TOOLS
# ---------------------------------------------------------------------------
print("\n[INFO] Exporting phase-folded light curve products for validation...")

df_binned = pd.DataFrame({
    'phase_days'  : lc_binned.time.value,
    'flux'        : lc_binned.flux.value,
    'flux_err'    : lc_binned.flux_err.value
}).sort_values('phase_days')

fname_binned = f"{TOI_ID.replace(' ', '')}_LC_folded_binned.csv"
df_binned.to_csv(fname_binned, index=False, header=False)
print(f"[INFO] Exported binned diagnostic file: {fname_binned}")

df_raw = pd.DataFrame({
    'time_from_midtransit' : lc_folded.time.value,
    'flux'                 : lc_folded.flux.value,
    'flux_err'             : lc_folded.flux_err.value
}).sort_values('time_from_midtransit')

fname_raw = f"{TOI_ID.replace(' ', '')}_LC_folded_raw.csv"
df_raw.to_csv(fname_raw, index=False, header=False)

print(f"[INFO] Exported master / TRICERATOPS input file: {fname_raw}")
print("[INFO] Pipeline execution successfully completed.")