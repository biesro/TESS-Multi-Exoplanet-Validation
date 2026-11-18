"""
=============================================================================
TITLE: Spatial Contamination Analysis Pipeline (Multi-Sector & Gaia DR3)
=============================================================================
Description:
    Performs an advanced astrometric contamination check for TESS targets:
      - MAST coordinate auto-resolution to prevent API formatting errors.
      - Gaia DR3 query with dynamic timeout and DR2 fallback.
      - Multi-sector iteration: Evaluates all available SPOC masks to find 
        the "Worst-Case" photometric dilution scenario.
      - Native Astropy proper motion propagation per sector epoch.
      - Exact SPOC pipeline mask cross-matching.
      - Kinematic/Distance diagnostics (Parallax & RUWE) for physical association.

Inputs:
    - TIC ID and TOI identifier.

Outputs:
    - Spatial contamination diagnostic figure (TPF aperture plot & Gaia DR3 map).
    - Detailed dilution and contaminant console report.

Author: Biel Escolà-Rodrigo
=============================================================================
"""

import lightkurve as lk
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
from astropy.time import Time
import astropy.units as u
from astroquery.vizier import Vizier
from astroquery.mast import Catalogs
import numpy as np
import warnings

warnings.simplefilter('ignore')

# ---------------------------------------------------------------------------
# TARGET CONFIGURATION (Universal & Dynamic)
# ---------------------------------------------------------------------------
TIC_ID = "149349867"
TOI_ID = "TOI 2392"

print(f"[INFO] Resolving exact coordinates for TIC {TIC_ID} via MAST...")
try:
    tic_data = Catalogs.query_object(f"TIC {TIC_ID}", catalog="TIC")
    target_coords = SkyCoord(tic_data['ra'][0], tic_data['dec'][0], unit="deg", frame='icrs')
    print(f"[INFO] Coordinates locked: RA={target_coords.ra.deg:.5f}°, Dec={target_coords.dec.deg:.5f}°")
except Exception as e:
    print(f"[ERROR] Could not resolve coordinates via MAST: {e}")
    exit()

TESS_PIXEL_ARCSEC  = 21.0   
TESS_3PIX_ARCSEC   = 63.0   
GAIA_SEARCH_ARCMIN = 1.0    
RUWE_BINARY_FLAG   = 1.4    
VIZIER_ROW_LIMIT   = 1000   

# ---------------------------------------------------------------------------
# 1. GAIA KINEMATIC QUERY (DR3 with DR2 Fallback & Multiple Mirrors)
# ---------------------------------------------------------------------------
print(f"\n[INFO] Querying Gaia within {GAIA_SEARCH_ARCMIN} arcmin for {TOI_ID}...")

try:
    # Define Vizier server mirrors for redundancy against network dropouts
    vizier_mirrors = ['vizier.u-strasbg.fr', 'vizier.cfa.harvard.edu', 'vizier.nao.ac.jp']
    
    result = None
    
    # Attempt DR3 connection across available mirrors
    for mirror in vizier_mirrors:
        try:
            print(f"  -> Attempting DR3 connection via {mirror}...")
            v = Vizier(columns=["*", "+_r", "Gmag", "pmRA", "pmDE", "RUWE", "Plx"], row_limit=VIZIER_ROW_LIMIT)
            v.TIMEOUT = 120
            v.VIZIER_SERVER = mirror
            
            search_radius = u.Quantity(GAIA_SEARCH_ARCMIN, u.arcmin)
            result_dr3 = v.query_region(target_coords, radius=search_radius, catalog="I/355/gaiadr3")
            
            if result_dr3 and len(result_dr3) > 0:
                result = result_dr3
                print("  -> DR3 Query Successful.")
                break
        except Exception as query_err:
            print(f"  -> DR3 query failed on {mirror}: {query_err}")
            continue

    # Fallback to DR2 if DR3 fails across all mirrors
    if not result:
        print("[WARNING] Gaia DR3 failed across all mirrors. Attempting fallback to Gaia DR2...")
        for mirror in vizier_mirrors:
            try:
                print(f"  -> Attempting DR2 connection via {mirror}...")
                v = Vizier(columns=["*", "+_r", "Gmag", "pmRA", "pmDE", "RUWE", "Plx"], row_limit=VIZIER_ROW_LIMIT)
                v.TIMEOUT = 120
                v.VIZIER_SERVER = mirror
                
                search_radius = u.Quantity(GAIA_SEARCH_ARCMIN, u.arcmin)
                result_dr2 = v.query_region(target_coords, radius=search_radius, catalog="I/345/gaia2")
                
                if result_dr2 and len(result_dr2) > 0:
                    result = result_dr2
                    print("  -> DR2 Query Successful.")
                    break
            except Exception as query_err:
                print(f"  -> DR2 query failed on {mirror}: {query_err}")
                continue

    # Abort execution if all catalog mirrors fail to respond
    if not result or len(result) == 0:
        raise ValueError("Vizier servers are not responding or returned zero sources across all catalogs and mirrors.")

    gaia_table = result[0]
    print(f"[INFO] {len(gaia_table)} Gaia sources successfully retrieved.")

    ra_raw    = np.array(gaia_table['RA_ICRS'].filled(np.nan), dtype=float)
    dec_raw   = np.array(gaia_table['DE_ICRS'].filled(np.nan), dtype=float)
    pmra_raw  = np.array(gaia_table['pmRA'].filled(0.0), dtype=float)
    pmdec_raw = np.array(gaia_table['pmDE'].filled(0.0), dtype=float)
    
    if 'RUWE' in gaia_table.colnames:
        ruwes_raw = np.array(gaia_table['RUWE'].filled(1.0), dtype=float)
    else:
        ruwes_raw = np.ones(len(gaia_table))
        
    plxs_raw  = np.array(gaia_table['Plx'].filled(np.nan), dtype=float)
    
    mags_raw = np.array([
        row['Gmag'] if 'Gmag' in row.colnames and not np.ma.is_masked(row['Gmag'])
        else np.nan
        for row in gaia_table
    ])

    valid_mask = np.isfinite(ra_raw) & np.isfinite(dec_raw)
    valid_mask = np.isfinite(mags_raw) & valid_mask
    
    ra_clean    = ra_raw[valid_mask]
    dec_clean   = dec_raw[valid_mask]
    pmra_clean  = pmra_raw[valid_mask]
    pmdec_clean = pmdec_raw[valid_mask]
    ruwes       = ruwes_raw[valid_mask]
    plxs        = plxs_raw[valid_mask]
    mags        = mags_raw[valid_mask]

    coords_2016 = SkyCoord(ra=ra_clean*u.deg, dec=dec_clean*u.deg, 
                           pm_ra_cosdec=pmra_clean*u.mas/u.yr, 
                           pm_dec=pmdec_clean*u.mas/u.yr, 
                           obstime=Time(2016.0, format='jyear'))

except Exception as e:
    print(f"[ERROR] Gaia Query Pipeline failed: {e}")
    exit()

# ---------------------------------------------------------------------------
# 2. AUTOMATED MULTI-SECTOR RETRIEVAL & WORST-CASE EVALUATION
# ---------------------------------------------------------------------------
print("\n[INFO] Querying MAST for all available TESS Target Pixel Files...")
search_all = lk.search_targetpixelfile(target_coords, mission='TESS')

if len(search_all) == 0:
    print("[ERROR] No TPF data found for this target at MAST.")
    exit()

unique_sectors = sorted(list(set(search_all.table['sequence_number'])))
print(f"[INFO] Target observed in {len(unique_sectors)} sector(s): {unique_sectors}")
print("[INFO] Evaluating masks for worst-case dilution (Prioritizing SPOC > TESS-SPOC)...")

worst_sector_data = {
    'tpf': None,
    'coords_geom': None,
    'epoch': None,
    'separations': None,
    'max_dilution': -1.0,
    'target_idx': None,
    'target_flux': None
}

for sec in unique_sectors:
    try:
        sec_results = search_all[search_all.table['sequence_number'] == sec]
        
        spoc_mask = sec_results.table['author'] == 'SPOC'
        tess_spoc_mask = sec_results.table['author'] == 'TESS-SPOC'
        
        if any(spoc_mask):
            best_result = sec_results[spoc_mask][0]
            author_used = "SPOC"
        elif any(tess_spoc_mask):
            best_result = sec_results[tess_spoc_mask][0]
            author_used = "TESS-SPOC"
        else:
            print(f"  -> Skipping Sector {sec}: No SPOC or TESS-SPOC TPF found.")
            continue

        tpf = best_result.download()
        tpf = tpf[~np.isnan(tpf.flux.value).all(axis=(1, 2))]
        
        tpf_jd = tpf.time.value.mean() + 2457000
        epoch = Time(tpf_jd, format='jd').jyear
        
        coords_tpf_epoch = coords_2016.apply_space_motion(new_obstime=Time(epoch, format='jyear'))
        coords_geom = SkyCoord(coords_tpf_epoch.ra, coords_tpf_epoch.dec, frame='icrs')
        
        separations = target_coords.separation(coords_geom).to(u.arcsec).value
        target_idx = np.argmin(separations)
        target_flux = 10 ** (-0.4 * mags[target_idx])
        
        pix_cols, pix_rows = tpf.wcs.world_to_pixel(coords_geom)
        mask = tpf.pipeline_mask
        mask_rows, mask_cols = mask.shape
        
        dilution_flux_mask = 0.0
        
        for i in range(len(ra_clean)):
            if i == target_idx or separations[i] > 60.0:
                continue
            
            px = int(np.round(pix_cols[i]))
            py = int(np.round(pix_rows[i]))
            
            if (0 <= py < mask_rows) and (0 <= px < mask_cols):
                if mask[py, px]:
                    dilution_flux_mask += 10 ** (-0.4 * mags[i])
                    
        dilution_pct = (dilution_flux_mask / target_flux) * 100
        print(f"  -> Sector {tpf.sector} [{author_used}] (Epoch {epoch:.2f}): {dilution_pct:.2f}% inside mask")
        
        if dilution_pct > worst_sector_data['max_dilution']:
            worst_sector_data['max_dilution'] = dilution_pct
            worst_sector_data['tpf'] = tpf
            worst_sector_data['coords_geom'] = coords_geom
            worst_sector_data['epoch'] = epoch
            worst_sector_data['separations'] = separations
            worst_sector_data['target_idx'] = target_idx
            worst_sector_data['target_flux'] = target_flux

    except Exception as e:
        print(f"  -> Skipping Sector {sec} due to error: {e}")

best_tpf = worst_sector_data['tpf']
if best_tpf is None:
    print("[ERROR] Could not process any sectors.")
    exit()

print(f"\n[RESULT] Worst-case scenario found in Sector {best_tpf.sector}.")

# ---------------------------------------------------------------------------
# 3. WORST-CASE FIGURE SETUP & WCS OVERLAY
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(16, 7))

best_tpf.plot(ax=ax[0], aperture_mask=best_tpf.pipeline_mask,
              title=f"TESS TPF — TIC {TIC_ID} (Worst-Case Sector {best_tpf.sector})")

ax[1].set_facecolor('#0a0e27')
ax[1].grid(True, alpha=0.2, color='white', linestyle='--')
ax[1].set_xlabel('$\\Delta$RA (arcsec)', fontsize=12, color='white')
ax[1].set_ylabel('$\\Delta$Dec (arcsec)', fontsize=12, color='white')
ax[1].set_title(f'Gaia DR3 Map (Sector {best_tpf.sector} Epoch: {worst_sector_data["epoch"]:.2f})', 
                fontsize=13, color='white', fontweight='bold', pad=12)
ax[1].tick_params(colors='white')

spherical_offsets = target_coords.spherical_offsets_to(worst_sector_data['coords_geom'])
offsets_ra  = spherical_offsets[0].to(u.arcsec).value
offsets_dec = spherical_offsets[1].to(u.arcsec).value

pix_cols, pix_rows = best_tpf.wcs.world_to_pixel(worst_sector_data['coords_geom'])
ax[0].scatter(pix_cols + best_tpf.column, pix_rows + best_tpf.row, c='red', marker='x', 
              s=30, alpha=0.9, linewidths=1.5, label='Gaia sources')
ax[0].legend(loc='upper right', fontsize=8)

sizes = 1000 * 10 ** (-mags / 5)
colors_stars = plt.cm.plasma((mags - mags.min()) / (mags.max() - mags.min() + 0.01))

for i in range(len(ra_clean)):
    ra, dec, mag, size, color, ruwe = offsets_ra[i], offsets_dec[i], mags[i], sizes[i], colors_stars[i], ruwes[i]

    if i == worst_sector_data['target_idx']:
        ax[1].scatter(ra, dec, s=size * 2, c='gold', marker='*', edgecolors='yellow', 
                      linewidths=3, zorder=100, label=f'Target (Mag={mag:.2f})')
        ax[1].text(ra, dec - 3, f'TIC {TIC_ID}', ha='center', va='top', color='yellow', fontsize=9, fontweight='bold')
    else:
        edge_color = 'red' if ruwe > RUWE_BINARY_FLAG else 'white'
        ax[1].scatter(ra, dec, s=size, c=[color], marker='o', edgecolors=edge_color, 
                      linewidths=1.5, alpha=0.85, zorder=50)
        if mag < 16:
            ax[1].text(ra, dec + 2, f'{mag:.1f}', ha='center', va='bottom', color='white', fontsize=7, alpha=0.8)

ax[1].add_patch(plt.Circle((0, 0), TESS_PIXEL_ARCSEC, color='red', fill=False, lw=2.5, ls='--', alpha=0.75,
                           label=f'1 TESS pixel ({TESS_PIXEL_ARCSEC:.0f}")'))
ax[1].add_patch(plt.Circle((0, 0), TESS_3PIX_ARCSEC, color='lime', fill=False, lw=1.8, ls=':', alpha=0.6,
                           label=f'~3 TESS pixels ({TESS_3PIX_ARCSEC:.0f}")'))

max_offset = max(np.abs(offsets_ra).max(), np.abs(offsets_dec).max()) * 1.05
max_offset = max(max_offset, 50.0)
ax[1].set_xlim(-max_offset, max_offset)
ax[1].set_ylim(-max_offset, max_offset)
ax[1].invert_xaxis()
ax[1].set_aspect('equal')
ax[1].legend(loc='upper right', fontsize=8, facecolor='#0a0e27', edgecolor='white', labelcolor='white')

# ---------------------------------------------------------------------------
# 4. CONSOLE REPORT (Worst-Case Sector)
# ---------------------------------------------------------------------------
print(f"\n[WORST-CASE SECTOR {best_tpf.sector}] Gaia sources within 60 arcsec:")
sep_header = 'Sep (")'
plx_header = 'Plx(mas)'
print(f"{'Rank':<5} {'Mag':>6} {sep_header:>8} {plx_header:>9} {'RUWE':>6}   Note")
print("-" * 65)

sort_idx = np.argsort(worst_sector_data['separations'])
mask = best_tpf.pipeline_mask
mask_rows, mask_cols = mask.shape
target_idx = worst_sector_data['target_idx']
separations = worst_sector_data['separations']

for rank, i in enumerate(sort_idx):
    if separations[i] > 60.0:
        continue

    note = ""
    if i == target_idx:
        note = "<-- TARGET"
    else:
        px = int(np.round(pix_cols[i]))
        py = int(np.round(pix_rows[i]))
        
        in_aperture = False
        if (0 <= py < mask_rows) and (0 <= px < mask_cols):
            in_aperture = mask[py, px]
            
        if in_aperture:
            note = "IN APERTURE MASK - Contaminant"
        elif separations[i] < TESS_PIXEL_ARCSEC:
            note = "Within 1px, but OUTSIDE aperture"

    plx_str = f"{plxs[i]:.2f}" if not np.isnan(plxs[i]) else "NaN"
    print(f"{rank+1:<5} {mags[i]:>6.2f} {separations[i]:>8.1f} {plx_str:>9} {ruwes[i]:>6.2f}   {note}")

if worst_sector_data['max_dilution'] > 0:
    print(f"\n[ANALYSIS] Worst-case flux dilution inside SPOC pipeline mask: {worst_sector_data['max_dilution']:.2f}%")
else:
    print("\n[ANALYSIS] No background sources found inside the SPOC mask across all sectors.")

plt.tight_layout()
plt.show()