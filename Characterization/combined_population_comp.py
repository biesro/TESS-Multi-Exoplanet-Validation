"""
=============================================================================
TITLE: Unified Demographic Context Generator (Figure 5)
=============================================================================
Description:
    Generates a single comprehensive population plot placing all 7 analyzed 
    TOI candidates against the backdrop of the NASA Exoplanet Archive.
    Clearly visually separates the 6 newly validated candidates from the 
    previously known benchmark (TOI-2392.03) using distinct markers and annotations.

Author: Biel Escolà-Rodrigo
=============================================================================
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================================
# 1. DEFINE TARGET PARAMETERS (Fill with your median results from summary.txt)
# ============================================================================
planets_data = {
    # NEWLY VALIDATED CANDIDATES
    "TOI-1277.02 (New Validation)": {"Teq": 666,  "Rp_Rjup": 0.2746, "color": "tab:blue",   "marker": "o", "size": 120},
    "TOI-1737.01 (New Validation)": {"Teq": 1124,  "Rp_Rjup": 0.2903, "color": "tab:orange", "marker": "o", "size": 120},
    "TOI-2392.01 (New Validation)": {"Teq": 1375, "Rp_Rjup": 0.331, "color": "tab:green",  "marker": "o", "size": 120},
    "TOI-2392.02 (New Validation)": {"Teq": 1657, "Rp_Rjup": 0.2744, "color": "tab:purple", "marker": "o", "size": 120},
    "TOI-4450.01 (New Validation)": {"Teq": 879, "Rp_Rjup": 0.3829, "color": "tab:cyan",   "marker": "o", "size": 120},
    "TOI-7721.01 (New Validation)": {"Teq": 1207,  "Rp_Rjup": 0.6417, "color": "tab:pink",   "marker": "o", "size": 120},
    
    # PREVIOUSLY KNOWN BENCHMARK
    "TOI-2392.03 (Known Benchmark)": {"Teq": 1194, "Rp_Rjup": 0.217, "color": "gold", "marker": "*", "size": 500}
}

# ============================================================================
# 2. LOAD NASA EXOPLANET ARCHIVE BACKGROUND
# ============================================================================
cache_file = Path('nasa_exoplanet_cache_universal.csv')

print("Loading NASA Exoplanet Archive demographic data...")
if cache_file.exists():
    df = pd.read_csv(cache_file)
else:
    try:
        from astroquery.nasa_exoplanet_archive import NasaExoplanetArchive
        raw = NasaExoplanetArchive.query_criteria(
            table="pscomppars",
            select="pl_name, pl_rade, pl_eqt, pl_orbper, pl_bmassj",
            where="pl_orbper < 100 AND pl_eqt > 0" 
        )
        df = raw.to_pandas()
        df.to_csv(cache_file, index=False)
    except Exception as e:
        print(f"Error fetching NASA data: {e}")
        # Fallback dummy data if offline
        df = pd.DataFrame({'pl_rade': [2, 4, 10, 15], 'pl_eqt': [400, 800, 1500, 2000]})

df['pl_eqt'] = pd.to_numeric(df['pl_eqt'], errors='coerce')
# Convert Earth radii to Jupiter radii (1 R_Jup = 11.2089 R_Earth)
df['pl_rade_jup'] = pd.to_numeric(df.get('pl_rade_jup', df.get('pl_rade', pd.Series()) / 11.2089), errors='coerce')

mask = df['pl_eqt'].notna() & df['pl_rade_jup'].notna()
df_clean = df[mask]

# ============================================================================
# 3. RENDER THE POPULATION PLOT
# ============================================================================
fig, ax = plt.subplots(figsize=(13, 8))

# Plot background confirmed exoplanets
ax.scatter(df_clean['pl_eqt'], df_clean['pl_rade_jup'], c='lightgray', s=25, alpha=0.3,
           edgecolors='darkgray', linewidths=0.3, label=f'NASA Archive Confirmed (N={len(df_clean)})', zorder=1)

# Plot the 7 specific targets
for name, props in planets_data.items():
    ax.scatter(props['Teq'], props['Rp_Rjup'], 
               c=props['color'], s=props['size'], marker=props['marker'], 
               edgecolors='black', linewidths=1.0, label=name, zorder=5)

# --- EXPLICIT ANNOTATION FOR THE BENCHMARK ---
# Extract Benchmark coordinates dynamically to point the arrow
bench_teq = planets_data["TOI-2392.03 (Known Benchmark)"]["Teq"]
bench_rp = planets_data["TOI-2392.03 (Known Benchmark)"]["Rp_Rjup"]

ax.annotate('Previously Validated\n(Methodology Benchmark)', 
            xy=(bench_teq, bench_rp), 
            xytext=(bench_teq + 150, bench_rp + 0.3),
            arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=7),
            fontsize=10, fontweight='bold', color='black',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9), zorder=6)

# Reference demarcation lines
ax.axhline(1.3, color='orange', linestyle='--', alpha=0.5, linewidth=1.2)
ax.text(300, 1.35, r'Inflated Giants ($R_p > 1.3\ R_J$)', color='darkorange', fontsize=10)

ax.axhline(0.16, color='brown', linestyle='--', alpha=0.5, linewidth=1.2)
ax.text(300, 0.18, r'Fulton Gap / Rocky Regime', color='saddlebrown', fontsize=10)

# Formatting
ax.set_xlabel('Equilibrium Temperature [K]', fontsize=14)
ax.set_ylabel(r'Planetary Radius [$R_{Jup}$]', fontsize=14)
ax.set_title('Figure 5: Demographic Context of Validated TOI Candidates', fontsize=15, fontweight='bold', pad=15)

# Dynamic Limits
ax.set_ylim(bottom=0.0, top=2.2)
ax.set_xlim(left=0, right=2800)

# Clean legend
ax.legend(fontsize=9, loc='upper right', framealpha=0.9)
ax.grid(alpha=0.25, linestyle='--')

plt.tight_layout()
out_plot = 'Figure_5_Population_Demographics.png'
plt.savefig(out_plot, dpi=300, bbox_inches='tight')
plt.close(fig)

print(f" ✓ Population comparison grid successfully saved to: {out_plot}")