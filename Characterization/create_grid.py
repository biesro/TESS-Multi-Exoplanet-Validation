"""
=============================================================================
TITLE: High-Resolution Photographic Grid Stitcher for Figure 4
=============================================================================
Description:
    Stitches individual post-fit transit model PNG plots into a unified,
    publication-ready 3x2 grid (Figure 4) for multi-planet validation.
    
    Layout Scheme (3x2):
      Row 1: TOI-1277.02 | TOI-1737.01
      Row 2: TOI-2392.01 | TOI-2392.02
      Row 3: TOI-4450.01 | TOI-7721.01

Author: Biel Escolà-Rodrigo
=============================================================================
"""

import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Main directory containing all TOI target folders
base_path = r"C:\Users\Usuari\Desktop\Estudi_Espai\NIVELL_1_El_Rellotger_(Sèries_Temporals_Estàtiques)\Auditor\Super_analisi\results"

# List of the 6 newly validated planets (Excluding the benchmark TOI-2392.03)
targets = [
    "TOI_1277_02", "TOI_1737_01",
    "TOI_2392_01", "TOI_2392_02",
    "TOI_4450_01", "TOI_7721_01"
]

# Set up the 3x2 subplots grid (inches: 18 wide, 18 high)
fig, axes = plt.subplots(3, 2, figsize=(18, 18))

# Flatten the 3x2 axes array to iterate in 1D
axes = axes.flatten()

found_all = True

for i, target in enumerate(targets):
    plot_path = os.path.join(base_path, target, f"{target}_post_fit_juliet.png")
    
    if os.path.exists(plot_path):
        # Load high-res individual plot
        img = mpimg.imread(plot_path)
        axes[i].imshow(img)
        axes[i].axis('off')  # Strip outer border and axes to seamlessly join them
    else:
        print(f"  -> [WARNING] Missing plot: {plot_path}")
        axes[i].text(0.5, 0.5, f"{target}\nPlot Not Found", 
                     ha='center', va='center', fontsize=16, color='red')
        axes[i].axis('off')
        found_all = False

# Tighten whitespace borders dynamically
plt.subplots_adjust(wspace=0.01, hspace=0.01)

# High-resolution export
out_grid = os.path.join(base_path, "Figure_4_Newly_Validated_Grid.png")
plt.savefig(out_grid, dpi=300, bbox_inches='tight')
plt.close()

if found_all:
    print(f"\n ✓ Stitched successfully into Figure 4 grid: {out_grid}\n")
else:
    print(f"\n ⚠️ Stitched grid created at '{out_grid}', but some panels are missing.")