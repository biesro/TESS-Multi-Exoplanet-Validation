"""
=============================================================================
TITLE: Exoplanet Physical Characterization Orchestrator (run_fit.py)
=============================================================================
Description:
    Master orchestration script for the physical characterization pipeline.
    Executes the end-to-end characterization workflow for a specified target:
      1. Photometric data retrieval, detrending, and pre-fit diagnostics.
      2. Bayesian transit modeling and parameter estimation via juliet.
      3. Monte Carlo error propagation and physical parameter derivation.
      4. High-resolution posterior plotting and population comparison.

Usage:
    - Direct execution (IDE): Uses DEFAULT_CONFIG.
    - CLI execution: python run_fit.py --config TOI_1277_02

Author: Biel Escolà-Rodrigo
=============================================================================
"""

# Compatibility fix for Windows environments using Lightkurve/Astroquery
import os
if 'HOME' not in os.environ:
    os.environ['HOME'] = os.environ.get('USERPROFILE', os.path.expanduser("~"))

import sys
import argparse
import importlib
from pathlib import Path
import warnings

# Suppress non-critical third-party library warnings
warnings.filterwarnings('ignore')

# Import core analytical engines
from core import transit_engine as te
from core import physics_models as pm

# ============================================================================
# DEFAULT TARGET CONFIGURATION
# ============================================================================
# Target configuration executed by default when run without CLI arguments.
DEFAULT_CONFIG = "TOI_1277_02_nodata"


def main():
    # 1. Parse command-line arguments (optional -c/--config flag)
    parser = argparse.ArgumentParser(
        description="Universal Exoplanet & Sub-stellar Companion Characterization Pipeline"
    )
    parser.add_argument(
        "-c", "--config", 
        required=False, 
        default=DEFAULT_CONFIG,
        help="Configuration module name inside 'configs/' (without .py extension)"
    )
    args = parser.parse_args()

    config_name = args.config.replace('.py', '')

    # 2. Dynamically import target configuration module
    try:
        config_module = importlib.import_module(f"configs.{config_name}")
        config = config_module.config
    except ModuleNotFoundError:
        print(f"\n[ERROR] Configuration file 'configs/{config_name}.py' not found.")
        print("        Verify that the file exists and the module name is correctly specified.")
        sys.exit(1)
    except AttributeError:
        print(f"\n[ERROR] Module 'configs/{config_name}.py' must define an instance named 'config'.")
        sys.exit(1)

    # 3. Initialize target-specific output directory
    out_dir = Path(config.out_folder) / config.target_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("UNIVERSAL PHYSICAL CHARACTERIZATION PIPELINE")
    print(f"Target : {config.target_name} (TIC {config.tic_id})")
    engine_desc = "Photometry + RV (Dynamical Mass)" if config.has_rv else "Photometry (mr_forecast Mass Prior)"
    print(f"Engine : {engine_desc}")
    print("=" * 70)

    try:
        # =================================================================
        # STAGE 1: Photometric Retrieval & Pre-fit Diagnostics (transit_engine)
        # =================================================================
        times, flux, flux_err, lc_raw = te.fetch_and_preprocess_tess(config, out_dir)
        te.plot_full_lc_with_transits(lc_raw, config, out_dir)
        te.plot_phased_check(times, flux, config, out_dir)

        # =================================================================
        # STAGE 2: Bayesian Transit Model Fitting (transit_engine)
        # =================================================================
        results = te.run_juliet_fit(times, flux, flux_err, config, out_dir)

        # =================================================================
        # STAGE 3: Derived Physical Parameters & Monte Carlo Propagation (physics_models)
        # =================================================================
        mc_results = pm.propagate_uncertainties(results, config)

        # =================================================================
        # STAGE 4: Synthesis, Diagnostic Reporting & Plotting (physics_models)
        # =================================================================
        print("\n[INFO] Generating derived physical results and publication figures...")
        cls, Rp, Mp, Teq, rho = pm.print_results(mc_results, config)
        pm.plot_posterior_distributions(mc_results, config, out_dir)
        pm.plot_population_comparison(Rp, Teq, config, out_dir)
        pm.save_summary(mc_results, config, cls, out_dir)

        print("\n" + "=" * 70)
        print(f"[SUCCESS] Characterization successfully completed for {config.target_name}")
        print("=" * 70)
        print(f"[INFO] All output products saved to: {out_dir}/")
        
    except Exception as e:
        print("\n" + "=" * 70)
        print(f"[CRITICAL ERROR] Pipeline execution failed: {e}")
        print("=" * 70)
        raise


if __name__ == '__main__':
    main()