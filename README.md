# TESS Multi-Exoplanet Validation

![Python](https://img.shields.io/badge/python-3.10-blue)
![License](https://img.shields.io/badge/license-MIT-green)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXXX)

Code and derived results accompanying the research manuscript **“Validation and Physical Characterization of Seven TESS Exoplanets: A Compact Multi-Planet System and Targets for Atmospheric Spectroscopy”** (Biel Escolà-Rodrigo, 2026).

## Overview

This repository contains the working analysis scripts and selected outputs used to investigate seven TESS planet candidates around five host stars. The workflow combines transit-signal recovery, pixel-level and Gaia DR3 contamination checks, statistical false-positive assessment with TRICERATOPS, Bayesian transit modeling, and derived physical-parameter estimates.

The analysis reports six new statistical validations under the adopted FPP/NFPP thresholds and includes TOI-2392.03, previously validated in the literature, as a methodological consistency check. Statistical validation should not be confused with dynamical mass confirmation.

## Targets

| Candidate | System folder | Role |
| --- | --- | --- |
| TOI-1277.02 | `1277.02/` | Candidate analyzed for statistical validation |
| TOI-1737.01 | `1737.01/` | Candidate analyzed for statistical validation |
| TOI-2392.01 | `2392/` | Inner companion in the TOI-2392 system |
| TOI-2392.02 | `2392/` | Inner companion in the TOI-2392 system |
| TOI-2392.03 | `2392/` | Previously validated companion; consistency check |
| TOI-4450.01 | `4450.01/` | Candidate analyzed for statistical validation |
| TOI-7721.01 | `7721.01/` | Candidate analyzed for statistical validation; FPP near the adopted threshold |

## Repository layout

```text
1277.02/             Target-specific validation files
1737.01/             Target-specific validation files
2392/                Validation files for the three TOI-2392 candidates
4450.01/             Target-specific validation files
7721.01/             Target-specific validation files
Characterization/    Transit modeling and derived physical characterization
Results/             Collected analysis outputs
astro_data_v5.yml     General analysis Conda environment
triceratops_plus.yml TRICERATOPS-focused Conda environment
ENVIRONMENT.md       Environment setup and usage notes
CITATION.cff         Software citation metadata
LICENSE              Software license
```

The directory organization follows the author's original research workspace rather than a packaged software-library layout.

## Methods

- TESS photometry retrieval and signal search with Lightkurve and Box Least Squares (BLS).
- Pixel-level centroid and Gaia DR3 source-contamination analysis.
- Bayesian astrophysical false-positive probabilities (FPP and NFPP) estimated with TRICERATOPS.
- Transit fitting with `juliet`/`dynesty`, with circular orbits assumed.
- Empirical mass estimates based on the mass–radius relation (`mr_forecast`), followed by Monte Carlo propagation of physical parameters.

**Important:** The planetary masses in the manuscript are inferred from radii, not measured from radial velocities. Bulk densities and atmospheric metrics consequently inherit model assumptions. See the manuscript for detailed priors, limitations, and definitions.

## Software environments

Two separate Conda environments were used: `triceratops_plus` for the TRICERATOPS statistical-validation work and `astro_data_v5` for the remaining processing and characterization tasks. Both exports specify Python 3.10.19. See [ENVIRONMENT.md](ENVIRONMENT.md) for installation instructions and platform caveats.

## Running the analysis

The repository preserves research scripts organized by target. It is **not** currently presented as a one-command reproducible pipeline. Review each script's inputs and local paths before execution. The exact execution order, command-line arguments (if any), and mapping from scripts to final manuscript tables/figures should be checked against the original run records before this section is expanded.

## Data availability

The underlying TESS photometric observations are publicly available through [MAST](https://mast.stsci.edu/). Scripts and selected derived outputs are contained in this repository. Refer to the manuscript for the observational sectors and methodology.

## Citation

If you use the code or derived results from this repository, please cite the archived software release on Zenodo and the associated research manuscript, as appropriate.

Citation metadata are provided in [CITATION.cff](CITATION.cff).

## License

See [LICENSE](LICENSE) for the repository software license. Rights to third-party data and software remain with their respective holders.
