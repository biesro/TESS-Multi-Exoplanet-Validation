# Software environments

The analysis was conducted using **two separate Conda environments**. The YAML files in the repository preserve the exported package versions, including their `pip` dependencies.

| Environment | File | Python | Intended use |
| --- | --- | --- | --- |
| `triceratops_plus` | `triceratops_plus.yml` | 3.10.19 | Statistical false-positive analysis with TRICERATOPS (FPP/NFPP) |
| `astro_data_v5` | `astro_data_v5.yml` | 3.10.19 | Other photometric analysis, Gaia/centroid work, transit fitting, characterization, and figures |

## Recreating the environments

Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda and run the following commands from the repository root:

```bash
conda env create -f triceratops_plus.yml
conda env create -f astro_data_v5.yml
```

Activate the environment appropriate to the task:

```bash
conda activate triceratops_plus
# Run the relevant TRICERATOPS script, as documented in its folder.
```

```bash
conda activate astro_data_v5
# Run the relevant general-analysis or characterization script.
```

To inspect the interpreter and installed versions:

```bash
python --version
python -m pip list
```

## Important package versions

| Package | `triceratops_plus` | `astro_data_v5` |
| --- | --- | --- |
| Python | 3.10.19 | 3.10.19 |
| NumPy | 2.2.6 | 1.26.4 |
| Lightkurve | 2.5.1 | 2.5.1 |
| TRICERATOPS | 1.0.17 | 1.0.20 |
| juliet | Not listed | 2.2.8 |
| dynesty | Not listed | 3.0.0 |
| batman-package | Not listed | 2.5.3 |

Although `triceratops` also appears in `astro_data_v5`, the intended environment for TRICERATOPS FPP/NFPP analyses is `triceratops_plus`. Verify the specific environment used for the final manuscript runs against the original logs before describing those results as reproducible from these exports.

## Platform and reproducibility notes

- These are exports from a **Windows** Conda installation. Some packages are platform-specific, and exact recreation on Linux or macOS is not guaranteed.
- Package versions are retained; the machine-specific Conda `prefix` path is omitted.
- A successful environment installation does **not** by itself guarantee that all scientific outputs can be regenerated without adapting paths, retrieving original data, or following the correct script order.
- Do not silently upgrade package versions when attempting to reproduce the published run. Record any changes in a separate environment or log.
- TESS observations can be retrieved from [MAST](https://mast.stsci.edu/). Network access is required for scripts that query external archives.

## Execution documentation still to verify

Before declaring end-to-end reproducibility, add or confirm for each analysis stage:

1. The exact script filename(s) and execution order.
2. Input file names and public data retrieval instructions.
3. Required configuration parameters and any local paths that need changing.
4. Expected output file names under `Results/` or the target-specific directories.
5. Which archived outputs generated each manuscript table and figure.

This file deliberately does not invent commands for scripts that have not yet been inspected.
