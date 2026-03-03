This directory contains scripts and notes used in development. It
should not be included in the distributed package.

## Benchmarking

Scripts to generate benchmark data from Abins. This requires Mantid so
is managed with Conda environments; we prefer not to deal with those
in the main test suite, so these scripts produce reference output
files to be stored in the main test suite.

To produce data, create a suitable Snakemake environment, e.g.

```
conda create -n snakemake -c bioconda -c conda-forge snakemake
conda activate snakemake
```

then run using parallel cores and automatically-generated conda environments:

```
cd phonon_data
snakemake -c 8 --sdm conda
```
