from tempfile import TemporaryDirectory

import mantid.simpleapi
import numpy as np
from snakemake.script import snakemake

import abins.parameters

WORKSPACE = "wrk"

if snakemake.params["raw"]:
    # Disable instrumental resolution broadening
    abins.parameters.sampling["broadening_scheme"] = "none"

if snakemake.params.get("isotropic", False):
    # Use a simpler fully-isotropic intensity model
    abins.parameters.development["isotropic_fundamentals"] = True

with TemporaryDirectory() as tmpdir:

    abins_kwargs = {
        "VibrationalOrPhononFile": snakemake.input[0],
        "AbInitioProgram": "JSON",
        "OutputWorkspace": WORKSPACE,
        "CacheDirectory": tmpdir,
    } | snakemake.params["abins_kwargs"]

    mantid.simpleapi.Abins(**abins_kwargs)

workspace = mantid.simpleapi.mtd[f"{WORKSPACE}_total"]
energy = workspace.getAxis(0).extractValues()
energy_unit = workspace.getAxis(0).getUnit().symbol().ascii()
intensity = workspace.extractY()

np.savez(
    snakemake.output[0],
    energy=energy,
    energy_unit=energy_unit,
    intensity=intensity)
