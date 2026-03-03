from pathlib import Path

from euphonic import ForceConstants
import numpy as np
from snakemake.script import snakemake

fc = ForceConstants.from_phonopy(
    summary_name=Path(snakemake.input[0]).resolve(),
    fc_name=Path(snakemake.input[1]).resolve(),
)

qpts = np.array(snakemake.params["qpts"])
weights = np.array(snakemake.params["weights"])

modes = fc.calculate_qpoint_phonon_modes(
    qpts=qpts,
    weights=weights,
    asr='reciprocal',
)

modes.to_json_file(snakemake.output[0])
