"""Convert .castep_bin force constants to Euphonic JSON."""

from euphonic import QpointPhononModes
from snakemake.script import snakemake

modes = QpointPhononModes.from_castep(snakemake.input[0])
modes.to_json_file(snakemake.output[0])
