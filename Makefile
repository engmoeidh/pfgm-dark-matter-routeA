.PHONY: env data figs tables test all

env:
\tconda env create -f environment.yml || conda env update -f environment.yml

data:
\tpython -m src.pfgm.dataio --validate data/raw || true
\tpython scripts/ggl_to_deltasigma.py --input data/raw/kids/KiDS_binA.csv --output data/processed/lensing/DeltaSigma_binA_clean.csv

figs:
\tpython scripts/btfr_build.py || true
\tpython scripts/ggl_to_deltasigma.py || true

tables:
\tpython -m src.pfgm.stats --summarize results/tables || true

test:
\tpytest -q || true

all: data figs tables test
