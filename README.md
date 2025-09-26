# PFGM Dark Matter — Route-A (Reproducible Repo)

This repository reproduces the Proto-Field Gravity Model (PFGM) Route-A
rotation-curve and galaxy–galaxy lensing results (KiDS/SDSS), with
figures and tables intended for editorial verification.

**Units (comoving end-to-end):**
- R [Mpc], k [1/Mpc], P(k) [Mpc^3], and ΔΣ [M⊙/kpc^2].

## Quick Start (Option A: Conda)

conda env create -f environment.yml
conda activate pfgm-dm
make all

## Quick Start (Option B: venv + pip)

python -m venv .venv
source .venv/Scripts/activate # (Git Bash)
pip install -r requirements.txt
python scripts/btfr_build.py
python scripts/ggl_to_deltasigma.py
python -m src.pfgm.stats --summarize results/tables
