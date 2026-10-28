#!/bin/bash
# Large 2D run (benchmarks/large_2d.py): 800 x 800 cells, 500M particles, CPU cores vs 1, 2, 4 A100s (80 GB).
# PyPIC3D itself is a clean clone of UWPlasma/PyPIC3D main, so no local code changes are timed.
# Submit from the repository root:   sbatch -A <your_project>_g benchmarks/perlmutter_2d.sh
# Interactive (inside salloc, from the repository root):   SLURM_SUBMIT_DIR=$PWD bash benchmarks/perlmutter_2d.sh
# Results: outputs/large_2d/large_2d_perlmutter.json and large_2d_perlmutter.png
#SBATCH -J pypic3d-large-2d
#SBATCH -C gpu&hbm80g
#SBATCH -q regular
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH -t 05:00:00
#SBATCH -o pypic3d_large_2d_%j.log

module load python
ENV=$SCRATCH/pypic3d-env
if [ ! -d "$ENV" ]; then
    python -m venv "$ENV"
    "$ENV/bin/pip" install -q -U "jax[cuda12]" matplotlib
    "$ENV/bin/pip" install -q -e "$SLURM_SUBMIT_DIR"
fi
UPSTREAM=$SCRATCH/pypic3d-upstream
[ -d "$UPSTREAM" ] || git clone -q https://github.com/UWPlasma/PyPIC3D.git "$UPSTREAM"
git -C "$UPSTREAM" pull -q --ff-only
export PYTHONPATH=$UPSTREAM
cd "$SLURM_SUBMIT_DIR"
echo "Starting large 2D run on $(hostname) at $(date), PyPIC3D $(git -C "$UPSTREAM" log -1 --format='%h %s')"
"$ENV/bin/python" -u benchmarks/large_2d.py
echo "Finished at $(date)"
