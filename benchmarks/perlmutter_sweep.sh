#!/bin/bash
# 1D problem-size sweep (benchmarks/sweep_1d.py) on one NERSC Perlmutter GPU node: 1 and 2 CPU devices, 1 and 2 A100s.
# Submit from the repository root:   sbatch -A <your_project>_g benchmarks/perlmutter_sweep.sh
# Interactive (inside salloc, from the repository root):   SLURM_SUBMIT_DIR=$PWD bash benchmarks/perlmutter_sweep.sh
# Results: outputs/sweep_1d/sweep_perlmutter.json and sweep_1d_perlmutter.png
#SBATCH -J pypic3d-sweep-1d
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH -t 03:00:00
#SBATCH -o pypic3d_sweep_1d_%j.log

module load python
ENV=$SCRATCH/pypic3d-env
if [ ! -d "$ENV" ]; then
    python -m venv "$ENV"
    "$ENV/bin/pip" install -q -U "jax[cuda12]" matplotlib
    "$ENV/bin/pip" install -q -e "$SLURM_SUBMIT_DIR"
fi
cd "$SLURM_SUBMIT_DIR"
echo "Starting 1D sweep on $(hostname) at $(date)"
"$ENV/bin/python" -u benchmarks/sweep_1d.py
echo "Finished at $(date)"
