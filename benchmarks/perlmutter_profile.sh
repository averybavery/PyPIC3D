#!/bin/bash
# Stage-by-stage profile of one PyPIC3D step (benchmarks/profile_step.py) on 1 and 2 A100s of one Perlmutter GPU node.
# Submit from the repository root:   sbatch -A <your_project>_g benchmarks/perlmutter_profile.sh
# Interactive (inside salloc, from the repository root):   SLURM_SUBMIT_DIR=$PWD bash benchmarks/perlmutter_profile.sh
# Results: printed table in the log, outputs/profile_step/profile_perlmutter.json, traces in outputs/profile_step/trace
#SBATCH -J pypic3d-profile
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH -t 01:00:00
#SBATCH -o pypic3d_profile_%j.log

module load python
ENV=$SCRATCH/pypic3d-env
if [ ! -d "$ENV" ]; then
    python -m venv "$ENV"
    "$ENV/bin/pip" install -q -U "jax[cuda12]" matplotlib
    "$ENV/bin/pip" install -q -e "$SLURM_SUBMIT_DIR"
fi
cd "$SLURM_SUBMIT_DIR"
echo "Starting step profile on $(hostname) at $(date)"
"$ENV/bin/python" -u benchmarks/profile_step.py
echo "Finished at $(date)"
