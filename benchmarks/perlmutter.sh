#!/bin/bash
# PyPIC3D scaling on one NERSC Perlmutter GPU node (64 CPU cores + 4 A100 GPUs).
# Submit from the repository root:   sbatch -A <your_project>_g benchmarks/perlmutter.sh
# Results: outputs/scaling/scaling_perlmutter.json (copy it next to scaling_laptop.json, then
#          python benchmarks/scaling.py --plot)
#SBATCH -J pypic3d-scaling
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH -t 01:00:00
#SBATCH -o pypic3d_scaling_%j.log

module load python
ENV=$SCRATCH/pypic3d-env
if [ ! -d "$ENV" ]; then
    python -m venv "$ENV"
    "$ENV/bin/pip" install -q -U "jax[cuda12]" matplotlib
    "$ENV/bin/pip" install -q -e "$SLURM_SUBMIT_DIR"
fi
cd "$SLURM_SUBMIT_DIR"
echo "Starting PyPIC3D scaling on $(hostname) at $(date)"
"$ENV/bin/python" -u benchmarks/scaling.py
echo "Finished at $(date)"
