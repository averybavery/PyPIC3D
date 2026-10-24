#!/bin/bash
# PyPIC3D multi-GPU strong scaling on larger problems: one Perlmutter GPU node (64 CPU cores + 4 A100 GPUs).
# Grids 64^3 (4.2M particles) and 128^3 (33.5M particles); 1, 2, 4 GPUs plus 1-core and 64-core CPU baselines.
# Submit from the repository root:   sbatch -A <your_project>_g benchmarks/perlmutter_large.sh
# Results: outputs/scaling/scaling_perlmutter_<grid>.json; redraw with python benchmarks/scaling.py --plot
#SBATCH -J pypic3d-scaling-large
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH -t 02:00:00
#SBATCH -o pypic3d_scaling_large_%j.log

module load python
ENV=$SCRATCH/pypic3d-env
if [ ! -d "$ENV" ]; then
    python -m venv "$ENV"
    "$ENV/bin/pip" install -q -U "jax[cuda12]" matplotlib
    "$ENV/bin/pip" install -q -e "$SLURM_SUBMIT_DIR"
fi
cd "$SLURM_SUBMIT_DIR"
export XLA_PYTHON_CLIENT_PREALLOCATE=false SCALING_CORES=1,64 SCALING_DEVICES=1
for GRID in 64,64,64 128,128,128; do
    echo "Starting grid $GRID on $(hostname) at $(date)"
    SCALING_GRID=$GRID "$ENV/bin/python" -u benchmarks/scaling.py
done
echo "Finished at $(date)"
