#!/usr/bin/env bash
#SBATCH -p compclass
#SBATCH -J lab2_mpi
#SBATCH -t 00:10:00

set -euo pipefail
module purge
module load compilers/gcc-15.2.0
module load mpi/openmpi-5.0.6 || module load mpi/openmpi-4.1.6 || module load mpi/openmpi

make -C "$SLURM_SUBMIT_DIR/lab2/mpi"
bash "$SLURM_SUBMIT_DIR/lab2/scripts/run_mpi.sh" 1000000 1e-8 10000
