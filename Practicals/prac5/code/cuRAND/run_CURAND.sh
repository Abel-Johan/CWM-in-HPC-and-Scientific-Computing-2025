#!/bin/bash
# set the number of nodes and processes per node
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --partition=short

# set max wallclock time
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:1

# set name of job
#SBATCH --job-name=cuFFT-C2C

# Use our reservation
#SBATCH --reservation=250620-cwm

export OMP_NUM_THREADS=16

module purge
module load CUDA

./cuRAND_part_c
