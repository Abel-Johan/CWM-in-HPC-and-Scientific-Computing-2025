#!/bin/bash
# set the number of nodes and processes per node
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=short

# set max wallclock time
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:1

# set name of job
#SBATCH --job-name=helloworld_scheduling

# Use our reservation
#SBATCH --reservation=250620-cwm

module purge
module load CUDA

./reduction_cuda < reduction.inp
