#!/bin/bash
# set the number of nodes and processes per node
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --partition=short

# set max wallclock time
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:1

# set name of job
#SBATCH --job-name=hello_omp

# Use our reservation
#SBATCH --reservation=250620-cwm


module purge
module load GCC/10.3.0

# set number of threads to use
export OMP_NUM_THREADS=5

./hello_omp
