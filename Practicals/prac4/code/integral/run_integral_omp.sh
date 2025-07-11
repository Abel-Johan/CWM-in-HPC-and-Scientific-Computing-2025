#!/bin/bash
# set the number of nodes and processes per node
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --partition=short

# set max wallclock time
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:1

# set name of job
#SBATCH --job-name=integral

# Use our reservation
#SBATCH --reservation=250620-cwm


module purge
module load GCC/10.3.0

export OMP_NUM_THREADS=16

./integral_omp < integralE4.inp

./integral_omp < integralE5.inp

./integral_omp < integralE6.inp

