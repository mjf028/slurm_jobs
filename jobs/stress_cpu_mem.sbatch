#!/bin/bash
#SBATCH --job-name=stress_cpu_mem
#SBATCH --partition=fantacone
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=3000M
#SBATCH --time=00:10:00
#SBATCH --output=output/stress_cpumem-%j.out

srun --cpu-bind=cores \
  stress-ng \
    --cpu ${SLURM_CPUS_PER_TASK} \
    --cpu-method matrixprod \
    --vm 1 \
    --vm-bytes 2000M \
    --vm-keep \
    --timeout 300s \
    --metrics-brief
