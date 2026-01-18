#!/bin/bash
#SBATCH --job-name=stress_cpu
#SBATCH --partition=fantacone
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --output=output/stress_cpu-%j.out

echo "Node: $(hostname)"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"

srun --cpu-bind=cores \
  stress-ng --cpu ${SLURM_CPUS_PER_TASK} \
            --cpu-method matrixprod \
            --timeout 300s \
            --metrics-brief
