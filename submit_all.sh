#!/bin/bash
mkdir -p output

for job in jobs/*.sbatch; do
  echo "Submitting $job"
  sbatch "$job"
done

