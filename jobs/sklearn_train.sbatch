#!/bin/bash
#SBATCH --job-name=sklearn_train
#SBATCH --partition=fantacone
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:15:00
#SBATCH --output=output/sklearn-%j.out


# ---- Conda (NON-INTERACTIVE SAFE) ----
source /share/apps/anaconda3/2024.06-1/etc/profile.d/conda.sh
conda activate sklearn-cpu

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

python3 - <<'PY'
import numpy as np, time, os
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

print("Threads:", os.environ.get("OMP_NUM_THREADS"))

X, y = make_classification(
    n_samples=600_000,
    n_features=40,
    n_informative=25,
    random_state=42
)

clf = RandomForestClassifier(
    n_estimators=200,
    n_jobs=4,
    max_depth=20
)

t0 = time.time()
clf.fit(X, y)
print("Training time:", time.time() - t0)
PY
