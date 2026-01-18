#!/bin/bash
#SBATCH --job-name=sklearn_sgd
#SBATCH --partition=fantacone
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=3200M
#SBATCH --time=00:20:00
#SBATCH --output=output/sklearn-%j.out

source /share/apps/anaconda3/2024.06-1/etc/profile.d/conda.sh
conda activate sklearn-cpu

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

python - <<'PY'
import numpy as np, time, os
from sklearn.datasets import make_classification
from sklearn.linear_model import SGDClassifier

print("Threads:", os.environ.get("OMP_NUM_THREADS"))

# Smaller + float32 = safe on 4GB nodes
X, y = make_classification(
    n_samples=800_000,
    n_features=60,
    n_informative=30,
    random_state=42
)

X = X.astype(np.float32, copy=False)

clf = SGDClassifier(loss="log_loss", max_iter=1, tol=None)

t0 = time.time()
for epoch in range(40):   # more epochs to keep runtime long
    clf.partial_fit(X, y, classes=np.array([0,1]))
    if epoch % 5 == 0:
        print("epoch", epoch, "elapsed", time.time()-t0)

print("done, total time:", time.time()-t0)
PY
