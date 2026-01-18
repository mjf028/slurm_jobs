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
from sklearn.linear_model import SGDClassifier

print("Threads:", os.environ.get("OMP_NUM_THREADS"))

X, y = make_classification(
    n_samples=1_500_000,
    n_features=60,
    n_informative=30,
    random_state=42
)

clf = SGDClassifier(loss="log_loss", max_iter=1, tol=None)

t0 = time.time()
# "epochs" loop to make it run longer and keep CPU hot
for epoch in range(30):
    clf.partial_fit(X, y, classes=np.array([0,1]))
    if epoch % 5 == 0:
        print("epoch", epoch, "elapsed", time.time()-t0)
print("done, total time:", time.time()-t0)

PY
