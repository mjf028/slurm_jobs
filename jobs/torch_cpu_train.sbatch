#!/bin/bash
#SBATCH --job-name=torch_cpu
#SBATCH --partition=fantacone
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=3000M
#SBATCH --time=00:20:00
#SBATCH --output=output/torch-%j.out

# ---- Conda init (non-interactive safe) ----
conda activate torch-cpu

# ---- Thread control (CRITICAL for metrics) ----
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

python - <<'PY'
import torch, time, os
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))

print("Host:", os.uname().nodename)
print("Threads:", torch.get_num_threads())

# ---- Dataset size tuned for 4GB nodes ----
N = 500_000     # safe
D = 100

X = torch.randn(N, D)
y = torch.randint(0, 2, (N,))

ds = TensorDataset(X, y)
dl = DataLoader(
    ds,
    batch_size=1024,
    shuffle=True,
    num_workers=0   # IMPORTANT for memory safety
)

model = nn.Sequential(
    nn.Linear(D, 512),
    nn.ReLU(),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Linear(256, 2)
)

loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(10):
    t0 = time.time()
    for xb, yb in dl:
        opt.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        opt.step()
    print(f"epoch {epoch} time {time.time()-t0:.1f}s")
PY