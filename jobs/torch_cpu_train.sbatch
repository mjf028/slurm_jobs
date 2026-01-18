#!/bin/bash
#SBATCH --job-name=torch_cpu
#SBATCH --partition=fantacone
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --output=output/torch-%j.out

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

python3 - <<'PY'
import torch, time
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

torch.set_num_threads(4)

N = 1_000_000
X = torch.randn(N, 100)
y = torch.randint(0, 2, (N,))

ds = TensorDataset(X, y)
dl = DataLoader(ds, batch_size=2048, shuffle=True)

model = nn.Sequential(
    nn.Linear(100, 512),
    nn.ReLU(),
    nn.Linear(512, 2)
)

loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(8):
    t0 = time.time()
    for xb, yb in dl:
        opt.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        opt.step()
    print(f"epoch {epoch} time {time.time()-t0:.1f}s")
PY
