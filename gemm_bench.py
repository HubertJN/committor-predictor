import time
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

# Simple GEMM benchmark
n = 8192
a = torch.randn(n, n, device=device)
b = torch.randn(n, n, device=device)

# Warmup
for _ in range(10):
    c = a @ b
torch.cuda.synchronize() if device == "cuda" else None

t0 = time.time()
iters = 20
for _ in range(iters):
    c = a @ b
torch.cuda.synchronize() if device == "cuda" else None
t1 = time.time()

secs = (t1 - t0) / iters
print(f"device={device} n={n} avg_time_s={secs:.6f}")
