# plot_performance.py
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt

if len(sys.argv) < 3:
    print("Usage: python plot_performance.py timings.csv output_dir")
    sys.exit(1)

csv_file = sys.argv[1]
output_dir = sys.argv[2]
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(csv_file)

# 1. OpenMP scaling: time vs threads for different N
omp = df[df["version"] == "omp"]

if not omp.empty:
    Ns = sorted(omp["N"].unique())
    plt.figure()
    for N in Ns:
        sub = omp[omp["N"] == N]
        sub = sub.sort_values("threads")
        plt.plot(sub["threads"], sub["time_sec"], marker="o", label=f"N = {N}")
    plt.xlabel("Threads")
    plt.ylabel("Execution time (s)")
    plt.title("OpenMP scaling")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    out_path = os.path.join(output_dir, "omp_scaling.png")
    plt.savefig(out_path)
    print(f"Saved {out_path}")

# 2. Time vs N for seq, omp, cuda
plt.figure()
Ns = sorted(df["N"].unique())
for version in ["seq", "omp", "cuda"]:
    sub = df[df["version"] == version]
    if sub.empty:
        continue
    if version == "omp":
        rows = []
        for N in Ns:
            sN = sub[sub["N"] == N]
            if sN.empty:
                continue
            best = sN.loc[sN["threads"].idxmax()]
            rows.append(best)
        if not rows:
            continue
        sub = pd.DataFrame(rows)

    sub = sub.sort_values("N")
    plt.plot(sub["N"], sub["time_sec"], marker="o", label=version)

plt.xlabel("N (number of bodies)")
plt.ylabel("Execution time (s)")
plt.title("Performance vs problem size")
plt.legend()
plt.grid(True)
plt.tight_layout()
out_path2 = os.path.join(output_dir, "perf_vs_N.png")
plt.savefig(out_path2)
print(f"Saved {out_path2}")
