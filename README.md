# Parallel N Body Simulation  
Sequential C++, OpenMP, and CUDA Implementations

This project implements a simple N body simulation in three variants:

- Sequential C++ (baseline)
- OpenMP parallel CPU version
- CUDA GPU version

Bodies interact through Newtonian gravity in a 2D unit square with reflective boundaries. At each time step all pairwise forces are computed, velocities and positions are updated, and wall reflections are applied.

---

## Physical model

For bodies \(i\) and \(j\),

$$
\vec{F}_{ij} = G \frac{m_i m_j}{\left(r_{ij}^2 + \varepsilon\right)^{3/2}} \,\vec{r}_{ij}
$$

with

- \(G = 1.0\)
- Softening \(\varepsilon = 10^{-4}\)
- Time step \(\Delta t = 0.01\)
- Domain: 2D box \([0, 1] \times [0, 1]\) with reflective boundaries

Integration scheme:

$$
\begin{aligned}
\vec{v}_i^{\,t+\Delta t} &= \vec{v}_i^{\,t} + \vec{a}_i^{\,t}\,\Delta t, \\
\vec{x}_i^{\,t+\Delta t} &= \vec{x}_i^{\,t} + \vec{v}_i^{\,t+\Delta t}\,\Delta t
\end{aligned}
$$

If a position leaves the box, it is clamped back to the boundary and the corresponding velocity component is flipped.

---

## Implementations

### 1. Sequential C++ (`nbody_seq.cpp`)

- Data structure: `struct Body { double x, y, vx, vy, m; }`
- `O(N^2)` force computation using a nested loop over all pairs
- Two-phase update:
  - First loop over all pairs to compute forces
  - Second loop to update velocities and positions and apply reflections

### 2. OpenMP C++ (`nbody_omp.cpp`)

- Same physics and data layout as the sequential code
- Parallelization:
  - Outer loop over `i` in the force computation is parallelized with
    ```cpp
    #pragma omp parallel for schedule(static)
    ```
  - Each thread accumulates forces for its own body and writes to its own entries in `fx` and `fy`
- Integration loop is serial and relatively cheap
- Number of threads is controlled via a command line argument and `omp_set_num_threads`

### 3. CUDA (`nbody_cuda.cu`)

- Data stored as separate arrays for positions, velocities, and masses (`x`, `y`, `vx`, `vy`, `m`) for better memory access
- One CUDA thread per body
- Each kernel call:
  - Loads state for body `i`
  - Loops over all bodies in global memory to accumulate gravitational force
  - Updates velocity and position and applies boundary reflections
- Host code:
  - Allocates and initializes host arrays
  - Copies data to the GPU
  - Runs a time stepping loop that invokes the kernel per step
  - Copies final state back and writes to file
- Uses single precision (`float`) on the device

---

## Build instructions

You will need:

- A C++17 compatible compiler
- OpenMP support for the OpenMP version
- CUDA toolkit for the CUDA version
- (Optional) Python with Matplotlib for visualization

Example build commands:

```bash
# Sequential
g++ -O3 -std=c++17 nbody_seq.cpp -o nbody_seq

# OpenMP
g++ -O3 -std=c++17 -fopenmp nbody_omp.cpp -o nbody_omp

# CUDA
nvcc -O3 nbody_cuda.cu -o nbody_cuda
```

---

## Usage

All executables operate on randomly initialized bodies in a 2D box.

### Sequential

```bash
./nbody_seq N num_steps output_dir write_trajectories
```

Arguments:

- `N`  
  Number of bodies
- `num_steps`  
  Number of time steps
- `output_dir`  
  Directory to write output files (created if needed)
- `write_trajectories`  
  `1` to write all trajectories at every step  
  `0` to write only the final state

Output files:

- With trajectories:  
  `seq_N<N>_steps<steps>_traj.txt`
- Final state only:  
  `seq_N<N>_steps<steps>_final.txt`

Each trajectory line has:
```text
step i x y vx vy
```

Final state lines have:
```text
i x y vx vy
```

---

### OpenMP

```bash
./nbody_omp N num_steps output_dir num_threads
```

Arguments:

- `N`  
  Number of bodies
- `num_steps`  
  Number of time steps
- `output_dir`  
  Directory for output files
- `num_threads`  
  Number of OpenMP threads

Output file:

```text
omp_N<N>_steps<steps>_t<num_threads>.txt
```

Format:

```text
i x y vx vy
```

The program prints the measured runtime in seconds.

---

### CUDA

```bash
./nbody_cuda N num_steps output_dir
```

Arguments:

- `N`  
  Number of bodies
- `num_steps`  
  Number of time steps
- `output_dir`  
  Directory for output files

Output file:

```text
cuda_N<N>_steps<steps>.txt
```

Format:

```text
i x y vx vy
```

The program prints the measured runtime labeled `CUDA time`.

---

## Performance

All performance measurements in this project are for 1000 time steps, using:

- Sequential and OpenMP: double precision
- CUDA: single precision
- OpenMP: 8 threads for the table below

### Execution time vs N (1000 steps)

| N     | Sequential (s) | OpenMP 8 threads (s) | CUDA (s) |
|-------|----------------|----------------------|----------|
| 64    | 0.02365        | 0.01760              | 0.01145  |
| 512   | 1.25299        | 0.20392              | 0.03489  |
| 1024  | 4.99316        | 0.70256              | 0.06615  |
| 2048  | 19.93950       | 2.62902              | 0.12617  |
| 4096  | 79.74770       | 10.24720             | 0.22465  |

Observations:

- All implementations scale roughly as \(O(N^2)\), as expected from the all pairs force computation.
- OpenMP provides close to an order of magnitude speedup over the sequential baseline at the largest \(N\) in this table.
- CUDA achieves the best performance and the smallest constant factor.

### Performance s (placeholders)

You can generate and save performance s to the `s/` directory and they will automatically show up here:

- OpenMP scaling with threads:

  ```text
  outputs/plots/omp_scaling.png
  ```

  ![OpenMP scaling vs threads](outputs/plots/omp_scaling.png)

- Runtime vs number of bodies for all implementations:

  ```text
  outputs/plots/perf_vs_N.png
  ```

  ![Execution time vs N](outputs/plots/perf_vs_N.png)

### Larger CUDA runs

Additional CUDA only scaling runs:

| N       | Steps   | Time (s) | Notes                          |
|---------|---------|----------|--------------------------------|
| 10 000  | 1 000   | 0.573022 | exploratory scaling run        |
| 20 000  | 1 000   | 1.38926  | exploratory scaling run        |
| 20 000  | 2 000   | 2.77037  | step scaling test              |
| 30 000  | 2 000   | 4.50866  | larger scaling test            |
| 100 000 | 5 000   | 126.952  | initial large run              |
| 100 000 | 140 000 | 3575.43  | largest run under one hour     |

The last configuration represents the largest simulation completed in under one hour on the tested GPU:

- \(N = 100\,000\) bodies  
- 140 000 time steps  
- Runtime ~ 3575.43 seconds (about 59 minutes 35 seconds)  
- Total work: \(1.4 \times 10^{10}\) body steps

---

## Visualization

The sequential code can optionally write full trajectories for all bodies at each time step. A Python script (not included here) can be used to:

1. Read the trajectory file produced by `nbody_seq` with `write_trajectories = 1`
2. Use Matplotlib to:
   - Plot positions for each frame
   - Save frames to an animated GIF

Example data used for visualization:

- `N = 256` bodies
- `num_steps = 1000`
- Output GIF:  
  `outputs/vis/nbody_N256_steps1000.gif`

### Visualization placeholders

Animated GIF:

```text
outputs/vis/nbody_N256_steps1000.gif
```

![N body animation (GIF)](outputs/vis/nbody_N256_steps1000.gif)

---

## Repository layout

A typical layout for this project:

```text
.
├── nbody_seq.cpp        # Sequential C++ implementation
├── nbody_omp.cpp        # OpenMP C++ implementation
├── nbody_cuda.cu        # CUDA implementation
├── plots/               # Generated performance plots and snapshots
│   ├── omp_scaling.png
│   ├── perf_vs_N.png
│   └── nbody_N256_steps1000.png
├── outputs/             # Simulation outputs and visualization artifacts
│   └── vis/
│       └── nbody_N256_steps1000.gif
└── scripts/             # Optional Python scripts for plotting / animation
```

Feel free to adapt this structure to your own setup.

---

## Summary

- Sequential implementation serves as a simple, clear baseline.
- OpenMP parallelization leverages multi core CPUs to significantly reduce runtime for larger N.
- CUDA implementation delivers the best performance and enables very large N body simulations within practical runtimes.
