# visualize.py
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

if len(sys.argv) < 5:
    print("Usage: python visualize.py traj_file N num_steps output_dir")
    sys.exit(1)

traj_file = sys.argv[1]
N = int(sys.argv[2])
num_steps = int(sys.argv[3])
output_dir = sys.argv[4]

os.makedirs(output_dir, exist_ok=True)

data = np.loadtxt(traj_file)
steps = data[:, 0].astype(int)
ids = data[:, 1].astype(int)
xs = data[:, 2]
ys = data[:, 3]

if xs.size != N * num_steps:
    print("Warning: data size does not match N * num_steps")
    print("Found", xs.size, "rows, expected", N * num_steps)

x_frames = xs.reshape((num_steps, N))
y_frames = ys.reshape((num_steps, N))

fig, ax = plt.subplots()
sc = ax.scatter(x_frames[0], y_frames[0], s=5)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect("equal", "box")
title = ax.text(
    0.02, 0.95,
    f"N = {N}, steps = {num_steps}",
    transform=ax.transAxes,
    fontsize=10,
    verticalalignment="top"
)

def update(frame):
    sc.set_offsets(np.c_[x_frames[frame], y_frames[frame]])
    title.set_text(f"Step {frame} / {num_steps}, N = {N}")
    return sc, title

ani = FuncAnimation(fig, update, frames=num_steps, interval=30, blit=True)

gif_path = os.path.join(output_dir, f"nbody_N{N}_steps{num_steps}.gif")
ani.save(gif_path, writer="pillow", fps=30)
print(f"Visualization saved to {gif_path}")
