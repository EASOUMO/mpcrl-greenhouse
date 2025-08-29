import numpy as np
import matplotlib.pyplot as plt
import os

# Path to your numpy file
pkl_path = "/home/esoumo/Documents/PhD_research/mpcrl-greenhouse/data/disturbances.npy"

# Load data
data = np.load(pkl_path, allow_pickle=True)
print("Data shape:", data.shape)

num_features, num_steps = data.shape

# Plot all features in one figure with subplots
fig, axs = plt.subplots(num_features, 1, figsize=(15, 3*num_features), sharex=True)

for i in range(num_features):
    axs[i].plot(np.arange(num_steps), data[i], label=f'Feature {i+1}')
    axs[i].set_ylabel(f'Feature {i+1}')
    axs[i].legend()
    axs[i].grid(True)

axs[-1].set_xlabel('Timestep')
plt.suptitle("Disturbance Features over Time", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])

# Save the figure in the same directory as the data file
save_path = os.path.join(os.path.dirname(pkl_path), "disturbance_features_plot.png")
plt.savefig(save_path)
print(f"Figure saved to: {save_path}")
