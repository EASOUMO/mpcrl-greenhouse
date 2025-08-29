import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

# Paths to your pickle files
pkl_path1 = "/home/esoumo/Documents/PhD_research/mpcrl-greenhouse/sac_eval_2.pkl"
pkl_path2 = "/home/esoumo/Documents/PhD_research/mpcrl-greenhouse/td3_eval_2.pkl"
pkl_path3 = "/home/esoumo/Documents/PhD_research/mpcrl-greenhouse/ddpg_eval_2.pkl"

# --- Load SAC data ---
with open(pkl_path1, "rb") as f:
    data = pickle.load(f)
X1 = data['X']  # (1000, 3841, 4)
U1 = data['U']  # (1000, 3840, 3)
d1 = data['d']  # (1050, 3840, 4)
R1 = data['R']  # (1000, 3840)

# --- Load TD3 data ---
with open(pkl_path2, "rb") as f:
    data1 = pickle.load(f)
X2 = data1['X']  # (1000, 3841, 4)
U2 = data1['U']  # (1000, 3840, 3)
d2 = data1['d']  # (1050, 3840, 4)
R2 = data1['R']  # (1000, 3840)

# --- Load DDPG data ---
with open(pkl_path3, "rb") as f:
    data2 = pickle.load(f)
X3 = data2['X']  # (1000, 3841, 4)
U3 = data2['U']  # (1000, 3840, 3)
d3 = data2['d']  # (1050, 3840, 4)
R3 = data2['R']  # (1000, 3840)

# --- Average over episodes ---
X_mean1, U_mean1, d_mean1, R_mean1 = X1.mean(axis=0), U1.mean(axis=0), d1.mean(axis=0), R1.mean(axis=0)
X_mean2, U_mean2, d_mean2, R_mean2 = X2.mean(axis=0), U2.mean(axis=0), d2.mean(axis=0), R2.mean(axis=0)
X_mean3, U_mean3, d_mean3, R_mean3 = X3.mean(axis=0), U3.mean(axis=0), d3.mean(axis=0), R3.mean(axis=0)

# --- Plotting ---
fig, axs = plt.subplots(12, 1, figsize=(12, 30))

# Plots 1–4: X
for i in range(4):
    axs[i].plot(X_mean1[:, i], label="SAC")
    axs[i].plot(X_mean2[:, i], label="TD3")
    axs[i].plot(X_mean3[:, i], label="DDPG")
    axs[i].set_title(f'X variable {i+1}')
    axs[i].set_xlabel('Timestep')
    axs[i].set_ylabel(f'X{i+1}')
    axs[i].legend()

# Plots 5–7: U
for i in range(3):
    axs[4+i].plot(U_mean1[:, i], label="SAC")
    axs[4+i].plot(U_mean2[:, i], label="TD3")
    axs[4+i].plot(U_mean3[:, i], label="DDPG")
    axs[4+i].set_title(f'U variable {i+1}')
    axs[4+i].set_xlabel('Timestep')
    axs[4+i].set_ylabel(f'U{i+1}')
    axs[4+i].legend()

# Plots 8–11: d
for i in range(4):
    axs[7+i].plot(d_mean1[:, i], label="SAC")
    axs[7+i].plot(d_mean2[:, i], label="TD3")
    axs[7+i].plot(d_mean3[:, i], label="DDPG")
    axs[7+i].set_title(f'd variable {i+1}')
    axs[7+i].set_xlabel('Timestep')
    axs[7+i].set_ylabel(f'd{i+1}')
    axs[7+i].legend()

# Plot 12: R
axs[11].plot(R_mean1, label="SAC")
axs[11].plot(R_mean2, label="TD3")
axs[11].plot(R_mean3, label="DDPG")
axs[11].set_title('Reward R')
axs[11].set_xlabel('Timestep')
axs[11].set_ylabel('R')
axs[11].legend()

plt.tight_layout()

# Save figure to same directory as pickle file
save_path = os.path.join(os.path.dirname(pkl_path1), "sac_td3_ddpg_eval_2_plots.png")
plt.savefig(save_path, dpi=300)
plt.close()

print(f"Plot saved to {save_path}")
