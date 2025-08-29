# -*- coding: utf-8 -*-
"""

"""
import numpy as np
import matplotlib.pyplot as plt

# Load data
gts = np.load('gt.npy')
preds = np.load('pred.npy')

gts=gts[800:1000,:]*0.01
preds=preds[800:1000,:]*0.01

# Create figure
plt.figure(figsize=(8, 6))

# Ground truth trajectory (blue dots)
plt.scatter(gts[:, 0], gts[:, 1], label="Ground Truth", color="blue", marker='o', s=10)

# Predicted trajectory (red dots)
plt.scatter(preds[:, 0], preds[:, 1], label="Predicted", color="red", marker='x', s=10)

# Labels and title
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
#plt.title("Predicted vs Ground Truth Trajectory")

# Show legend
plt.legend()
plt.savefig('traj.png', dpi=300)
# Show plot
plt.show()
