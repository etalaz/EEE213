"""Illustration for the k-NN presentation: the k nearest training hours of one test hour.

Run from this folder:  python plot_neighbours.py
Shown in two features (AT and V) so it fits on a plane; the model itself uses all four.
Writes knn_neighbours.png.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

folder = Path(__file__).parent
df = pd.read_csv(folder / "ccpp.csv")
train, test = df.iloc[:7654], df.iloc[-1914:]
cols = ["AT", "V"]

scaler = StandardScaler().fit(train[cols])
Z = scaler.transform(train[cols])
query = scaler.transform(test[cols].iloc[[500]])[0]
distance = np.linalg.norm(Z - query, axis=1)
nearest = np.argsort(distance)[:3]
radius = distance[nearest].max()

fig, ax = plt.subplots(figsize=(4.4, 3.6), dpi=200)
points = ax.scatter(Z[:, 0], Z[:, 1], c=train["PE"], cmap="viridis_r", s=4, alpha=0.6, zorder=1)
ax.add_patch(plt.Circle(query, radius, fill=False, color="#d9480f", lw=1.6, zorder=3))
ax.scatter(Z[nearest, 0], Z[nearest, 1], s=40, color="#d9480f", edgecolor="white", zorder=4, label="3 nearest hours")
ax.scatter(*query, marker="*", s=170, color="#14213d", edgecolor="white", zorder=5, label="new hour")
ax.set_xlim(query[0] - 0.9, query[0] + 0.9)
ax.set_ylim(query[1] - 0.9, query[1] + 0.9)
ax.set_aspect("equal")
ax.set_xlabel("AT, scaled (dimensionless)", fontsize=8)
ax.set_ylabel("V, scaled (dimensionless)", fontsize=8)
ax.tick_params(labelsize=7)
cbar = fig.colorbar(points, ax=ax, fraction=0.046, pad=0.03)
cbar.set_label("PE (MW)", fontsize=8)
cbar.ax.tick_params(labelsize=7)
ax.legend(frameon=True, fontsize=7, loc="upper right")
fig.tight_layout()
fig.savefig(folder / "knn_neighbours.png")
print(f"saved knn_neighbours.png; neighbour PE: {train['PE'].iloc[nearest].round(1).tolist()} MW, "
      f"mean {train['PE'].iloc[nearest].mean():.1f} MW, true {test['PE'].iloc[500]:.1f} MW")
