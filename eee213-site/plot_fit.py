"""Figure for the Week 10 presentation: least-squares line and the errors it minimises.

Run from this folder:  python plot_fit.py
Requires ccpp.csv in the same folder, plus pandas, scikit-learn and matplotlib.
Writes fit_errors.png.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

folder = Path(__file__).parent
df = pd.read_csv(folder / "ccpp.csv")
train = df.iloc[:7654]

model = LinearRegression().fit(train[["AT"]], train["PE"])
a, b = model.coef_[0], model.intercept_

# Twelve training hours, evenly spaced in the file, to show the errors clearly
shown = train.iloc[::638].head(12)
predicted = model.predict(shown[["AT"]])

fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=200)
ax.scatter(train["AT"], train["PE"], s=4, color="#c3cad3", label="7,654 training hours", zorder=1)

xs = [train["AT"].min(), train["AT"].max()]
ax.plot(xs, [a * x + b for x in xs], color="#c62828", lw=2,
        label=f"line: PE = {a:.3f} AT + {b:.2f}", zorder=3)

for x, y_true, y_pred in zip(shown["AT"], shown["PE"], predicted):
    ax.plot([x, x], [y_true, y_pred], color="#1f2328", ls="--", lw=1, zorder=4)
ax.scatter(shown["AT"], shown["PE"], s=36, color="#1f2328", zorder=5,
           label="12 hours with their errors (dashed)")

ax.set_xlabel("Ambient temperature AT (°C)")
ax.set_ylabel("Net electrical output PE (MW)")
ax.set_title("Least squares: the line that minimises the sum of squared errors")
ax.grid(alpha=0.25)
ax.legend(frameon=False, loc="upper right", fontsize=8)
fig.tight_layout()
fig.savefig(folder / "fit_errors.png")
print(f"saved fit_errors.png: slope {a:.3f} MW/°C, intercept {b:.2f} MW")
