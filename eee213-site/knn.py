"""k-nearest neighbours on the combined cycle power plant data (presentation).

Run from this folder:  python knn.py
Requires ccpp.csv in the same folder, plus pandas, scikit-learn and matplotlib.
Writes knn_cv_curve.png and knn_comparison.png.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

folder = Path(__file__).parent
df = pd.read_csv(folder / "ccpp.csv")
features = ["AT", "V", "AP", "RH"]
train, test = df.iloc[:7654], df.iloc[-1914:]
X_train, y_train = train[features], train["PE"]
X_test, y_test = test[features], test["PE"]


def knn(k):
    # The scaler is inside the pipeline, so in cross-validation it only sees the fitting folds
    return make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=k))


# Choose k by 5-fold cross-validation inside the training rows only
ks = [1, 2, 3, 5, 7, 10, 15, 20, 30, 50]
cv_mae = []
for k in ks:
    score = -cross_val_score(knn(k), X_train, y_train, cv=KFold(5),
                             scoring="neg_mean_absolute_error").mean()
    cv_mae.append(score)
    print(f"k = {k:>2}: cross-validation MAE {score:.2f} MW")
best_k = ks[int(np.argmin(cv_mae))]
print(f"chosen k = {best_k} (lowest cross-validation MAE)")

# One evaluation on the test rows
model = knn(best_k).fit(X_train, y_train)
train_mae = mean_absolute_error(y_train, model.predict(X_train))
test_mae = mean_absolute_error(y_test, model.predict(X_test))
print(f"k-NN (k = {best_k}): train MAE {train_mae:.2f} MW, test MAE {test_mae:.2f} MW")

# Reference models on the same split
baseline_mae = mean_absolute_error(y_test, np.full(len(y_test), y_train.mean()))
line_at = LinearRegression().fit(train[["AT"]], y_train)
line_at_mae = mean_absolute_error(y_test, line_at.predict(test[["AT"]]))
linear4 = LinearRegression().fit(X_train, y_train)
linear4_mae = mean_absolute_error(y_test, linear4.predict(X_test))
print(f"baseline {baseline_mae:.2f}, line from AT {line_at_mae:.2f}, "
      f"linear four features {linear4_mae:.2f}, k-NN {test_mae:.2f} (test MAE, MW)")

# Figure 1: cross-validation error against k
fig, ax = plt.subplots(figsize=(7, 4.4), dpi=200)
ax.plot(ks, cv_mae, marker="o", color="#0b5cad", lw=2)
ax.scatter([best_k], [min(cv_mae)], s=110, color="#c62828", zorder=5,
           label=f"chosen k = {best_k}: {min(cv_mae):.2f} MW")
ax.set_xscale("log")
ax.set_xticks(ks)
ax.set_xticklabels([str(k) for k in ks])
ax.set_xlabel("Number of neighbours k (count)")
ax.set_ylabel("Cross-validation MAE (MW)")
ax.set_title("Choosing k: 5-fold cross-validation on the training rows")
ax.annotate("small k: flexible,\nfollows the noise", xy=(1, cv_mae[0]), xytext=(1.1, cv_mae[0] + 0.12), fontsize=8)
ax.annotate("large k: rigid,\ntoo smooth", xy=(50, cv_mae[-1]), xytext=(18, cv_mae[-1] - 0.05), fontsize=8)
ax.grid(alpha=0.25)
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(folder / "knn_cv_curve.png")

# Figure 2: test error of all models
names = ["Baseline\n(training mean)", "Line from AT", "Linear,\nfour features", f"k-NN\n(k = {best_k})"]
values = [baseline_mae, line_at_mae, linear4_mae, test_mae]
fig, ax = plt.subplots(figsize=(7, 4.4), dpi=200)
bars = ax.bar(names, values, color=["#9aa4b2", "#6e8fb8", "#4a74a8", "#c62828"])
for bar, value in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width() / 2, value + 0.25, f"{value:.2f}", ha="center", fontsize=10)
ax.set_ylabel("Test MAE (MW), 1,914 test rows")
ax.set_title("Same split for every model: shorter is better")
ax.set_ylim(0, max(values) * 1.15)
ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
fig.savefig(folder / "knn_comparison.png")
print("saved knn_cv_curve.png and knn_comparison.png")
