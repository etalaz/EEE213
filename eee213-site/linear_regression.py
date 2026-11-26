"""Linear regression on the combined cycle power plant data (Week 10 method 1).

Run from this folder:  python linear_regression.py
Requires ccpp.csv in the same folder, plus pandas and scikit-learn.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

df = pd.read_csv(Path(__file__).with_name("ccpp.csv"))
features = ["AT", "V", "AP", "RH"]
train, test = df.iloc[:7654], df.iloc[-1914:]

print("rows:", len(df), "missing:", int(df.isna().sum().sum()),
      "exact duplicate rows:", int(df.duplicated().sum()),
      "training rows:", len(train), "test rows:", len(test))

baseline = np.full(len(test), train["PE"].mean())
print(f"baseline (training mean {train['PE'].mean():.2f} MW): "
      f"test MAE {mean_absolute_error(test['PE'], baseline):.2f} MW")

for name, cols in (("AT only", ["AT"]), ("four features", features)):
    model = LinearRegression().fit(train[cols], train["PE"])
    train_mae = mean_absolute_error(train["PE"], model.predict(train[cols]))
    test_mae = mean_absolute_error(test["PE"], model.predict(test[cols]))
    coefs = ", ".join(f"{c} {w:+.3f}" for c, w in zip(cols, model.coef_))
    print(f"linear, {name}: weights (MW per unit) {coefs}, intercept {model.intercept_:.2f} MW, "
          f"train MAE {train_mae:.2f} MW, test MAE {test_mae:.2f} MW")
