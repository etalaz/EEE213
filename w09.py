"""EEE213 Week 09 lab: machine learning on the combined cycle power plant data.

Run from this folder:  python w09.py
Requires ccpp.csv in the same folder, plus pandas and scikit-learn.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

df = pd.read_csv(Path(__file__).with_name("ccpp.csv"))
features = ["AT", "V", "AP", "RH"]

# Section 3: size, completeness, ranges, duplicates (all rows)
print("3.1 rows, columns:", df.shape, "missing:", int(df.isna().sum().sum()),
      "exact duplicate rows:", int(df.duplicated().sum()))
print("3.2 min / max per column:")
print(df.agg(["min", "max"]).round(2).T)

# Split once, before any training: first 7,654 rows train, last 1,914 rows test
train, test = df.iloc[:7654], df.iloc[-1914:]
print("3.3 training rows:", len(train), "test rows:", len(test))

# Section 4 (all rows)
print("4.1 mean / median / std:")
print(df[["AT", "PE"]].agg(["mean", "median", "std"]).round(2).T)
print("4.2 correlation with PE:")
print(df.corr()["PE"].drop("PE").round(3))

# Section 5 (fit on training rows, report on test rows)
mean_pe = train["PE"].mean()
baseline = np.full(len(test), mean_pe)
print(f"5.1 training mean of PE: {mean_pe:.2f} MW, "
      f"baseline test MAE: {mean_absolute_error(test['PE'], baseline):.2f} MW")

for feature in ("AT", "V"):
    line = LinearRegression().fit(train[[feature]], train["PE"])
    mae = mean_absolute_error(test["PE"], line.predict(test[[feature]]))
    print(f"5.2 {feature}: slope a = {line.coef_[0]:.3f} MW per unit, "
          f"intercept b = {line.intercept_:.2f} MW, test MAE = {mae:.2f} MW")

# 5.5: one test row checked by hand (file line 8000; the header is line 1)
line_number = 8000
row = df.iloc[line_number - 2]
a, b = -1.168, 517.78
predicted = a * row["V"] + b
print(f"5.5 line {line_number}: V = {row['V']}, predicted = {predicted:.2f} MW, "
      f"true = {row['PE']} MW, absolute error = {abs(row['PE'] - predicted):.2f} MW")
