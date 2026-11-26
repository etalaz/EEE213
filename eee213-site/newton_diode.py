"""Diode-resistor operating point solved with Newton-Raphson.

Circuit: source Vs in series with resistor R and a diode.
Unknown: diode voltage Vd (V). Run:  python newton_diode.py
"""
import numpy as np
from scipy.optimize import brentq
from scipy.special import lambertw

Vs = 5.0        # source voltage, V
R = 1000.0      # resistance, ohm
Is = 1e-12      # diode saturation current, A
n = 1.0         # ideality factor, dimensionless
VT = 0.02585    # thermal voltage at about 300 K, V
nVT = n * VT


def f(vd):
    """Current balance, A: diode current minus resistor current."""
    return Is * (np.exp(vd / nVT) - 1.0) - (Vs - vd) / R


def df(vd):
    """Derivative of f with respect to Vd, A/V (derived by hand)."""
    return Is / nVT * np.exp(vd / nVT) + 1.0 / R


vd = 0.6        # starting guess, V
print(f"{'k':>2} {'Vd (V)':>12} {'f(Vd) (A)':>14} {'step (V)':>14}")
print(f"{0:>2} {vd:12.9f} {f(vd):14.6e} {'-':>14}")
for k in range(1, 50):
    step = -f(vd) / df(vd)
    vd += step
    print(f"{k:>2} {vd:12.9f} {f(vd):14.6e} {step:14.6e}")
    if abs(step) < 1e-12:
        break

i_newton = (Vs - vd) / R
print(f"\nNewton-Raphson: Vd = {vd:.6f} V, Id = {i_newton * 1e3:.6f} mA, iterations = {k} (count)")

# Independent check 1: closed form with the Lambert W function
w = lambertw(R * Is / nVT * np.exp((Vs + R * Is) / nVT)).real
i_lambert = nVT / R * w - Is
print(f"Lambert W:      Vd = {Vs - R * i_lambert:.6f} V, Id = {i_lambert * 1e3:.6f} mA")

# Independent check 2: bracketing root finder, no derivative used
vd_brent = brentq(f, 0.0, Vs, xtol=1e-14)
print(f"Brent:          Vd = {vd_brent:.6f} V, Id = {(Vs - vd_brent) / R * 1e3:.6f} mA")
print(f"Difference Newton - Lambert W: {abs(vd - (Vs - R * i_lambert)):.2e} V")
