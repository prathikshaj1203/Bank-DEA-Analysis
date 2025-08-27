import numpy as np
import pandas as pd
from scipy.optimize import linprog

def dea_ccr(X: pd.DataFrame, Y: pd.DataFrame):
    """
    CCR DEA model (constant returns to scale) using SciPy linprog.
    Input-oriented.
    """
    n, m = X.shape
    _, s = Y.shape
    eff_scores = []

    for j in range(n):
        x0, y0 = X.iloc[j].values, Y.iloc[j].values

        # Variables: θ + λ1..λn
        c = np.zeros(n + 1)
        c[0] = 1  # minimize θ

        A = []
        b = []

        # Input constraints: Σ λ x ≤ θ * x0
        for i in range(m):
            row = np.zeros(n + 1)
            row[0] = -x0[i]         # -θ * x0
            row[1:] = X.iloc[:, i]  # + Σ λ x
            A.append(row)
            b.append(0)

        # Output constraints: Σ λ y ≥ y0
        for r in range(s):
            row = np.zeros(n + 1)
            row[1:] = -Y.iloc[:, r]  # -Σ λ y
            A.append(row)
            b.append(-y0[r])

        bounds = [(0, None)] * (n + 1)  # θ ≥ 0, λ ≥ 0

        res = linprog(c, A_ub=np.array(A), b_ub=np.array(b),
                      bounds=bounds, method="highs")

        eff_scores.append(res.x[0] if res.success else np.nan)

    return np.array(eff_scores)


def dea_bcc(X: pd.DataFrame, Y: pd.DataFrame):
    """
    BCC DEA model (variable returns to scale) using SciPy linprog.
    Input-oriented.
    """
    n, m = X.shape
    _, s = Y.shape
    eff_scores = []

    for j in range(n):
        x0, y0 = X.iloc[j].values, Y.iloc[j].values

        # Variables: θ + λ1..λn
        c = np.zeros(n + 1)
        c[0] = 1

        A = []
        b = []

        # Input constraints
        for i in range(m):
            row = np.zeros(n + 1)
            row[0] = -x0[i]
            row[1:] = X.iloc[:, i]
            A.append(row)
            b.append(0)

        # Output constraints
        for r in range(s):
            row = np.zeros(n + 1)
            row[1:] = -Y.iloc[:, r]
            A.append(row)
            b.append(-y0[r])

        # Convexity constraint: Σ λ = 1
        Aeq = np.zeros((1, n + 1))
        Aeq[0, 1:] = 1
        beq = [1]

        bounds = [(0, None)] * (n + 1)

        res = linprog(c, A_ub=np.array(A), b_ub=np.array(b),
                      A_eq=Aeq, b_eq=beq,
                      bounds=bounds, method="highs")

        eff_scores.append(res.x[0] if res.success else np.nan)

    return np.array(eff_scores)
