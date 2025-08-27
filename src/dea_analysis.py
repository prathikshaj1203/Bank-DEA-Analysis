import numpy as np
import pandas as pd
from pyomo.environ import *

def dea_ccr(X: pd.DataFrame, Y: pd.DataFrame):
    """CCR DEA model (constant returns to scale)."""
    n, m = X.shape
    _, s = Y.shape
    eff_scores = []

    for j in range(n):
        model = ConcreteModel()
        model.u = Var(range(s), domain=NonNegativeReals)
        model.v = Var(range(m), domain=NonNegativeReals)

        # Objective: Maximize efficiency of DMU j
        model.obj = Objective(
            expr=sum(model.u[r] * Y.iloc[j, r] for r in range(s)),
            sense=maximize
        )

        # Constraint: denominator = 1
        model.con1 = Constraint(expr=sum(model.v[i] * X.iloc[j, i] for i in range(m)) == 1)

        # Constraints for all DMUs
        model.cons = ConstraintList()
        for k in range(n):
            model.cons.add(
                sum(model.u[r] * Y.iloc[k, r] for r in range(s))
                - sum(model.v[i] * X.iloc[k, i] for i in range(m)) <= 0
            )

        SolverFactory("highs").solve(model, tee=False)

        eff = sum(model.u[r].value * Y.iloc[j, r] for r in range(s))
        eff_scores.append(eff)

    return np.array(eff_scores)

def dea_bcc(X: pd.DataFrame, Y: pd.DataFrame):
    """BCC DEA model (variable returns to scale)."""
    n, m = X.shape
    _, s = Y.shape
    eff_scores = []

    for j in range(n):
        model = ConcreteModel()
        model.u = Var(range(s), domain=NonNegativeReals)
        model.v = Var(range(m), domain=NonNegativeReals)
        model.lmbda = Var(range(n), domain=NonNegativeReals)

        # Objective
        model.obj = Objective(
            expr=sum(model.u[r] * Y.iloc[j, r] for r in range(s)),
            sense=maximize
        )

        model.con1 = Constraint(expr=sum(model.v[i] * X.iloc[j, i] for i in range(m)) == 1)

        model.cons = ConstraintList()
        for k in range(n):
            model.cons.add(
                sum(model.u[r] * Y.iloc[k, r] for r in range(s))
                - sum(model.v[i] * X.iloc[k, i] for i in range(m)) <= 0
            )

        # Convexity constraint
        model.conv = Constraint(expr=sum(model.lmbda[k] for k in range(n)) == 1)

        SolverFactory("highs").solve(model, tee=False)

        eff = sum(model.u[r].value * Y.iloc[j, r] for r in range(s))
        eff_scores.append(eff)

    return np.array(eff_scores)
