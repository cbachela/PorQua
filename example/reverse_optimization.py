############################################################################
### REVERSE OPTIMIZATION
############################################################################

# --------------------------------------------------------------------------
# Cyril Bachelard
# This version:     05.05.2025
# First version:    05.05.2025
# --------------------------------------------------------------------------





# Standard library imports
from typing import Union
    
# Third party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB, Model, Env

# Add the project root directory to Python path
import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(project_root, 'src')
sys.path.append(project_root)
sys.path.append(src_path)

# Local application imports
from backtest import (
    BacktestService,
)
from optimization import MeanVariance, Objective
from covariance import Covariance
from mean_estimation import MeanEstimator
from constraints import Constraints





# --------------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------------

def load_data_msci(path: str = None, n: int = 24) -> dict[str, pd.DataFrame]:
    '''Loads MSCI daily returns data from 1999-01-01 to 2023-04-18'''

    path = os.path.join(os.getcwd(), f'data{os.sep}') if path is None else path

    # Load msci country index return series
    df = pd.read_csv(os.path.join(path, 'msci_country_indices.csv'),
                        index_col=0,
                        header=0,
                        parse_dates=True,
                        date_format='%d-%m-%Y')
    series_id = df.columns[0:n]
    X = df[series_id]

    # Load msci world index return series
    y = pd.read_csv(f'{path}NDDLWI.csv',
                    index_col=0,
                    header=0,
                    parse_dates=True,
                    date_format='%d-%m-%Y')

    return {'return_series': X, 'bm_series': y}



def infer_mean_vector_gurobi(
    covmat: Union[np.ndarray, pd.DataFrame],
    w_optimal: pd.Series,
    constraints: Constraints,
) -> pd.Series:
    """
    Infers the mean return vector (mu) that would lead to w_optimal being 
    the solution of mean-variance optimization with constraints using Gurobi.

    Parameters:
        Sigma (pd.DataFrame, numpy.ndarray): Covariance matrix (n x n).
        w_optimal (numpy.ndarray): Optimal weights (n x 1).
        constraints (Constraints): Constraints object from which to extract
            the equality and inequality constraints. I.e.,
            - A (numpy.ndarray): Equality constraint matrix (m x n)
            - b (numpy.ndarray): Equality constraint vector (m x 1)
            - G (numpy.ndarray): Inequality constraint matrix (p x n)
            - h (numpy.ndarray): Inequality constraint vector (p x 1).
        Notice that lower and upper bounds must be included in the inequality
        constraints (G @ w <= h).

    Returns:
        numpy.ndarray: Implied mean return vector (n x 1)
    """

    GhAb = constraints.to_GhAb(lbub_to_G=True)
    A = GhAb["A"]
    # Ensure that A is a 2D array
    if len(A.shape) == 1:
        A = A.reshape(1, -1)
    G = GhAb["G"]
    h = GhAb["h"]
    n = len(w_optimal)

    # Create Gurobi model
    model = Model("Infer_Mean_Vector")
    model.setParam('OutputFlag', 0)  # Suppress Gurobi output

    # Define variables
    mu = model.addMVar(n, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="mu")
    nu = model.addMVar(A.shape[0], lb=-GRB.INFINITY, ub=GRB.INFINITY, name="nu")  # Equality constraint multipliers
    eta = model.addMVar(G.shape[0], lb=0, name="eta")  # Inequality constraint multipliers
    # lambda_var = model.addVar(lb=0, ub=GRB.INFINITY, name="lambda")  # Risk aversion parameter

    # Define KKT conditions
    if hasattr(covmat, 'to_numpy'):
        covmat = covmat.to_numpy()
    # model.addConstr((covmat @ w_optimal).to_numpy() - lambda_var * mu - A.T @ nu - G.T @ eta == 0, "KKT")
    model.addConstr(covmat @ w_optimal + A.T @ nu + G.T @ eta + mu == 0, "KKT")
    model.addConstr(eta * (G @ w_optimal - h) == np.zeros(G.shape[0]), "Complementary Slackness")

    # Define objective: Minimize deviation of mu (regularization)
    model.setObjective(mu @ mu, GRB.MINIMIZE)  # Minimize ||mu||_2^2

    # Solve the optimization problem
    model.optimize()

    # Extract solution
    if model.status == GRB.OPTIMAL:
        mu_inferred = pd.Series(mu.X, constraints.selection)
        return mu_inferred * (-1)
    else:
        print("Optimization problem not solved to optimality.")
        return None




# --------------------------------------------------------------------------
# Load data (returns series)
# --------------------------------------------------------------------------

data = load_data_msci(path = '../data/', n = 10)
data['return_series'].head()



# --------------------------------------------------------------------------
# Estimate the covariance matrix
# --------------------------------------------------------------------------

covariance = Covariance(method='pearson')


# --------------------------------------------------------------------------
# Initialize the mean estimator
# --------------------------------------------------------------------------

mean_estimator = MeanEstimator(method='geometric')



# --------------------------------------------------------------------------
# Prepare the constraints object
# --------------------------------------------------------------------------

# Instantiate the class
constraints = Constraints(selection=data['return_series'].columns.tolist())

# Add budget constraint
constraints.add_budget(rhs=1, sense='=')

# Add box constraints (i.e., lower and upper bounds)
constraints.add_box(lower=0, upper=0.2)

# Add linear constraints
G = pd.DataFrame(
    np.zeros((2, len(constraints.selection))),
    columns=constraints.selection
)
G.iloc[0, 0:5] = 1
G.iloc[1, 6:10] = 1
h = pd.Series([0.5, 0.5])
constraints.add_linear(
    Amat=G,
    sense='<=',
    rhs=h
)

# Inspect the constraints
constraints.budget
constraints.box
constraints.linear
constraints.to_GhAb()
constraints.to_GhAb(lbub_to_G=True)




# --------------------------------------------------------------------------
# Run a mean-variance optimization
# --------------------------------------------------------------------------

solver_name = 'cvxopt'
# solver_name = 'gurobi'


mv = MeanVariance(
    covariance=covariance,
    mean_estimator=mean_estimator,
    constraints=constraints,
    solver_name=solver_name,
    risk_aversion=1,
)
mv.set_objective(optimization_data=data)
mv.solve()



# Extract the optimal weights and covariance matrix
w_star = pd.Series(mv.results['weights'])
covmat = mv.objective['P'].copy() / 2
mu = pd.Series(mv.objective['q'], constraints.selection) * (-1)




# --------------------------------------------------------------------------
# Reverse optimization - Get Implied Expected Returns
# --------------------------------------------------------------------------

# Implied expected return of benchmark
mu_implied = infer_mean_vector_gurobi(
    covmat=covmat,
    w_optimal=w_star,
    constraints=constraints,
)
mu_implied



# Analytical solution
mu_implied_analytic = pd.Series(covmat @ w_star, constraints.selection)



# Compare the estimated mean with the implied mean
Mu = pd.DataFrame(
    [mu, mu_implied, mu_implied_analytic],
    index=["mu", "mu_implied", "mu_implied_analytic"]
).T
Mu.plot(kind="bar", figsize=(10, 5))




# Assert that mean-variance optimization with mu_implied gives the same result

# Analytical solution
w_star_analytic = pd.Series(
    np.linalg.inv(covmat) @ mu_implied_analytic,
    index=constraints.selection
)


# Using the solver
mv2 = MeanVariance(
    constraints=constraints,
    solver_name=solver_name,
)
mv2.objective = Objective(
    q=pd.Series(mu_implied, constraints.selection) * (-1),
    # q=mu_implied_analytic * (-1),
    P=covmat * 2,
)
mv2.solve()
w_star_2 = pd.Series(mv2.results["weights"])

W = pd.DataFrame([w_star, w_star_2, w_star_analytic], index=["w_star", "w_star_2", "w_start_analytic"]).T
W.plot(kind="bar", figsize=(10, 5))








