############################################################################
### SOFT CONSTRAINTS
############################################################################

# --------------------------------------------------------------------------
# Cyril Bachelard
# This version:     05.05.2025
# First version:    05.05.2025
# --------------------------------------------------------------------------



    
# Third party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB, Model

# Add the project root directory to Python path
import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(project_root, 'src')
sys.path.append(project_root)
sys.path.append(src_path)

# Local application imports
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



def solve_gurobi(optimization) -> bool:

    optimization.model.optimize()
    status = optimization.model.status
    solved = True if status == 2 else False
    if status == 13:
        if optimization.params.get('allow_suboptimal'):
            solved = True
    ids = optimization.constraints.selection
    N = len(ids)
    if solved:
        weights = {
            ids[i]: optimization.model.x[i]
            for i in range(len(optimization.model.x[0:N]))
        }
        obj_val = optimization.model.objVal
    else:
        weights = {key: np.nan for key in ids}
        obj_val = np.nan

    optimization.results = {
        'weights': weights,
        'objective': obj_val,
        'status': status
    }
    optimization.model.dispose()
    return True


def model_gurobi(optimization) -> None:

    P = (
        optimization.objective['P'].to_numpy()
        if hasattr(optimization.objective['P'], "to_numpy")
        else optimization.objective['P']
    )
    q = (
        optimization.objective['q'].to_numpy()
        if hasattr(optimization.objective['q'], "to_numpy")
        else optimization.objective['q']
    )
    GhAb = optimization.constraints.to_GhAb()
    N = len(optimization.constraints.selection)
    transaction_cost = optimization.params.get('transaction_cost')

    # Initialize Gurobi model
    optimization.model = Model('portfolio')
    optimization.model.Params.LogToConsole = 0

    # Prepare decision variable 'x' as matrix variable including
    # lower and upper bounds and variable type
    lb = optimization.constraints.box['lower'].to_numpy()
    ub = optimization.constraints.box['upper'].to_numpy()
    v_type = GRB.CONTINUOUS
    x = optimization.model.addMVar(N, lb = lb, ub = ub, vtype = v_type, name = 'x')

    # Objective function
    if transaction_cost is not None:
        # Auxiliary variables to deal with the abs() function
        x_init = pd.Series(optimization.params.get('x_init')).to_numpy()
        aux_turnover = optimization.model.addMVar(N, lb = np.zeros(N), name = 'aux_turnover')
        optimization.model.addConstr(x - aux_turnover <= x_init, name = 'aux_turnover_leq')
        optimization.model.addConstr(x + aux_turnover >= x_init, name = 'aux_turnover_geq')
        obj_fun = q.T @ x + 0.5 * (x @ P @ x) + transaction_cost * aux_turnover.sum()
    else:
        obj_fun = q.T @ x + 0.5 * (x @ P @ x)

    optimization.model.setObjective(obj_fun, GRB.MINIMIZE)

    # Add linear inequality constraints
    if GhAb['G'] is not None:
        optimization.model.addConstr(GhAb['G'] @ x <= GhAb['h'], 'Gh')

    # Add linear equality constraints
    if GhAb['A'] is not None:
        optimization.model.addConstr(GhAb['A'] @ x == GhAb['b'], 'Ab')

    # # Add quadratic inequality constraints
    # quadcon = optimization.constraints.quadratic
    # if quadcon:
    #     for key, value in quadcon.items():
    #         if isinstance(value, dict):
    #             optimization.model.addConstr(value['q'].T @ x + (x @ value['Qc'] @ x) <= value['rhs'], key)

    # Turnover constraint
    if 'turnover' in optimization.constraints.l1.keys():

        tocon = optimization.constraints.l1['turnover']
        x_init = pd.Series(tocon['x0']).to_numpy()

        # Auxiliary variables to deal with the abs() function
        aux_turnover = optimization.model.addMVar(N, lb = np.zeros(N), name = 'aux_turnover')
        optimization.model.addConstr(x - aux_turnover <= x_init, name = 'aux_turnover_leq')
        optimization.model.addConstr(x + aux_turnover >= x_init, name = 'aux_turnover_geq')
        optimization.model.addConstr(aux_turnover.sum() <= tocon['rhs'], 'turnover_budget')

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
constraints.add_box(lower=0, upper=0.9)

# Add linear constraints
G = pd.DataFrame(
    np.zeros((3, len(constraints.selection))),
    columns=constraints.selection,
    index=['g1', 'g2', 'g3'],
)
G.iloc[0, 0:3] = 1
G.iloc[1, 3:6] = 1
G.iloc[2, 6:10] = 1

h = pd.Series(
    [0.5, 0.5, 0.5], # feasible
    # [0.3, 0.3, 0.3], # infeasible
    index=G.index
)
soft = pd.Series(
    [True, True, True],
    index=G.index
)
constraints.add_linear(
    Amat=G,
    sense='<=',
    rhs=h,
    soft=soft,
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

mv = MeanVariance(
    covariance=covariance,
    mean_estimator=mean_estimator,
    constraints=constraints,
)
mv.set_objective(optimization_data=data)




model_gurobi(optimization=mv)  # Creates an attribute 'model' in the optimization object
solve_gurobi(optimization=mv)  # Creates an attribute 'results' in the optimization object

mv.results  # Extract the results from the optimization object






