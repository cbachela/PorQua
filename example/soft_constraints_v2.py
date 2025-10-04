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
from optimization import Optimization, MeanVariance, Objective
from covariance import Covariance
from mean_estimation import MeanEstimator
from constraints import Constraints





# --------------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------------

def load_data_msci(path: str = None, n: int = 24) -> dict[str, pd.DataFrame]:
    '''Loads MSCI daily returns data from 1999-01-01 to 2023-04-18'''

    if path is None:
        # Get the project root directory (parent of the script's directory)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        path = os.path.join(project_root, 'data')
    
    # Ensure path ends with separator
    if not path.endswith(os.sep):
        path = path + os.sep

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



def solve_gurobi(optimization: Optimization) -> bool:

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
        
        # Extract slack variable values for soft constraints
        relaxed_constraints = {}
        if hasattr(optimization, 'slack_var') and optimization.slack_var is not None:
            slack_values = optimization.slack_var.X  # Get slack variable values
            soft_indices = optimization.soft_indices
            normalization_factors = optimization.normalization_factors
            
            # Get constraint names from the linear constraints
            if optimization.constraints.linear['Amat'] is not None:
                constraint_names = optimization.constraints.linear['Amat'].index.tolist()
                
                # Map soft indices to constraint information
                for i, (slack_idx, norm_factor) in enumerate(zip(soft_indices, normalization_factors)):
                    slack_val = slack_values[i]
                    if slack_val > 1e-6:  # Only report constraints that were actually relaxed
                        linear_ineq_count = 0
                        constraint_name = None
                        for j, (name, sense) in enumerate(zip(constraint_names, optimization.constraints.linear['sense'])):
                            if sense in ['<=', '>=']:
                                if linear_ineq_count == slack_idx:
                                    constraint_name = name
                                    break
                                linear_ineq_count += 1
                        
                        if constraint_name is None:
                            constraint_name = f"constraint_{slack_idx}"
                        
                        relaxed_constraints[constraint_name] = {
                            'slack_value': slack_val,
                            'normalization_factor': norm_factor,
                            'constraint_index': slack_idx
                        }
        
    else:
        weights = {key: np.nan for key in ids}
        obj_val = np.nan
        relaxed_constraints = {}

    optimization.results = {
        'weights': weights,
        'objective': obj_val,
        'status': status,
        'relaxed_constraints': relaxed_constraints
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
    slack_penalty = optimization.params.get('slack_penalty', 1e6)

    # Initialize Gurobi model
    optimization.model = Model('portfolio')
    optimization.model.Params.LogToConsole = 0

    # Prepare decision variable 'x' as matrix variable including
    # lower and upper bounds and variable type
    lb = optimization.constraints.box['lower'].to_numpy()
    ub = optimization.constraints.box['upper'].to_numpy()
    v_type = GRB.CONTINUOUS
    x = optimization.model.addMVar(N, lb = lb, ub = ub, vtype = v_type, name = 'x')

    # Store soft constraint information for later use in solve_gurobi
    optimization.soft_indices = GhAb.get('soft_indices', [])
    optimization.normalization_factors = GhAb.get('normalization_factors', [])

    # Add slack variables to the model for soft constraints
    num_slack = len(optimization.soft_indices)
    if num_slack > 0:
        slack = optimization.model.addMVar(num_slack, lb = np.zeros(num_slack), name = 'slack')
        optimization.slack_var = slack  # Store for later retrieval
    else:
        slack = None
        optimization.slack_var = None

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
    
    # Add slack penalty term to the objective function
    if slack is not None:
        obj_fun = obj_fun + slack_penalty * slack.sum()

    optimization.model.setObjective(obj_fun, GRB.MINIMIZE)

    # Add linear inequality constraints with slack variables for soft constraints
    if GhAb['G'] is not None:
        if num_slack > 0:
            # For each inequality constraint, add slack if it's soft
            for i in range(GhAb['G'].shape[0]):
                if i in optimization.soft_indices:
                    # Get the index in the slack array
                    slack_idx = optimization.soft_indices.index(i)
                    optimization.model.addConstr(GhAb['G'][i] @ x <= GhAb['h'][i] + slack[slack_idx], f'Gh_soft_{i}')
                else:
                    # Add hard constraint
                    optimization.model.addConstr(GhAb['G'][i] @ x <= GhAb['h'][i], f'Gh_hard_{i}')
        else:
            # No soft constraints, add all as hard constraints
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

data = load_data_msci(n = 10)
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
    # [0.5, 0.5, 0.5], # feasible
    [0.3, 0.3, 0.3], # infeasible
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
print(mv.results)
