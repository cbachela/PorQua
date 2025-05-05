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
from optimization import MeanVariance
from covariance import Covariance
from mean_estimation import MeanEstimator
from builders import (
    SelectionItemBuilder,
    OptimizationItemBuilder,
    bibfn_selection_data,
    bibfn_return_series,
    bibfn_budget_constraint,
    bibfn_box_constraints,
)






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


# Backtest item builder functions - Optimization constraints
def bibfn_group_constraints(bs: 'BacktestService', rebdate: str, **kwargs) -> None:

    '''
    Backtest item builder function for setting linear group constraints.
    '''

    # Arguments
    Amat = kwargs.get('Amat')
    a_values = kwargs.get('a_values')
    sense = kwargs.get('sense', '<=')
    rhs = kwargs.get('rhs')
    name = kwargs.get('name', 'group_constraint')
    soft = kwargs.get('soft', False)

    # Constraints
    bs.optimization.constraints.add_linear(
        Amat = Amat,
        a_values = a_values,
        rhs = rhs,
        sense = sense,
        name = name,
        # soft = soft,
    )
    return None



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

    optimization.results = {'weights': weights,
                    'objective': obj_val,
                    'status': status}
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
# Define the selection and optimization item builders
# --------------------------------------------------------------------------

selection_item_builders = {
    'data': SelectionItemBuilder(
        bibfn = bibfn_selection_data
    ),
}

optimization_item_builders = {
    'return_series': OptimizationItemBuilder(
        bibfn = bibfn_return_series,
        width = 365 * 3,
    ),
    'budget_constraint': OptimizationItemBuilder(
        bibfn = bibfn_budget_constraint,
        budget = 1,
    ),
    'box_constraints': OptimizationItemBuilder(
        bibfn = bibfn_box_constraints,
        upper = 0.9,
        lower = 0,
    ),
    'group_constraints_g1': OptimizationItemBuilder(
        bibfn = bibfn_group_constraints,
        a_values = pd.Series(1, index=['AT', 'AU', 'BE']),
        # rhs = 0.1,  # --> infeasible
        rhs = 0.3,    # --> infeasible
        name = 'group_1',
        soft = True,
    ),
    'group_constraints_g2': OptimizationItemBuilder(
        bibfn = bibfn_group_constraints,
        a_values = pd.Series(1, index=['CA', 'CH', 'DE']),
        # rhs = 0.1,  # --> infeasible
        rhs = 0.3,    # --> infeasible
        name = 'group_2',
        soft = True,
    ),
    'group_constraints_g3': OptimizationItemBuilder(
        bibfn = bibfn_group_constraints,
        a_values = pd.Series(1, index=['DK', 'ES', 'FI', 'FR']),
        # rhs = 0.1,  # --> infeasible
        # rhs = 0.3,    # --> infeasible
        rhs = 0.5,  # --> feasible
        name = 'group_3',
        soft = True,
    ),
}



# --------------------------------------------------------------------------
# Initialize the optimization object and required estimators
# --------------------------------------------------------------------------

covariance = Covariance(method='pearson')
mean_estimator = MeanEstimator(method='geometric')

optim = MeanVariance(
    covariance=covariance,
    mean_estimator=mean_estimator,
    solver_name='gurobi',
)




      
# --------------------------------------------------------------------------
# Initialize the backtest service
# --------------------------------------------------------------------------

bs = BacktestService(
    data = data,
    optimization = optim,
    selection_item_builders = selection_item_builders,
    optimization_item_builders = optimization_item_builders,
    rebdates = ['2023-03-27'],
)



# --------------------------------------------------------------------------
# Prepare optimization for a specific date
# --------------------------------------------------------------------------

rebalancing_date = bs.settings['rebdates'][0]
bs.prepare_rebalancing(rebalancing_date=rebalancing_date)
bs.optimization.set_objective(bs.optimization_data)


model_gurobi(bs.optimization)  # Creates an attribute 'model' in the optimization object
solve_gurobi(bs.optimization)

bs.optimization.results











