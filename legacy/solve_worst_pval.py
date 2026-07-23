import gurobipy as gp
from gurobipy import GRB, quicksum
import numpy as np
from scipy.stats import chi2

def solve_zeta_get_solution_c(I, n_i, q_array, t, Gamma, c, OutputFlag=0):
    """
    Solve the quadratic program

       min_{ρ in P_Γ} ζ(ρ;c)

    where
         ζ(ρ;c) = (t - s)^2 - [chi2.ppf(1-c, df=1)] * sum_{i=1}^I [w_i - v_i^2],
    with
         v_i = sum_j ρ_{ij} * q_array[i, j],
         w_i = sum_j ρ_{ij} * (q_array[i, j])^2,
         s   = sum_i v_i.

    Parameters:
      I         : number of strata.
      n_i       : list containing the number of elements in each stratum.
      q_array   : array of q values of shape (I, max(n_i)).
      t      : observed test statistic.
      Gamma     : sensitivity parameter.
      c         : current critical value (p-value).
      OutputFlag: Gurobi output flag.
      
    Returns a triple: (objective value, s_opt, D) where
         s_opt = s at optimum,
         D = sum_{i=1}^I (w_i - v_i^2),
    computed using the optimal ρ.
    """
    # Compute the multiplier from the current critical value.
    multiplier = chi2.ppf(1 - c, df=1)
    
    # Build a new model.
    model = gp.Model()
    model.setParam('OutputFlag', OutputFlag)
    
    # Decision variables: ρ_{ij}.
    rho = {}
    for i in range(I):
        for j in range(n_i[i]):
            rho[(i, j)] = model.addVar(lb=0.0, ub=1.0, name=f"rho_{i}_{j}")
    
    # Auxiliary variables: v_i, w_i and s.
    v = {}
    w = {}
    for i in range(I):
        v[i] = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"v_{i}")
        w[i] = model.addVar(lb=0.0, ub=GRB.INFINITY, name=f"w_{i}")
    s = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="s")
    
    # Constraints.
    for i in range(I):
        # (1) Sum_j ρ_{ij} = 1.
        model.addConstr(
            quicksum(rho[(i, j)] for j in range(n_i[i])) == 1.0,
            name=f"sum_rho_{i}"
        )
        # (2) Gamma-bounds: 1/(1+(n_i-1)*Gamma) ≤ ρ_{ij} ≤ Gamma/(Gamma+(n_i-1)).
        lb_rho = 1.0 / (1.0 + (n_i[i] - 1.0) * Gamma)
        ub_rho = Gamma / (Gamma + (n_i[i] - 1.0))
        for j in range(n_i[i]):
            model.addConstr(rho[(i, j)] >= lb_rho, name=f"rho_lb_{i}_{j}")
            model.addConstr(rho[(i, j)] <= ub_rho, name=f"rho_ub_{i}_{j}")
        
        # (3) Define v_i = sum_j ρ_{ij} * q_array[i, j].
        model.addConstr(
            v[i] == quicksum(rho[(i, j)] * q_array[i, j] for j in range(n_i[i])),
            name=f"v_def_{i}"
        )
        # (4) Define w_i = sum_j ρ_{ij} * (q_array[i, j]^2).
        model.addConstr(
            w[i] == quicksum(rho[(i, j)] * (q_array[i, j]**2) for j in range(n_i[i])),
            name=f"w_def_{i}"
        )
    
    # (5) Define s = sum_i v_i.
    model.addConstr(
        s == quicksum(v[i] for i in range(I)),
        name="s_def"
    )
    
    # Build the objective:
    # ζ(ρ;c) = (t - s)^2 - multiplier * sum_i (w_i - v_i^2).
    obj = (s - t) * (s - t) - multiplier * quicksum(w[i] - (v[i] * v[i]) for i in range(I))
    model.setObjective(obj, GRB.MINIMIZE)
    model.optimize()
    
    if model.status in [GRB.INFEASIBLE, GRB.INF_OR_UNBD]:
        return np.inf, None, None
    
    # Get the optimal s.
    s_opt = s.X
    # Compute D = sum_i (w_i - v_i^2) from the solution.
    D = 0.0
    for i in range(I):
        D += (w[i].X - v[i].X**2)
    
    return model.objVal, s_opt, D

def solve_worst_pval(I, n_i, q_array, t, Gamma=1.0, c_init=0.05, tol=1e-7, max_iter=50, OutputFlag=0):
    """
    Iterative fixed‑point procedure to solve for the critical value c.
    
    The condition for the solution is that for the optimizer ρ,
         (t - s)^2 - chi2.ppf(1-c, df=1) * [sum_i (w_i - v_i^2)] = 0.
    Equivalently, if we denote
         A = (t - s)^2   and   D = sum_i (w_i - v_i^2),
    then the equality is
         chi2.ppf(1-c,1) = A/D.
    
    Thus, given a candidate solution (s, D) from the optimizer (using c^(n)),
    we update c via
         c^(n+1) = 1 - chi2.cdf(A/D, 1).
    
    Parameters:
      I         : Number of strata.
      n_i       : List with the number of elements in each stratum.
      q_array   : Array of q values with shape (I, max(n_i)).
      t      : Observed test statistic.
      Gamma     : Sensitivity parameter.
      c_init    : Initial guess for the critical value (p‑value).
      tol       : Tolerance for convergence.
      max_iter  : Maximum number of iterations.
      OutputFlag: Gurobi output flag.
      
    Returns:
      (worst_pval, iter_count)
    where worst_pval is the converged critical value and iter_count is the number of iterations.
    """
    iter_count = 0
    c_old = c_init
    
    for _ in range(max_iter):
        iter_count += 1
        f_val, s_opt, D = solve_zeta_get_solution_c(I, n_i, q_array, t, Gamma, c_old, OutputFlag)
        if D is None or D <= 0:
            # If the denominator is nonpositive, we cannot update.
            break
        A = (t - s_opt)**2
        # The optimality condition would be chi2.ppf(1-c,1) = A/D.
        # Solve for c: c = 1 - chi2.cdf(A/D, 1).
        c_new = 1 - chi2.cdf(A / D, df=1)
        
        if abs(c_new - c_old) < tol:
            c_old = c_new
            break
        c_old = c_new

    worst_pval = c_old
    return worst_pval
