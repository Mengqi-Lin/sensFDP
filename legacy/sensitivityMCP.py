import gurobipy as gp
from gurobipy import Model, GRB, quicksum
import time
from itertools import chain, combinations, product
import itertools
import numpy as np
import pandas as pd
from scipy.stats import t
import collections
import numpy as np
from scipy.stats import median_abs_deviation as mad
from scipy.stats import multivariate_normal as mvn
from scipy.stats import chi2 
import random
from tqdm import tqdm
import math
import sys
import csv
from statistics import mean
from scipy.stats import norm
from PO_to_Qmat import PO_to_Qmat

class MyPackageException(Exception):
    def __init__(self, message, error_type=None):
        super().__init__(message)
        self.error_type = error_type

        
def holm_bonferroni(p_values, alpha = 0.05):
    # Number of tests
    K = len(p_values)
    
    # Sort the p-values
    sorted_indices = np.argsort(p_values)
    sorted_p_values = np.array(p_values)[sorted_indices]
    
    # Adjust the alpha level and find where p > adjusted_alpha
    adjusted_alpha = alpha / (K - np.arange(K))
    reject = sorted_p_values <= adjusted_alpha
    
    # If at any point p > adjusted_alpha, reject all hypotheses beyond this point
    if any(~reject):
        reject[np.argmax(~reject):] = False
    
    # Return the sorted p-values, adjusted_alpha and rejection decision in original order
    rejected_indices = sorted_indices[reject]
    
    return set(rejected_indices)


def init_R0(n_i, Tobs, Qarray, alpha):

    I = len(n_i)
    K = len(Tobs)

    rho_ij = np.zeros((I, max(n_i)))

    # Loop over each group i
    for i in range(I):
        # For each i, loop over the observations j
        for j in range(n_i[i]):
            # Set rho_ij[i, j] to 1 / n_i[i], so the sum over j will be 1
            rho_ij[i, j] = 1 / n_i[i]

    p_values = []
    for k in range(K):
        chi_value = (Tobs[k] - sum(rho_ij[i, j]*Qarray[i, j, k] for i in range(I) for j in range(n_i[i])))**2
        chi_value /= sum(sum(rho_ij[i, j]*(Qarray[i, j, k]**2) for j in range(n_i[i])) - sum((rho_ij[i, j]*Qarray[i, j, k]) for j in range(n_i[i]))**2 for i in range(I))
        p_value = 1 - chi2.cdf(chi_value, df = 1)
        p_values.append(p_value)
        
    R0 = holm_bonferroni(p_values, alpha = alpha)
    
    return R0


## Calculate v(cR) for Bonferroni-Based closed testing procedure with given p-values.
def vR_given_pvals(p_values, cR, alpha):
    
    K = len(p_values)
    RHB = holm_bonferroni(p_values, alpha = alpha)
    nonHB = set(range(K)) - RHB
    cR_nrej = nonHB.intersection(cR)
    return len(cR_nrej)

## A uitil function that calculates the worst-case expectation and sigma for alternative == "G"; other alternatives can be obtained based on this.
def calculate_mu_and_sigma(I, n_i, q, Gamma):
    mu_i_values = []
    nu_i_values_squared = []
    sigma_I_squared = 0

    for i in range(I):
        q_i_sorted = np.sort(q[i, :n_i[i]])
        mu_ia_max = float('-inf')
        nu_ia_max_squared = float('-inf')
        A = []

        # Compute mu_ia and nu_ia_squared for each a
        for a in range(1, n_i[i]):
            sum_q_below = sum(q_i_sorted[:a])
            sum_q_above = sum(q_i_sorted[a:])
            sum_q_squared_below = sum(x**2 for x in q_i_sorted[:a])
            sum_q_squared_above = sum(x**2 for x in q_i_sorted[a:])
            denominator = a + Gamma * (n_i[i] - a)

            mu_ia = (sum_q_below + Gamma * sum_q_above) / denominator
            nu_ia_squared = (sum_q_squared_below + Gamma * sum_q_squared_above) / denominator - mu_ia**2

            # Update maxima and set A
            if mu_ia > mu_ia_max or np.isclose(mu_ia, mu_ia_max):
                if mu_ia > mu_ia_max:
                    A = [a]
                    mu_ia_max = mu_ia
                    nu_ia_max_squared = nu_ia_squared
                else:
                    A.append(a)
                    nu_ia_max_squared = max(nu_ia_max_squared, nu_ia_squared)

        mu_i_values.append(mu_ia_max)
        nu_i_values_squared.append(nu_ia_max_squared)
        sigma_I_squared += nu_ia_max_squared

    # Calculate sum of mu_i values and standard deviation
    sum_mu_i = sum(mu_i_values)
    sigma_I = np.sqrt(sigma_I_squared)

    return sum_mu_i, sigma_I
    
    
## Using separability aglorithm to calculate the worst-case p-value, for alterantive == "G", "L", "TS".
def worst_pval(I, n_i, q, Gamma, t, alternative="TS"):
    pval = 1
    
    ## Special case for Gamma = 1.
    if abs(Gamma - 1) < 1e-7:
        mu = sum((1 / n_i[i]) * q[i, j] for i in range(I) for j in range(n_i[i]))
        sigma_squared = sum(sum((1 / n_i[i]) * (q[i, j]**2) for j in range(n_i[i])) - sum(((1 / n_i[i]) * q[i, j]) for j in range(n_i[i]))**2 for i in range(I))
        z_score = (t - mu) / np.sqrt(sigma_squared)
        if alternative == "G":
            pval = 1 - norm.cdf(z_score)
        elif alternative == "L":
            pval = norm.cdf(z_score)
        elif alternative == "TS":
            pval = 2 * (1 - norm.cdf(abs(z_score)))
        return pval
    
    # Calculate for the "G" (Greater) alternative
    if alternative == "G":
        mu_G, sigma_G = calculate_mu_and_sigma(I, n_i, q, Gamma)
        z_score = (t - mu_G) / sigma_G
        pval = 1 - norm.cdf(z_score)
            
    # Calculate for the "L" (Less) alternative by negating q
    elif alternative == "L":
        mu_L, sigma_L = calculate_mu_and_sigma(I, n_i, -q, Gamma)
        mu_L = -mu_L  # Reverse sign for "L"
        z_score = (t - mu_L) / sigma_L
        pval = norm.cdf(z_score)

    # For "Two-Sided" alternative
    elif alternative == "TS":
        mu_G, sigma_G = calculate_mu_and_sigma(I, n_i, q, Gamma)
        mu_L, sigma_L = calculate_mu_and_sigma(I, n_i, -q, Gamma)
        mu_L = -mu_L  # Reverse sign for "L"
        if mu_L <= t <= mu_G:
            pval = 1
        else:
            # Calculate two-sided p-value based on the worst-case deviation
            if t > mu_G:
                z_score = (t - mu_G) / sigma_G
                pval = 2 * (1 - norm.cdf(abs(z_score)))
            else:
                z_score = (mu_L - t) / sigma_L
                pval = 2 * (1 - norm.cdf(abs(z_score)))

    return pval




def zeta_lower_bound(I, n_i, q_ij, Tval, Gamma, level):
    chi = chi2.ppf(1 - level, df=1)
    model = gp.Model()
    model.setParam('OutputFlag',0)

    #rho_ij
    rho_ij = model.addVars(I, range(max(n_i)), name="rho")
    s = model.addVars(I, lb=0.0, name="s")

    #zeta_k = model.addVars(K, lb = -GRB.INFINITY, ub = GRB.INFINITY, name = "zeta")
    zeta_k = model.addVar(lb = -GRB.INFINITY, ub = GRB.INFINITY, name = "zeta")

    #  m_k: observed test statistic minus expectation summed across all matched sets
    # adding this variable is ESSENTIAL for speedup, since it eliminates need for I^2 cross terms
    # in constraints

    m_k = model.addVar(lb = -GRB.INFINITY, ub = GRB.INFINITY, name = "m")
    # expectation of Tik, test statistic for kth outcome in ith stratum. Also seems
    # beneficial to encode this, since it eliminates $n_i^2$ cross-terms.
    v_ik = model.addVars(I, lb = -GRB.INFINITY, ub = GRB.INFINITY, name = "v")

    model.setObjective(zeta_k, GRB.MINIMIZE)

    # rhoij
    for i in range(I):
        model.addConstr(gp.quicksum(rho_ij[i, j] for j in range(n_i[i])) == 1)
        for j in range(n_i[i]):
            model.addConstr(s[i] <= rho_ij[i, j], name=f"lower_bound_set_{i}_{j}")
            model.addConstr(rho_ij[i, j] <= Gamma * s[i], name=f"upper_bound_set_{i}_{j}")

    # z_k temporary variable (seems useful for calculating bounds during presolve, and
    # for encoding general constraint)
    z_k = m_k**2 + chi*quicksum(v_ik[i]**2 - quicksum(rho_ij[i, j]*(q_ij[i, j]**2) for j in range(n_i[i])) for i in range(I))

    #expectation of Tik, test statistic for kth outcome in ith stratum
    for i in range(I):
        model.addConstr(v_ik[i] == quicksum(rho_ij[i, j]*q_ij[i, j] for j in range(n_i[i])))
    # observed test statistic minus expectation summed across all matched sets
    model.addConstr(m_k == (Tval - quicksum(rho_ij[i, j]*q_ij[i, j] for i in range(I) for j in range(n_i[i]))))

    # zeta_k[k] >= z_k. Since minimizing, inequality constraint is actually equality at solution.
    # needs to be inequality to be a convex constraint
    model.addConstr(zeta_k >= z_k)
    model.optimize()
    return model.objVal



# Function solve_gnull: Solving quardratic constraints optimization in (Fogarty and Small).
# Return True if Reject.
def solve_gnull(I, K, n_i, Qarray, Tobs, criticalval, Gamma = 1, OutputFlag=0):

    # Compute the chi-squared quantile.
    chi = chi2.ppf(1 - criticalval, df=1)

    # Create the Gurobi model.
    model = gp.Model()
    model.setParam('OutputFlag', OutputFlag)
    model.setParam('SolutionLimit', 1)

    # Create variables.
    y = model.addVar(lb=-GRB.INFINITY, ub=0, name="y")
    rho_ij = model.addVars(I, range(max(n_i)), name="rho")
    s = model.addVars(I, lb=0.0, name="s")
    zeta_k = model.addVars(K, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="zeta")
    m_k = model.addVars(K, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="m")
    v_ik = model.addVars(I, K, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="v")

    # Set the objective.
    model.setObjective(y, GRB.MINIMIZE)

    # Array to store computed lower bounds for each k.
    zetaLB = np.zeros(K)

    # Loop over each k to add constraints.
    for k in range(K):
        # Build temporary expression z_k.
        z_k = m_k[k]**2 + chi * quicksum(v_ik[i, k]**2 - quicksum(rho_ij[i, j] * (Qarray[i, j, k]**2) for j in range(n_i[i])) for i in range(I))
        # Define v_ik in terms of rho_ij and Qarray.
        for i in range(I):
            model.addConstr(v_ik[i, k] == quicksum(rho_ij[i, j] * Qarray[i, j, k] for j in range(n_i[i])))
        # Define m_k as the difference between observed statistic and expected sum.
        model.addConstr(
            m_k[k] == (Tobs[k] - quicksum(rho_ij[i, j] * Qarray[i, j, k] for i in range(I) for j in range(n_i[i]))))
        # zeta_k[k] must be at least z_k.
        model.addConstr(zeta_k[k] >= z_k, name=f"zeta_lower_bound_{k}")
        # y is an upper bound on all zeta_k.
        model.addConstr(y >= zeta_k[k], name=f"zeta_constraint_{k}")

        # Compute a lower bound for the kth outcome.
        zetaLB[k] = zeta_lower_bound(I = I, n_i = n_i, q_ij=Qarray[:, :, k],
                                       Tval=Tobs[k], Gamma = Gamma, level=criticalval)

    # Enforce y to be at least the maximum of these computed lower bounds.
    model.addConstr(y >= zetaLB.max(), name="zeta_lb_global")

    # Constraints for the rho variables.
    for i in range(I):
        model.addConstr(quicksum(rho_ij[i, j] for j in range(n_i[i])) == 1)
        for j in range(n_i[i]):
            model.addConstr(s[i] <= rho_ij[i, j], name=f"lower_bound_set_{i}_{j}")
            model.addConstr(rho_ij[i, j] <= Gamma * s[i], name=f"upper_bound_set_{i}_{j}")



    # Optimize the model.
    model.optimize()

    # If numeric difficulties arise, adjust parameters and re-optimize.
    if model.status == GRB.Status.NUMERIC:
        model.setParam('BarHomogeneous', 1)
        model.setParam("NumericFocus", 3)
        model.setParam("ScaleFlag", 2)
        model.optimize()

    # Decide based on the model status.
    if model.status in [GRB.Status.OPTIMAL, GRB.Status.SOLUTION_LIMIT, GRB.Status.SUBOPTIMAL]:
        # Feasible solution found: fail to reject the global null.
        return False
    elif model.status in [GRB.Status.INFEASIBLE, GRB.Status.INF_OR_UNBD]:
        # No feasible solution: reject the global null.
        return True
    else:
        print(f"Numeric issue encountered.")
        return False

        
# Function does solve_vR when R is a singleton.
def solve_vR_singleton(I, K, k0, n_i, Qarray, Tobs, rejected, worst_pvalues, alpha = 0.05, Gamma = 1, OutputFlag=0):
    nrejected = set(range(K)) - rejected
    v_up = len(nrejected)
    # make sure p_{k0,Gamma}^* > alpha / v_lo
    v_lo = math.floor(alpha / worst_pvalues[k0]) + 1
    for v in range(v_lo, v_up + 1):
        cand_J = {k for k in nrejected if worst_pvalues[k] > alpha / v}
        if len(cand_J) < v:
            continue
            
            
        zetaLB = np.zeros(K)
        for k in cand_J:
            zetaLB[k] = zeta_lower_bound(I = I, n_i = n_i, q_ij=Qarray[:, :, k],
                                         Tval=Tobs[k], Gamma = Gamma,
                                         level=alpha / v)
        # Sort and determine lower bound for y.
        sortLB = np.sort([zetaLB[k] for k in cand_J])
        
        # Note: Ensure vR_low-1 and v-1 are within bounds.
        try:
            yLB = np.max([zetaLB[k0], sortLB[v - 1]])
        except Exception as e:
            yLB = -GRB.INFINITY


        # The Gurobi model
        model = gp.Model()
        model.setParam('OutputFlag', OutputFlag)
        model.setParam('SolutionLimit', 1)
        model.setParam('MIPFocus', 1)
        model.setParam('MIQCPMethod', 1)
        model.setParam('Method', 1)

        #rho_ij
        rho_ij = model.addVars(I, range(max(n_i)), name="rho")
        # s
        s = model.addVars(I, lb=0.0, name="s")
        # variable indicators
        theta_k = model.addVars(K, vtype=GRB.BINARY, name="theta")
        # zetas
        zeta_k = model.addVars(K, lb = -GRB.INFINITY, ub = GRB.INFINITY, name = "zeta")
        # max of variables
        y = model.addVar(lb=yLB, ub = 0 , name="y")

        #  m_k: observed test statistic minus expectation summed across all matched sets
        # adding this variable is ESSENTIAL for speedup, since it eliminates need for I^2 cross terms
        # in constraints

        m_k = model.addVars(K, lb = -GRB.INFINITY, ub = GRB.INFINITY, name = "m")

        # expectation of Tik, test statistic for kth outcome in ith stratum. Also seems
        # beneficial to encode this, since it eliminates $n_i^2$ cross-terms.
        v_ik = model.addVars(I, K, lb = -GRB.INFINITY, ub = GRB.INFINITY, name = "v")

        model.setObjective(y, GRB.MINIMIZE)

        chi = chi2.ppf(1 - alpha / v, df=1)
        # rhoij
        for i in range(I):
            model.addConstr(gp.quicksum(rho_ij[i, j] for j in range(n_i[i])) == 1)
            for j in range(n_i[i]):
                model.addConstr(s[i] <= rho_ij[i, j], name=f"lower_bound_set_{i}_{j}")
                model.addConstr(rho_ij[i, j] <= Gamma * s[i], name=f"upper_bound_set_{i}_{j}")

        for k in cand_J:

            # z_k temporary variable (seems useful for calculating bounds during presolve, and for encoding general constraint)
            z_k = m_k[k]**2 + chi*quicksum(v_ik[i, k]**2 - quicksum(rho_ij[i, j]*(Qarray[i, j, k]**2) for j in range(n_i[i]))for i in range(I))
            #expectation of Tik, test statistic for kth outcome in ith stratum
            for i in range(I):
                model.addConstr(v_ik[i, k] == quicksum(rho_ij[i, j]*Qarray[i, j, k] for j in range(n_i[i])))
            # observed test statistic minus expectation summed across all matched sets
            model.addConstr(m_k[k] == (Tobs[k] - quicksum(rho_ij[i, j]*Qarray[i, j, k] for i in range(I) for j in range(n_i[i]))))

            # zeta_k[k] >= z_k. Since minimizing, inequality constraint is actually equality at solution.
            # needs to be inequality to be a convex constraint
            model.addConstr(zeta_k[k] >= z_k)

            #Indicator constraint
            model.addGenConstrIndicator(theta_k[k], True, y >= zeta_k[k])


        # theta constraints
        model.addConstr(quicksum(theta_k[k] for k in cand_J) == v)
        model.addConstr(theta_k[k0] == 1)

        try:
            # Optimize
            model.optimize()
            if model.status in [GRB.Status.OPTIMAL, GRB.Status.SOLUTION_LIMIT, GRB.Status.SUBOPTIMAL]:  # OPTIMAL or SOLUTION_LIMIT
                try:
                    # Attempt to retrieve the 'X' attribute
                    acpts = [k for k in cand_J if theta_k[k].X == 1]
                    # If successful, you can add additional logic here
                    # For example, appending to a list or performing calculations
                except AttributeError:
                    # Handle the case where 'X' cannot be retrieved
                    # You can print a message, set x_value to a default value, or pass
                    print(f"Unable to retrieve attribute 'X' for variable theta_k[{k}], model status{model.status}")
                    # Optionally set x_value to None or some default value if needed
                    acpts = []  # or some default value
                return False, acpts  # Accept H_{k_0} and other whose thetas == 1
            
            elif model.status in [GRB.Status.INFEASIBLE, GRB.Status.INF_OR_UNBD]:  # Infeasible
                continue  # Try the next v value
            else:
                raise Exception(f"Unexpected Gurobi status: {model.status}")

        except gp.GurobiError as e:
            raise MyPackageException(f"Gurobi encountered an error: {e}", error_type="GUROBI_ERROR")

    return True, None  # Reject H_{k_0}




def solve_vR(I, K, cR, n_i, Qarray, Tobs, rejected, worst_pvalues, alpha=0.05, Gamma = 1, alternative="TS", OutputFlag=0):
    
    # Prepare sets and initial values. We can treat nrejected as if they are the whole set, i.e., ignoring those that are rejected.
    nrejected = set(range(K)) - rejected
    ncR = set(range(K)) - set(cR)        
    cR = nrejected.intersection(cR)
    ncR = nrejected.intersection(ncR)
    cR_array = np.array(list(cR))

    vR_up = len(cR)
    vR_low = 0
    if vR_up == 0:
        return 0
    if any(worst_pvalues > alpha):
        vR_low = 1


    # Fast-screening at top node cR. If we can directly fail to reject cR, return len(cR).
    lb = solve_gnull(I, vR_up, n_i, Qarray[:, :, cR_array], Tobs[cR_array],
                     criticalval = alpha/vR_up, Gamma = Gamma, OutputFlag = OutputFlag)
    if not lb:
        return vR_up

    # When there are more nrejected, compare top node with extreme value alpha/len(nrejected).
    if len(nrejected) > vR_up:
        ub = solve_gnull(I, vR_up, n_i, Qarray[:, :, cR_array], Tobs[cR_array],
                         criticalval = alpha/len(nrejected), Gamma = Gamma, OutputFlag = OutputFlag)
        if ub:
            vR_up -= 1

    if vR_up <= vR_low:
        return vR_low

    # Main loop: try different values for r and v.
    for r in range(vR_up, vR_low, -1):
        for val in range(len(ncR) + 1):
            v = r + val
            cand_I = {k for k in cR if worst_pvalues[k] > alpha / v}
            cand_J = {k for k in nrejected if worst_pvalues[k] > alpha / v}
            if len(cand_I) < r or len(cand_J) < v:
                continue
            # Calculate lower bounds for zeta_k if lb_opt is set.
            zetaLB = np.zeros(K)
            for k in cand_J:
                zetaLB[k] = zeta_lower_bound(I = I, n_i = n_i, q_ij=Qarray[:, :, k],
                                             Tval=Tobs[k], Gamma = Gamma,
                                             level=alpha / v)
            # Sort and determine lower bound for y.
            sortLB = np.sort([zetaLB[k] for k in cand_J])
            sortLB_cR = np.sort([zetaLB[k] for k in cand_I])

            # Note: Ensure vR_low-1 and v-1 are within bounds.
            try:
                yLB = np.max([sortLB_cR[r - 1], sortLB[v - 1]])
            except Exception as e:
                yLB = -GRB.INFINITY

            model = gp.Model()
            model.setParam('OutputFlag', OutputFlag)
            model.setParam('SolutionLimit', 1)
            model.setParam('MIPFocus', 1)
            model.setParam('MIQCPMethod', 1)
            model.setParam('Method', 1)

            # Create variables.
            rho_ij = model.addVars(I, range(max(n_i)), name="rho")
            s = model.addVars(I, lb=0.0, name="s")
            theta_k = model.addVars(K, vtype=GRB.BINARY, name="theta")
            zeta_k = model.addVars(K, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="zeta")
            y = model.addVar(lb=yLB, ub=0, name="y")
            m_k = model.addVars(K, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="m")
            v_ik = model.addVars(I, K, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="v")

            model.setObjective(y, GRB.MINIMIZE)
            chi = chi2.ppf(1 - alpha / v, df=1)

            # Constraints for rho_ij.
            for i in range(I):
                model.addConstr(gp.quicksum(rho_ij[i, j] for j in range(n_i[i])) == 1)
                for j in range(n_i[i]):
                    model.addConstr(s[i] <= rho_ij[i, j], name=f"lower_bound_set_{i}_{j}")
                    model.addConstr(rho_ij[i, j] <= Gamma * s[i], name=f"upper_bound_set_{i}_{j}")

            # Constraints for each k in cand_J.
            for k in cand_J:
                z_k = m_k[k]**2 + chi * gp.quicksum(v_ik[i, k]**2 - gp.quicksum(rho_ij[i, j] * (Qarray[i, j, k]**2) for j in range(n_i[i])) for i in range(I))
                for i in range(I):
                    model.addConstr(v_ik[i, k] == gp.quicksum(rho_ij[i, j] * Qarray[i, j, k] for j in range(n_i[i])))
                model.addConstr(m_k[k] == (Tobs[k] - gp.quicksum(rho_ij[i, j] * Qarray[i, j, k] for i in range(I) for j in range(n_i[i]))))
                model.addConstr(zeta_k[k] >= z_k)

                model.addGenConstrIndicator(theta_k[k], True, y >= zeta_k[k])

            # Theta constraints.
            model.addConstr(gp.quicksum(theta_k[k] for k in cand_I) == r)
            model.addConstr(gp.quicksum(theta_k[k] for k in cand_J) == v)

            try:
                model.optimize()
                if model.status in [GRB.Status.OPTIMAL, GRB.Status.SOLUTION_LIMIT, GRB.Status.SUBOPTIMAL]:
                    return r
                elif model.status in [GRB.Status.INFEASIBLE, GRB.Status.INF_OR_UNBD]:
                    # Infeasible: try next combination.
                    continue
            except gp.GurobiError as e:
                raise MyPackageException(f"Gurobi encountered an error: {e}", error_type="GUROBI_ERROR")

    return vR_low



def data_process(index, Qmat, Z):
    """
    Process the data for matched sets and compute observed test statistics.
    
    This function performs the following steps:
      1. Converts inputs to copies so that the originals are not modified.
      2. Counts the number of observations and treated individuals in each stratum.
      3. Checks that each stratum has either one treated or one control.
      4. Flips the treatment assignment and corresponding Qmat values in strata where needed.
      5. Scales Qmat for optimization and computes the observed t-statistics.
      6. Reshapes Qmat into a 3D array Qarray.
    
    Parameters
    ----------
    index : array-like
        Array of stratum indicators.
    Qmat : array-like, shape (n, m)
        Outcome matrix.
    Z : array-like
        Treatment vector (should be binary: 0/1).
    
    Returns
    -------
    I : int
        Number of strata.
    K : int
        Number of columns in Qmat.
    n_i : list
        List containing the number of observations in each stratum.
    Qarray : ndarray, shape (I, max(n_i), K)
        Reshaped outcome array.
    Tobs : ndarray, shape (m,)
        Observed test statistics computed as the column sums of Qmat for treated observations.
    """
    # Create copies of the inputs to avoid modifying the originals.
    index = np.copy(index)
    Qmat = np.copy(Qmat)
    Z = np.copy(Z)
    
    # Force Z to be numeric (0/1) and create a copy.
    Z = 1 * Z
    
    # Count total observations per stratum and treated counts.
    ns = collections.Counter(index)
    n_i = list(ns.values())
    ms = collections.Counter(index[Z == 1])
    I = len(np.unique(index))
    N_total = Qmat.shape[0]
    treatment = (Z == 1)
    
    # Determine the number of columns.
    if len(Qmat.shape) == 1:  # If Qmat is 1D, then there is only one column.
        K = 1
        # Convert to a 2D array for consistency.
        Qmat = Qmat.reshape(-1, 1)
    else:
        K = Qmat.shape[1]
        
    # Check that each stratum has either exactly one treated or exactly one control.
    for key in ns:
        treated = ms[key]
        controls = ns[key] - treated
        if treated != 1 and controls != 1:
            raise ValueError(
                "Strata must have either one treated and the rest controls, "
                "or one control and the rest treated."
            )
    
    if np.any((Z != 0) & (Z != 1)):
        raise ValueError("Treatment Vector (Z) Must be Binary")
    
    # Flip treatment and Qmat values in strata where more than one subject is treated.
    for i in range(I):
        if ms[i] > 1:
            ind = np.where(index == i)[0]
            if len(ind) - 1 != ms[i]:
                raise ValueError("bug!")
            # Flip treatment: 0 -> 1 and 1 -> 0.
            Z[ind] = 1 - Z[ind]
            # Flip Qmat values: For each column, set Qmat[ind] = qsum - Qmat[ind].
            for k in range(K):
                qsum = np.sum(Qmat[ind, k])
                Qmat[ind, k] = qsum - Qmat[ind, k]
    
    # Recount treated individuals after flipping.
    msnew = collections.Counter(index[Z == 1])
    if not all(count == 1 for count in msnew.values()):
        raise ValueError("bug!")
    
    treatment = (Z == 1)
    
    # Scale Qmat for optimization.
    sds = np.std(Qmat, axis=0)
    SDS = np.tile(sds, (Qmat.shape[0], 1)) * np.sqrt(Qmat.shape[0])
    Qmat = Qmat / SDS
    
    # Observed t-statistics: sum over treated rows for each column.
    Tobs = np.sum(Qmat[treatment, :], axis=0)
    
    # Reshape Qmat into Qarray: shape (I, max(n_i), K)
    Qarray = np.zeros((I, np.max(n_i), K))
    for k in range(K):
        for i in range(I):
            Qarray[i, :n_i[i], k] = Qmat[index == i, k]
    
    return I, K, n_i, Qarray, Tobs



def gnullsensitivity(index, Qmat, Z, alpha = 0.05, alternative = "TS", Gamma = 1, OutputFlag = 0):

    I, K, n_i, Qarray, Tobs = data_process(index, Qmat, Z)

    rej = solve_gnull(I = I, K = K, n_i = n_i, Qarray = Qarray, Tobs = Tobs, criticalval = alpha/K, Gamma = Gamma, OutputFlag = OutputFlag)
    
    return rej



def gSensitivity_value_local(index, Qmat, Z,
                             alpha=0.05,
                             loGamma=1.0, upGamma=3.0,
                             precision=None,
                             alternative="TS",
                             OutputFlag=0):
    """
    If precision is provided (not None), runs the original single-pass search at that precision.
    Otherwise, performs a two-phase search: coarse step=0.1, then refine at 0.01.
    
    Returns the estimated change-point Gamma.
    """
    # Step 1: process data once
    I, K, n_i, Qarray, Tobs = data_process(index, Qmat, Z)
    cval = alpha/K

    def _two_false_search(gammas, K, Qarray, Tobs, criticalval):
        prev_false = None
        for G in gammas:
            rej = solve_gnull(I, K, n_i, Qarray, Tobs,
                              criticalval=cval, Gamma=G, OutputFlag=OutputFlag)
            if not rej:
                if prev_false is None:
                    prev_false = G
                else:
                    return prev_false
            else:
                prev_false = None
        return None

    # Original single-pass behavior when precision specified
    if precision is not None:
        # Round bounds based on precision
        dp = math.ceil(-math.log10(precision))
        loG = round(loGamma, dp)
        upG = round(upGamma, dp)
        num_steps = int(round((upG - loG) / precision)) + 1
        gammas = np.linspace(loG, upG, num=num_steps)
        # single-pass two‐false rule
        S = _two_false_search(gammas, K, Qarray, Tobs, cval)
        return S if S is not None else upG

    # Otherwise: two-phase search
    coarse = 0.1
    fine   = 0.01
    # rounding for fine precision
    dp = math.ceil(-math.log10(fine))
    loG = round(loGamma, dp)
    upG = round(upGamma, dp)

    # phase 1: coarse
    gam_coarse = np.arange(loG, upG + coarse/2, coarse)
    S_coarse = _two_false_search(gam_coarse, K, Qarray, Tobs, cval)
    if S_coarse is None:
        return upG

    # phase 2: refine in [S_coarse - coarse, S_coarse]
    lower = max(loG, S_coarse - coarse)
    gam_refine = np.arange(lower, S_coarse + fine/2, fine)
    S_ref = _two_false_search(gam_refine, K, Qarray, Tobs, cval)
    return S_ref if S_ref is not None else S_coarse


def closed_sensitivity(index, Qmat, Z, cR = None, alpha = 0.05, alternative="TS", Gamma = 1, OutputFlag=1):

    I, K, n_i, Qarray, Tobs = data_process(index, Qmat, Z)
    if cR == None:
        cR = set(range(K))
    else:
        cR = set(cR)
        
    ## screening
    
    # Initialize the acceptance set and rejection set 
    # Those that are not rejected in a trivial HB, will also not be rejected in closes sensitivity analysis.
    R0 = init_R0(n_i, Tobs, Qarray, alpha)
    acceptance_set = set(range(K)) - set(R0)
    rejection_set = set()  
    
    worst_pvalues = np.zeros(K)
    for k0 in range(K):
        worst_pvalue = worst_pval(I = I, n_i = n_i, q = Qarray[:,:,k0], Gamma = Gamma, t = Tobs[k0], alternative = alternative)
        worst_pvalues[k0] = worst_pvalue
        if worst_pvalue <= alpha/K:
            rejection_set.add(k0)
        elif worst_pvalue > alpha:
            acceptance_set.add(k0)

    indices = cR - acceptance_set - rejection_set
    
    optcalled = 0
    while indices:
        k0 = indices.pop()
        optcalled += 1
        k0_accepted = False
        
        nrejected = set(range(K)) - rejection_set
        v_up = len(nrejected)
        v_lo = math.floor(alpha / worst_pvalues[k0]) + 1
        for v in range(v_lo, v_up + 1):
            cand_J = {k for k in nrejected if worst_pvalues[k] > alpha / v}
            if len(cand_J) < v:
                continue
            # Create combinations of size r-1 from the set s excluding k0
            s_withouTobs = cand_J - {k0}
            subsets = [combo for combo in combinations(s_withouTobs, v - 1)]
        
            # Add k0 back to each combination
            subsets = [{k0} | set(combo) for combo in subsets]
            subsets = [np.array(list(subset), dtype=int) for subset in subsets]
            
            
            for subset in subsets:
                # Call the function
                rej = solve_gnull(I, len(subset), n_i, Qarray[:,:,np.array(subset)], Tobs[np.array(subset)], criticalval = alpha/len(subset), Gamma = Gamma, OutputFlag = OutputFlag)
                if not rej:
                    indices.difference_update(set(subset))
                    k0_accepted = True
                    break

            if k0_accepted:
                break
        if not k0_accepted:
            rejection_set.add(k0)
    rejection_set = rejection_set.intersection(cR)
    
    return rejection_set, optcalled


def fast_closed_sensitivity(index, Qmat, Z, cR = None, alpha = 0.05, alternative="TS", Gamma = 1, OutputFlag=0):

    I, K, n_i, Qarray, Tobs = data_process(index, Qmat, Z)
    if cR == None:
        cR = set(range(K))
    else:
        cR = set(cR)
        
    ## screening
    # Initialize rho_ij array with zeros
    R0 = init_R0(n_i, Tobs, Qarray, alpha)
    
    # Initialize the acceptance set and rejection set
    acceptance_set = set(range(K)) - set(R0)
    rejection_set = set()  
    


    worst_pvalues = np.zeros(K)
    for k0 in range(K):
        worst_pvalue = worst_pval(I = I, n_i = n_i, q = Qarray[:,:,k0], Gamma = Gamma, t = Tobs[k0], alternative = alternative)
        worst_pvalues[k0] = worst_pvalue
        if worst_pvalue <= alpha/K:
            rejection_set.add(k0)
        elif worst_pvalue > alpha:
            acceptance_set.add(k0)
            
            
    indices = cR - acceptance_set - rejection_set
    
    optcalled = 0
    while indices:
        this_k = indices.pop()
        optcalled += 1
        rej, others_acpts = solve_vR_singleton(I = I, K = K, k0 = this_k, n_i = n_i, Qarray = Qarray, Tobs = Tobs, rejected = rejection_set, worst_pvalues = worst_pvalues, alpha = alpha, Gamma = Gamma, OutputFlag = OutputFlag)
        if rej:
            rejection_set.add(this_k)
        else:
            indices.difference_update(set(others_acpts))
    rejection_set = rejection_set.intersection(cR)                
    
    return rejection_set, optcalled


def IPcalled(index, Qmat, Z, alpha = 0.05, alternative="TS", Gamma = 1, OutputFlag=0, progm = 0):

    I, K, n_i, Qarray, Tobs = data_process(index, Qmat, Z)
    
    ## screening
    # Initialize rho_ij array with zeros
    R0 = init_R0(n_i, Tobs, Qarray, alpha)
    
    # Initialize the acceptance set and rejection set
    acceptance_set = set(range(K)) - set(R0)
    rejection_set = set()  

    R_star = np.zeros(K)
    for k0 in R0:
        worst_pvalue = worst_pval(I = I, n_i = n_i, q = Qarray[:,:,k0], Gamma = Gamma, t = Tobs[k0], alternative = alternative)
        if worst_pvalue <= alpha/K:
            rejection_set.add(k0)
        elif worst_pvalue > alpha:
            acceptance_set.add(k0)
        else:
            r_star = math.floor(alpha / worst_pvalue)
            R_star[k0] = r_star
            
    indices = set(range(K)) - acceptance_set - rejection_set
    
                    
    return len(indices) > 0



def generalized_sensitivity_value(cR, index, Qmat, Z,
                                  alpha=0.05,
                                  loGamma=1.0, upGamma=3.0,
                                  precision=None,
                                  thresholds=None,
                                  alternative="TS",
                                  OutputFlag=0):
    # 1) Setup thresholds
    if thresholds is None:
        thresholds = np.arange(len(cR))
    else:
        thresholds = np.array(thresholds)
    sorted_idx = np.argsort(thresholds)
    sorted_thr = thresholds[sorted_idx]
    gSvals     = np.full_like(thresholds, upGamma, dtype=float)

    # 2) One‐time data processing
    I, K, n_i, Qarray, Tobs = data_process(index, Qmat, Z)

    def compute_vR(G):
        wp = np.array([
            worst_pval(I=I, n_i=n_i, q=Qarray[:,:,k], Gamma=G, t=Tobs[k],
                       alternative=alternative)
            for k in range(K)
        ])
        rej = holm_bonferroni(wp, alpha=alpha)
        return solve_vR(I, K, cR, n_i,
                        Qarray=Qarray, Tobs=Tobs,
                        rejected=rej,
                        worst_pvalues=wp,
                        alpha=alpha,
                        Gamma=G,
                        OutputFlag=OutputFlag)

    # 3) Special‐case at loGamma
    vR_lo = compute_vR(loGamma)
    for i, thr in enumerate(sorted_thr):
        if thr < vR_lo:
            gSvals[sorted_idx[i]] = loGamma
    if np.all(sorted_thr < vR_lo):
        return gSvals

    # 4) Decide steps
    if precision is not None:
        # single‐pass
        fine_step   = precision
        coarse_step = None
    else:
        # two‐phase
        fine_step   = 0.01
        coarse_step = 0.1

    # 5) Outer loop
    if coarse_step is None:
        grid = np.arange(loGamma + fine_step, upGamma + fine_step/2, fine_step)
    else:
        grid = np.arange(loGamma + coarse_step, upGamma + coarse_step/2, coarse_step)

    for Gamma in grid:
        vR = compute_vR(Gamma)

        # update any threshold not yet assigned
        for i, thr in enumerate(sorted_thr):
            orig = sorted_idx[i]
            if gSvals[orig] == upGamma and vR > thr:
                # if two‐phase, refine
                if coarse_step is not None:
                    low = max(loGamma, Gamma - coarse_step)
                    for Gref in np.arange(low, Gamma + fine_step/2, fine_step):
                        if compute_vR(Gref) > thr:
                            gSvals[orig] = Gref
                            break
                    else:
                        gSvals[orig] = Gamma
                else:
                    # single‐pass: first Gamma where vR>thr
                    gSvals[orig] = Gamma

        # stop if all thresholds assigned
        if np.all(gSvals != upGamma):
            break

    return gSvals
