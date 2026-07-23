import gurobipy as gp
from gurobipy import GRB
from gurobipy import Model, GRB, quicksum
import numpy as np
import pandas as pd
from scipy.stats import t
import collections
from itertools import product
import random
from tqdm import tqdm
import pickle
import sys
import csv
import time
import matplotlib.pyplot as plt
from scipy.stats import median_abs_deviation as mad
from scipy.stats import multivariate_normal as mvn
from scipy.stats import chi2
import sensitivityMCP
from statistics import mean
import importlib
from itertools import combinations
import itertools
from math import log
from sensitivityMCP import holm_bonferroni, closed_sensitivity, gnullsensitivity, fast_closed_sensitivity, IPcalled, worst_pval, generalized_sensitivity_value, data_process, solve_vR, vR_given_pvals
from generate_data import generate_pair_data
from PO_to_Qmat import PO_to_Qmat_DiM, PO_to_Qmat_Mstats


## Testing the equivalence of closed testing and the fast_closed testing.
def closed_testing_equi(nsim, K, I, btau_list, Sigma_list, Gamma_list, Gamma_bias = 1, alpha = 0.05, alternative = "TS", seed = 0):
    
    # Create all possible combinations of settings
    settings = list(product(btau_list, Sigma_list, Gamma_list))
    nsettings = len(settings)

    # Initialize power counters for each method
    power_closed = [[] for _ in range(nsettings)]
    time_closed = [[] for _ in range(nsettings)]
    cond_time_closed = [[] for _ in range(nsettings)]
    power_fast = [[] for _ in range(nsettings)]
    time_fast = [[] for _ in range(nsettings)]
    cond_time_fast = [[] for _ in range(nsettings)]
    called = [[] for _ in range(nsettings)]
    propcalled = [[] for _ in range(nsettings)]
    
    for ss in tqdm(range(nsettings), desc="Processing settings"):
        setting = settings[ss]
        btau, Sigma, Gamma = setting
        np.random.seed(seed) 
        for mm in tqdm(range(nsim), desc=f"Simulation for setting #{ss}", leave=False, position=1):
            Z, PO, index = generate_pair_data(I = I, btau = btau, Sigma = Sigma, Gamma_bias = Gamma_bias)
            treatment = (Z==1)
            Qmat = np.full_like(PO, np.nan, dtype=float)
            for k in range(K):
                Qmat[:,k] = PO_to_Qmat_Mstats(PO = PO[:,k], I = I, index = index, trim=2.5, qu=0.5, TonT=False)
            start = time.perf_counter()
            try:
                clo_rejs, optcalled = closed_sensitivity(index = index, Qmat = Qmat, Z = treatment, alpha = alpha, alternative = alternative, Gamma= Gamma, OutputFlag =0)
                end = time.perf_counter()
                clo_time = end - start
                
            except Exception as e:
                # If closed_sensitivity throws due to numerical issues or any other error:
                end = time.perf_counter()
                clo_time = end - start
                # Log or handle the error
                print(f"Solver failed with error: {e}")
                # Optionally set clo_rejs, optcalled to NaN or something indicative of failure
                clo_rejs = np.nan
                optcalled = np.nan

            # Repeat for the other function calls
            start = time.perf_counter()
            fast_rejs, IPcalled = fast_closed_sensitivity(index = index, Qmat = Qmat, Z = treatment, alpha = alpha, alternative = alternative, Gamma= Gamma, OutputFlag =0)
            end = time.perf_counter()
            fast_time = end - start

                    
            power_closed[ss].append(len(clo_rejs)/K)
            time_closed[ss].append(clo_time)
            power_fast[ss].append(len(fast_rejs)/K)
            time_fast[ss].append(fast_time)
            called[ss].append(int(optcalled > 0))
            propcalled[ss].append(optcalled/K)
            
            if optcalled > 0:
                cond_time_closed[ss].append(clo_time)
                cond_time_fast[ss].append(fast_time)
                
            
    # Average of power_closed
    avg_power_closed = [mean(lst) for lst in power_closed]

    # Average of time_closed
    avg_time_closed = [mean(lst) for lst in time_closed]
    
    # Average of power_fast
    avg_power_fast = [mean(lst) for lst in power_fast]

    # Average of time_fast
    avg_time_fast = [mean(lst) for lst in time_fast]

#     # Average of cond_time_closed
#     avg_cond_time_closed = [mean(lst) for lst in cond_time_closed]

#     # Average of cond_time_fast
#     avg_cond_time_fast = [mean(lst) for lst in cond_time_fast]
    avg_cond_time_closed = []
    avg_cond_time_fast = []
    
    for lst in cond_time_closed:
        if len(lst) > 0:
            avg_cond_time_closed.append(mean(lst))
        else:
            avg_cond_time_closed.append(None)  # or any other default value you prefer
            
    for lst in cond_time_fast:
        if len(lst) > 0:
            avg_cond_time_fast.append(mean(lst))
        else:
            avg_cond_time_fast.append(None)  # or any other default value you prefer
            
            
    avg_called = [mean(lst) for lst in called]
    avg_propcalled = [mean(lst) for lst in propcalled]
    
    return avg_power_closed, avg_time_closed, avg_power_fast, avg_time_fast, avg_cond_time_closed, avg_cond_time_fast, avg_called, avg_propcalled
    
    


def optcall_expr(nsim, K_list, I_list, Gamma = 1.5, alpha = 0.05, alternative = "TS", seed = 0):
        
    # Create all possible combinations of settings
    settings = list(product(K_list, I_list))
    nsettings = len(settings)    
    
    # power & time initialization
    called = [[] for _ in range(nsettings)]
    propcalled = [[] for _ in range(nsettings)]
    
    for ss in tqdm(range(nsettings), desc="Processing settings"):
        setting = settings[ss]
        K, I = setting
        ## mu, Sigma specified after K.
        btau = np.array([0.3] * ((K // 2) + (K % 2)) + [0] * (K // 2))
        Sigma = np.eye(K)
        np.random.seed(seed) 
        for mm in tqdm(range(nsim), desc=f"Simulation for setting #{ss}", leave=False, position=1):
            Z, PO, index = generate_pair_data(I = I, btau = btau, Sigma = Sigma, Gamma_bias = 1)
            treatment = (Z==1)
            Qmat = np.full_like(PO, np.nan, dtype=float)
            for k in range(K):
                Qmat[:,k] = PO_to_Qmat_Mstats(PO = PO[:,k], I = I, index = index, trim=2.5, qu=0.5, TonT=False)
            I, K, n_i, Qarray, Tobs = data_process(index = index, Qmat = Qmat, Z = treatment)
            acceptance_set = set() 
            rejection_set = set()  
            worst_pvalues = np.zeros(K)
            for k0 in range(K):
                worst_pvalue = worst_pval(I = I, n_i = n_i, q = Qarray[:,:,k0], Gamma = Gamma, t = Tobs[k0], alternative = alternative)
                worst_pvalues[k0] = worst_pvalue
                if worst_pvalue <= alpha/K:
                    rejection_set.add(k0)
                elif worst_pvalue > alpha:
                    acceptance_set.add(k0)

            nscreen = set(range(K)) - acceptance_set - rejection_set
            num_nscreen = len(nscreen)
            
            called[ss].append(int(num_nscreen>0))
            propcalled[ss].append(num_nscreen/K)

    # Average of called
    avg_called = [mean(lst) for lst in called]

    # Average of propcalled
    avg_propcalled = [mean(lst) for lst in propcalled]
    
    return avg_called, avg_propcalled


def compare_vR_naive_vs_exact_expr(K, I, btau_list, Sigma_list, Gamma_list, 
                                   Gamma_bias=1, alpha=0.05, alternative="TS", 
                                   seed=0, nsim=10):
    
    # Create all possible combinations of settings.
    settings = list(product(btau_list, Sigma_list, Gamma_list))
    nsettings = len(settings)

    # These will store the average vR values over nsim simulations for each setting.
    vR_exact = [None] * nsettings
    vR_naive = [None] * nsettings
    
    # Iterate over each setting.
    for ss in tqdm(range(nsettings), desc="Processing settings"):
        setting = settings[ss]
        btau, Sigma, Gamma = setting
        
        # Lists to collect simulation results for the current setting.
        sim_vR_exact = []
        sim_vR_naive = []
        
        np.random.seed(seed)
        for mm in tqdm(range(nsim), desc=f"Simulation for setting #{ss}", leave=False, position=1):
            
            # Generate the data.
            Z, PO, index = generate_pair_data(I=I, btau=btau, Sigma=Sigma, Gamma_bias=Gamma_bias)
            treatment = (Z == 1)
            
            # Build Qmat from PO for each outcome.
            Qmat = np.full_like(PO, np.nan, dtype=float)
            for k in range(K):
                Qmat[:, k] = PO_to_Qmat_Mstats(PO=PO[:, k], I=I, index=index, trim=2.5, qu=0.5, TonT=False)
            
            # Process data.
            I, K, n_i, Qarray, Tobs = data_process(index=index, Qmat=Qmat, Z=treatment)
            cR = set(range(K))
            
            # Compute worst p-values for each outcome.
            worst_pvalues = np.zeros(K)
            for k in range(K):
                worst_pvalues[k] = worst_pval(I=I, n_i=n_i, q=Qarray[:, :, k],
                                               Gamma=Gamma, t=Tobs[k], alternative=alternative)
            # Holm-Bonferroni rejection on the worst p-values.
            rejected = holm_bonferroni(worst_pvalues, alpha=alpha)
            
            # Compute vR via the exact method.
            vR1 = solve_vR(I=I, K=K, cR=cR, n_i=n_i, Qarray=Qarray, Tobs=Tobs,
                           rejected=rejected, worst_pvalues=worst_pvalues, alpha=alpha,
                           Gamma=Gamma, OutputFlag=0)
            # Compute vR via the naive method.
            vR2 = vR_given_pvals(worst_pvalues, cR, alpha)
            
            sim_vR_exact.append(vR1)
            sim_vR_naive.append(vR2)
        
        vR_exact[ss] = sim_vR_exact
        vR_naive[ss] = sim_vR_naive
    
    return vR_exact, vR_naive
    




def solve_vR_singleton_equi(nsim, K, I, btau_list, Sigma_list, Gamma_list, Gamma_bias=1, alpha=0.05, alternative="TS", seed=0):
    
    # Create all possible combinations of settings.
    settings = list(product(btau_list, Sigma_list, Gamma_list))
    nsettings = len(settings)

    # Initialize lists to store results for each setting.
    time_solver1 = [[] for _ in range(nsettings)]
    time_solver2 = [[] for _ in range(nsettings)]
    
    # List to store simulation details when the two solvers' rejection sets differ.
    mismatch_info = []
    
    for ss in tqdm(range(nsettings), desc="Processing settings"):
        setting = settings[ss]
        btau, Sigma, Gamma = setting
        np.random.seed(seed) 
        for mm in tqdm(range(nsim), desc=f"Simulation for setting #{ss}", leave=False, position=1):
            Z, PO, index = generate_pair_data(I=I, btau=btau, Sigma=Sigma, Gamma_bias=Gamma_bias)
            treatment = (Z == 1)
            Qmat = np.full_like(PO, np.nan, dtype=float)
            for k in range(K):
                Qmat[:, k] = PO_to_Qmat_Mstats(PO=PO[:, k], I=I, index=index, trim=2.5, qu=0.5, TonT=False)
            
            # Process data.
            start = time.perf_counter()
            I_proc, K_proc, n_i, Qarray, Tobs = data_process(index, Qmat, Z)
            # (Assuming I_proc and K_proc are the same as I and K or updated accordingly.)
            cR = set(range(K))
            
            # Initialize local rejection sets for the two solvers.
            rejs_solver1_local = []
            rejs_solver2_local = []
            
            # Solve with solve_vR_singleton.
            start_time = time.perf_counter()
            for k in range(K):
                rej1, _ = solve_vR_singleton(I=I_proc, K=K_proc, k0=k, n_i=n_i, Qarray=Qarray, Tobs=Tobs,
                                             rejected=rejection_set, worst_pvalues=worst_pvalues, alpha=alpha,
                                             Gamma=Gamma, OutputFlag=OutputFlag)
                if rej1:
                    rejs_solver1_local.append(k)
            end_time = time.perf_counter()
            solver1_time = end_time - start_time
            
            # Solve with solve_vR.
            start_time = time.perf_counter()
            for k in range(K):
                # Note: use {k} instead of set(k) to create a set with the single element k.
                rej2, _ = solve_vR(I=I_proc, K=K_proc, cR={k}, n_i=n_i, Qarray=Qarray, Tobs=Tobs,
                                   rejected=rejection_set, worst_pvalues=worst_pvalues, alpha=alpha,
                                   Gamma=Gamma, OutputFlag=OutputFlag)
                if rej2:
                    rejs_solver2_local.append(k)
            end_time = time.perf_counter()
            solver2_time = end_time - start_time
            
            # Compare the rejection sets.
            if set(rejs_solver1_local) == set(rejs_solver2_local):
                time_solver1[ss].append(solver1_time)
                time_solver2[ss].append(solver2_time)
            else:
                # Save simulation information for this mismatch.
                mismatch_info.append({
                    'setting': setting,
                    'simulation_index': mm,
                    'rejs_solver1': rejs_solver1_local,
                    'rejs_solver2': rejs_solver2_local,
                    'solver1_time': solver1_time,
                    'solver2_time': solver2_time,
                    'n_i': n_i,
                    'Qarray': Qarray,
                    'Tobs': Tobs
                })
    
    # Compute averages using a mean function (assumed to be imported, e.g., from statistics import mean)
    avg_time_solver1 = [mean(lst) for lst in time_solver1]
    avg_time_solver2 = [mean(lst) for lst in time_solver2]
    
    return avg_time_solver1, avg_time_solver2, mismatch_info



def pseudo_vR_sensitivity_expr(K, I, btau_list, Sigma_list, Gamma_list, alpha = 0.05, alternative = "TS", seed = 0):
    
    # Create all possible combinations of settings
    settings = list(product(btau_list, Sigma_list, Gamma_list))
    nsettings = len(settings)

    # Initialize power counters for each method
    power_screen = [[] for _ in range(nsettings)]
    time_screen = [[] for _ in range(nsettings)]
    power_nscreen = [[] for _ in range(nsettings)]
    time_nscreen = [[] for _ in range(nsettings)]
    
    for ss in tqdm(range(nsettings), desc="Processing settings"):
        setting = settings[ss]
        mu, Sigma, Gamma = setting

        np.random.seed(seed)
        Z, PO, index = generate_pair_data(I = I, btau = btau, Sigma = Sigma, Gamma_bias = Gamma_bias)
        treatment = (Z==1)
        
        Qmat = np.full_like(PO, np.nan, dtype=float)
        for k in range(K):
            Qmat[:,k] = PO_to_Qmat_DiM(PO[:,k], I, index)

        for r in range(1, K+1, 1):
            # random sample subset of size r from [K]
            cR = random.sample(range(K),r)

            start = time.perf_counter()
            vR_screen = vR_sensitivity(cR, index, Q = PO , Z = treatment, alpha = alpha, alternative = alternative, Gamma= Gamma, OutputFlag =0, screen = 1)
            end = time.perf_counter()
            screen_time = end - start

            start = time.perf_counter()
            vR_nscreen = pseudo_vR_sensitivity(cR, index, Q = PO , Z = treatment, alpha = alpha, alternative = alternative, Gamma= Gamma, OutputFlag =0)
            end = time.perf_counter()
            nscreen_time = end - start
        
            power_screen[ss].append(vR_screen)
            time_screen[ss].append(screen_time)
            power_nscreen[ss].append(vR_nscreen)
            time_nscreen[ss].append(nscreen_time)
    
    return power_screen, time_screen, power_nscreen, time_nscreen

   
    




def subsets_compete(I, btau, Sigma_list, Gamma_bias_list, subsetsize=2, precision=0.01, 
                      alpha=0.05, OutputFlag=0, seed=0, nsim=10, alternative="TS"):
    """
    Run simulation experiments over various settings (each setting defined by a Sigma and Gamma_bias).

    For each setting (given by corresponding elements of Sigma_list and Gamma_bias_list):
      - Run nsim simulation runs (using the same seed for reproducibility within a setting).
      - In each run, generate a dataset via generate_pair_data().
      - Build Qmat for each outcome using PO_to_Qmat_Mstats().
      - Process the data with data_process() and compute:
          * pvals0: p-values (one for each outcome, with Gamma fixed to 1).
          * gSvals0 and gSvals1: generalized sensitivity values for each subset of outcomes of size subsetsize.
    
    Parameters:
      I             : int
                      Sample size.
      btau          : array-like
                      Vector of effect sizes; its length defines the number of outcomes K.
      Sigma_list    : list of numpy arrays
                      Each is a covariance matrix for a given simulation setting.
      Gamma_bias_list: list
                      List of Gamma_bias values corresponding to each setting.
      subsetsize    : int, default 2
                      Size of the subset (number of outcomes) to consider when computing generalized sensitivity values.
      precision      : float, default 0.01
                      Step size used in generalized sensitivity value computation.
      alpha         : float, default 0.05
                      Significance level.
      OutputFlag    : int, default 0
                      Flag to control the output of the generalized sensitivity function.
      seed          : int, default 0
                      Seed for random number generation (set identically for each simulation run in a setting).
      nsim          : int, default 10
                      Number of simulation runs per setting.
      alternative   : str, default "TS"
                      Alternative hypothesis indicator to pass to processing functions.
    
    Returns:
      A tuple (all_pvals0, all_gSvals0, all_gSvals1) where each element is a list of length nsettings.
      For each setting, the list element is itself a list (length nsim) of NumPy arrays:
        - all_pvals0[ss][mm] contains an array of p-values for the outcomes in simulation mm.
        - all_gSvals0[ss][mm] and all_gSvals1[ss][mm] are arrays of generalized sensitivity values for each subset.
    """

    # Check that the number of settings for Sigma and Gamma_bias agree.
    nsettings = len(Sigma_list)
    assert nsettings == len(Gamma_bias_list), "Sigma_list and Gamma_bias_list must have the same length."

    # Determine number of outcomes from btau.
    K = len(btau)
    # Generate all subsets (combinations) of outcomes of the given size.
    subsets = list(itertools.combinations(range(K), subsetsize))
    nsubsets = len(subsets)
    
    # Initialize nested lists to store simulation results for each setting.
    all_pvals0  = [[] for _ in range(nsettings)]
    all_gSvals0 = [[] for _ in range(nsettings)]
    all_gSvals1 = [[] for _ in range(nsettings)]
    
    # Loop over each simulation setting.
    for ss in tqdm(range(nsettings), desc="Processing settings"):
        Sigma = Sigma_list[ss]
        Gamma_bias = Gamma_bias_list[ss]
        
        # Set the random seed at the start of each setting.
        np.random.seed(seed)
        
        # Run nsim simulation runs for this setting.
        for mm in tqdm(range(nsim), desc=f"Simulation for setting #{ss}", leave=False, position=1):
            # Generate the dataset.
            Z, PO, index = generate_pair_data(I=I, btau=btau, Sigma=Sigma, Gamma_bias=Gamma_bias)
            treatment = (Z == 1)
            
            # Build Qmat: process each outcome's PO into a Q vector.
            Qmat = np.full_like(PO, np.nan, dtype=float)
            for k in range(K):
                Qmat[:, k] = PO_to_Qmat_Mstats(PO=PO[:, k], I=I, index=index, trim=2.5, qu=0.5, TonT=False)
            
            # Process the data.
            I, K, n_i, Qarray, Tobs = data_process(index=index, Qmat=Qmat, Z=treatment)
            # Note: We assume that I and K match I and K, respectively.
            
            # Compute p-values for each outcome.
            pvals0 = np.zeros(K)
            for k in range(K):
                pvals0[k] = worst_pval(I=I, n_i=n_i, q=Qarray[:, :, k],
                                       Gamma=1, t=Tobs[k], alternative=alternative)
            
            # Compute generalized sensitivity values for each subset of outcomes.
            gSvals0 = np.zeros(nsubsets)
            gSvals1 = np.zeros(nsubsets)
            for i, cR in enumerate(subsets):
                gSvals0[i], gSvals1[i] = generalized_sensitivity_value(
                    cR, index, Qmat, Z, alpha=alpha, loGamma=1, upGamma=3,
                    precision=precision, thresholds=None, alternative=alternative, OutputFlag=OutputFlag)
            
            # Record the simulation outputs.
            all_pvals0[ss].append(pvals0)
            all_gSvals0[ss].append(gSvals0)
            all_gSvals1[ss].append(gSvals1)
    
    return all_pvals0, all_gSvals0, all_gSvals1
