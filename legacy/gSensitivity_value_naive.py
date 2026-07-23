from solve_worst_pval import solve_worst_pval

def subset_to_id(subset, K):
    """Convert a set of indices to a unique subset ID in [0, 2^K-1]."""
    return sum(1 << k for k in subset)


def gSensitivity_value_naive(
    index,
    PO,
    Z,
    alpha=0.05,
    loGamma=1,
    upGamma=3,
    stepsize=0.1,
    alternative="TS",
    testStats="DiM",
    OutputFlag=0,
    output_csv="gSensitivity_naive.csv"
):
    """
    Compute the naive gSensitivity_value for *all* subsets of [K] in a single pass of Gamma.
    
    Parameters
    ----------
    index : array-like
        Stratum indices or matching indices.
    PO : array-like, shape (N, K)
        Potential outcomes or outcome measures.
    Z : array-like, shape (N,)
        Treatment indicator.
    alpha : float
        Significance level.
    loGamma : float
        Lower bound for Gamma.
    upGamma : float
        Upper bound for Gamma.
    stepsize : float
        Step size for searching Gamma.
    alternative : str
        Type of alternative ("TS", ...).
    testStats : str
        Type of test statistic ("DiM", ...).
    OutputFlag : int
        If >0, print debug info.
    output_csv : str
        Path to save the output CSV file.
    
    Returns
    -------
    None
        Saves results in a CSV file.
    """
    I, K, n_i, Qmat, Qarray, Tobs = data_process(
        index, PO, Z, alternative=alternative, testStats=testStats
    )

    all_sets = []
    for size in range(1, K+1):
        for combo in combinations(range(K), size):
            all_sets.append(frozenset(combo))

    gSens_naive = {S: None for S in all_sets}

    Gamma_values = np.arange(loGamma, upGamma + stepsize, stepsize)

    subsets_not_failed = set(all_sets)

    for Gamma in Gamma_values:
        worst_pvals = np.zeros(K)
        for k in range(K):
            worst_pvals[k] = solve_worst_pval(
                I = I, 
                n_i = n_i, 
                q_array = Qarray[:,:,k], 
                t = Tobs[k], 
                Gamma=Gamma, 
                tol=1e-7, 
                max_iter=50, 
                OutputFlag=0)
#             worst_pvals[k] = worst_pval(
#                 I=I,
#                 n_i=n_i,
#                 q=Qarray[:, :, k],
#                 Gamma=Gamma,
#                 t=Tobs[k],
#                 alternative=alternative
#             )
        RHB = holm_bonferroni(worst_pvals, alpha=0.05)
    
        # Compute the set of indices not in RHB
        remaining = set(range(K)) - RHB

        # Get all subsets of remaining indices.
        subsets_remaining = [S for S in all_sets if S.issubset(remaining)]

        newly_failed = []
        for S in subsets_remaining:
            if S in subsets_not_failed:
                gSens_naive[S] = Gamma
                newly_failed.append(S)
    
#         newly_failed = []
#         for S in subsets_not_failed:
#             pvals_in_S = [worst_pvals[k] for k in S]
#             if np.min(pvals_in_S) > alpha / len(S):
#                 gSens_naive[S] = Gamma
#                 newly_failed.append(S)
    
                
        for s_failed in newly_failed:
            subsets_not_failed.remove(s_failed)

        if not subsets_not_failed:
            break

    for S in subsets_not_failed:
        gSens_naive[S] = upGamma

    if OutputFlag > 0:
        print("Naive gSensitivity_value computation complete.")

    # Convert to DataFrame and save as CSV
    results = []
    for S, gamma_val in gSens_naive.items():
        subset_id = subset_to_id(S, K)
        results.append([subset_id, list(S), gamma_val])
    
    df_results = pd.DataFrame(results, columns=["subset_id", "chosen_cols", "gSval_naive"])
    df_results = df_results.round(2)
    df_results.to_csv(output_csv, index=False)
    
    if OutputFlag > 0:
        print(f"Results saved to {output_csv}")
    return df_results