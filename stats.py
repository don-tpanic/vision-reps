import numpy as np

def compare_distributions_mean_diff_and_bootstrap_ci(dist1, dist2, n_bootstrap=100, ci_level=0.95, random_state=None):
    """
    Compare two distributions by computing mean difference and bootstrapped CI.
    
    Parameters:
    -----------
    dist1, dist2 : array-like
        The two distributions to compare
    n_bootstrap : int, default=100
        Number of bootstrap samples
    ci_level : float, default=0.95
        Confidence level for the interval
    random_state : int or None, default=None
        Random seed for reproducibility
    
    Returns:
    --------
    dict
        Contains mean difference and confidence interval bounds
    """
    import numpy as np
    
    if random_state is not None:
        np.random.seed(random_state)
    
    # Calculate observed mean difference
    mean_diff = np.mean(dist1) - np.mean(dist2)
    
    # Bootstrap sampling
    n1, n2 = len(dist1), len(dist2)
    bootstrap_diffs = np.zeros(n_bootstrap)
    
    for i in range(n_bootstrap):
        # Sample with replacement from both distributions
        sample1 = np.random.choice(dist1, size=n1, replace=True)
        sample2 = np.random.choice(dist2, size=n2, replace=True)
        
        # Calculate and store mean difference for this bootstrap sample
        bootstrap_diffs[i] = np.mean(sample1) - np.mean(sample2)
    
    # Calculate confidence interval
    alpha = 1 - ci_level
    ci_lower = np.percentile(bootstrap_diffs, alpha/2 * 100)
    ci_upper = np.percentile(bootstrap_diffs, (1 - alpha/2) * 100)
    
    return {
        'mean_diff': mean_diff,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper
    }