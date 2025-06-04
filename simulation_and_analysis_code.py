import numpy as np

# Distribution generation functions
def generate_distribution(dist_type, mean, std, size):
    if dist_type == 'uniform':
        # Adjust uniform to match mean and std
        width = np.sqrt(12 * std**2)
        center = mean
        return np.random.uniform(center - width/2, center + width/2, size)
    elif dist_type == 'normal':
        return np.random.normal(mean, std, size)
    elif dist_type == 'negative_binomial':
        # Convert mean and std to negative binomial parameters
        # For negative binomial: mean = r * (1-p)/p, var = r(1-p)/p^2
        # where r is number of failures, p is probability of success
        if mean <= 0: # set floor
            mean = 0.01
        
        if mean < 1:
            p = 1 / (1 + std**2 / mean)
            r = mean * p / (1 - p)
            sample_counts = np.random.negative_binomial(r, p, size)
            sample_counts = np.clip(sample_counts, 0, 10) # cap the values between 0 and 10, otherwise it'll occassionally have very high values like 99
        else:
            p = 1 / (1 + std**2 / mean)
            r = mean * p / (1 - p)
            sample_counts = np.random.negative_binomial(r, p, size)

        return sample_counts
    else:
        raise ValueError(f"Unsupported distribution: {dist_type}")
        
def generate_distributions(dist_type, pre_mean, pre_sd, n_pre, post_mean, post_sd, n_post):
    pre = generate_distribution(dist_type, pre_mean, pre_sd, n_pre)
    post = generate_distribution(dist_type, post_mean, post_sd, n_post)
    return pre, post

def calculate_pre_sd(sigma_epsilon):
    return sigma_epsilon

def calculate_post_sd(sigma_epsilon):
    return np.sqrt(sigma_epsilon**2)

def mad(data):
    """Median Absolute Deviation (MAD) as a robust measure of spread."""
    return np.median(np.abs(data - np.median(data)))

def hodges_lehmann(x, y):
    pairwise_diffs = np.array([yi - xi for xi in x for yi in y])
    hl_estimate = np.median(pairwise_diffs)
    return hl_estimate


def bootstrap_hodges_lehmann(x, y, num_bootstrap):
    bootstrap_estimates = []
    n_x = len(x)
    n_y = len(y)
    
    for _ in range(num_bootstrap):
        sample_x = np.random.choice(x, size=n_x, replace=True)
        sample_y = np.random.choice(y, size=n_y, replace=True)
        hl_estimate = hodges_lehmann(sample_x, sample_y)
        bootstrap_estimates.append(hl_estimate)
    
    
    observed_effect = np.median(bootstrap_estimates)
    ci_lower, ci_upper = np.percentile(bootstrap_estimates, [2.5, 97.5])
    
    return observed_effect, ci_lower, ci_upper

def analyse_treatment(pre, post, confidence_level, n_bootstrap):
    """
    Analyse pre- and post-treatment data using bootstrap resampling.
    
    Parameters:
        pre (list or array): Pre-treatment data.
        post (list or array): Post-treatment data.
        confidence_level (float): Confidence level for the bootstrap CI (default: 0.95).
        n_bootstrap (int): Number of bootstrap samples to generate (default: 10,000).
    
    Returns:
        dict: Analysis results in intuitive format.
    """
    pre = np.array(pre)
    post = np.array(post)
    
    observed_effect, ci_lower, ci_upper = bootstrap_hodges_lehmann(pre, post, n_bootstrap)
    
    # Determine significance
    significant_increase = ci_lower > 0
    significant_decrease = ci_upper < 0
    result_text = "no significant change"
    if significant_increase:
        result_text = "significant increase"
    elif significant_decrease:
        result_text = "significant decrease"
    
    # Results
    result = {
        'result': f"{result_text} of {observed_effect:.2f} (95% CI: {ci_lower:.2f}, {ci_upper:.2f})",
        'observed_difference': observed_effect,
        'confidence_interval': (ci_lower, ci_upper),
        # 'bootstrap_distribution': bootstrap_diffs
    }
    
    return result

def simulate_treatment_effect(pre_mean, sigma_epsilon, n_pre, n_post, dist_type, effect_sizes, effect_direction, n_simulations, n_bootstrap, simulations,include_severity_composite, confidence_level=0.95, clip_to_zero=True):
    """
    Simulate treatment effects and evaluate detection reliability.
    
    Parameters:
        pre_mean (float): Mean of pre-treatment data.
        pre_sd (float): Standard deviation of pre-treatment data.
        n_pre (int): Sample size for pre-treatment group.
        n_post (int): Sample size for post-treatment group.
        effect_sizes (list): List of effect sizes (Cohen's d) to simulate.
        confidence_level (float): Confidence level for bootstrap CI.
        n_simulations (int): Number of simulations to run per effect size.
        n_bootstrap (int): Number of bootstrap samples in each analysis.
        clip_to_zero (boolean): set negative values to 0 e.g. for count data
    
    Returns:
        dict: Simulation results with detection rates and effect size estimates.
    """
    proportions_increase = []
    proportions_decrease = []
    all_effect_size_estimates = []
    all_observed_diff_estimates = []
    pre_sd = sigma_epsilon
        
    for d in effect_sizes:
        if effect_direction == "decrease":
            post_mean = pre_mean - d * pre_sd  # Adjust post-mean for each effect size
        elif effect_direction == "increase":
            post_mean = pre_mean + d * pre_sd  # Adjust post-mean for each effect size
        
        post_sd = calculate_post_sd(sigma_epsilon) # Calculate post-treatment standard deviation - based upon both random variance and treatment effect variance
            
        significant_increases = 0
        significant_decreases = 0
        effect_size_estimates = []
        observed_diff_estimates = []

        for _ in range(n_simulations):
            
            pre, post = generate_distributions(dist_type, pre_mean, pre_sd, n_pre, post_mean, post_sd, n_post)
            
            if clip_to_zero:
                pre = np.clip(pre, 0, 30)  # Set negative values to 0 and cap at 30 days/month
                post = np.clip(post, 0, 30)  # Set negative values to 0
            
            # if including consideration of headache severity
            if include_severity_composite == 1:
                pre_sev_mean = 8; pre_sev_sd = 3 # 8, 2
                pre_dur_mean = 8; pre_dur_sd = 3
                
                if d == 0: # if it's an ineffective treatment, make it so severity also doesn't change
                    post_sev_mean = pre_sev_mean
                    post_sev_sd = pre_sev_sd
                    post_dur_mean = pre_dur_mean
                    post_dur_sd = pre_dur_sd
                else:
                    post_sev_mean = 4; post_sev_sd = 3 # 4, 2
                    post_dur_mean = 4; post_dur_sd = 3
                # sev_dist = "normal" # how severity values are distributed
                
                pre_sev = []
                for month in pre:
                    severity_vals = np.clip(np.round(np.random.normal(pre_sev_mean, pre_sev_sd, month)),0,10) # severity values rounded to nearest integer and between 0-10
                    # duration_vals = np.clip(np.round(np.random.normal(pre_dur_mean, pre_dur_sd, month)),0,None) # severity values rounded to nearest integer and between 0-10
                    # severity_vals = [a * b for a, b in zip(severity_vals, duration_vals)]
                    total_burden = np.sum(severity_vals)
                    pre_sev.append(total_burden)
                    
                post_sev = []
                for month in post:
                    severity_vals = np.clip(np.round(np.random.normal(post_sev_mean, post_sev_sd, month)),0,10) # severity values rounded to nearest integer and between 0-10
                    # duration_vals = np.clip(np.round(np.random.normal(post_dur_mean, post_dur_sd, month)),0,None) # severity values rounded to nearest integer and between 0-10
                    # severity_vals = [a * b for a, b in zip(severity_vals, duration_vals)]
                    total_burden = np.sum(severity_vals)
                    post_sev.append(total_burden)
                
                pre = pre_sev
                post = post_sev
            
            # Analyse the simulated data
            result = analyse_treatment(pre, post, confidence_level, n_bootstrap)

            # Track significance
            if result['confidence_interval'][0] > 0:
                significant_increases += 1
            if result['confidence_interval'][1] < 0:
                significant_decreases += 1

            # Track effect size estimates
            observed_diff = result['observed_difference']
            
            # Calculate pooled sd of the two samples using mad which doesn't assume normal distrobution (unlike sd)
            mad_pre = mad(pre)
            mad_post = mad(post)
            pooled_sd = np.sqrt(((n_pre - 1) * mad_pre**2 + (n_post - 1) * mad_post**2) / (n_pre + n_post - 2))
            
            observed_diff_estimates.append(observed_diff)
            effect_size = observed_diff / pooled_sd
            effect_size_estimates.append(effect_size)
        
        # Proportions of significant results
        proportions_increase.append(significant_increases / n_simulations)
        proportions_decrease.append(significant_decreases / n_simulations)
        all_effect_size_estimates.append(effect_size_estimates)
        all_observed_diff_estimates.append(observed_diff_estimates)
        
    simulations = {
        'proportions_increase': proportions_increase,
        'proportions_decrease': proportions_decrease,
        'effect_size_estimates': all_effect_size_estimates,
        'all_observed_diff_estimates': all_observed_diff_estimates
        }
    
    return  simulations

# Specify paramaters
pre_mean = 4
sigma_epsilon = 3 # Random error standard deviation - level of variability due to random factors i.e. day-to-day fluctuations e.g. due to factors like stress, physical activity, or measurement inaccuracies
dist_type = "negative_binomial"
effect_sizes = [0, 2/3] # effect size in terms of the SD e.g. for SD 3 and effect size 2/3 it's modelling a reduction of 2
effect_direction = "decrease" # "increase" if the effect is to decrease or an increase a value
include_severity_composite = 0

n_pre = 12
n_post = 12

n_simulations = 200
n_bootstrap = 100

conditions_key = ", ".join(map(str, (n_pre, n_post, pre_mean, sigma_epsilon, dist_type)))
simulations = {}
simulations = simulate_treatment_effect(pre_mean, sigma_epsilon, n_pre, n_post, dist_type, effect_sizes, effect_direction, n_simulations, n_bootstrap, simulations, include_severity_composite)
