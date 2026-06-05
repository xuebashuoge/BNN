import numpy as np
from scipy.optimize import minimize_scalar

def kl_div(x, y):
    # Add epsilon to prevent log(0)
    y = np.clip(y, 1e-9, 1 - 1e-9)
    x = np.clip(x, 1e-9, 1 - 1e-9)
    return x * np.log(x / y) + (1 - x) * np.log((1 - x) / (1 - y))

def optimize_p(c, sigma_art, m, sigma, n, epsilon):
    theta = [0.5, 0.75, 0.9]
    
    def objective_F(p):
        # Calculate T1
        T1 = kl_div(c, p)
        
        # Calculate q(p) and T3
        q_p = -0.1 * p**2 + 0.5 * p + 0.5
        T3 = kl_div(q_p, 0.5)
        
        # Calculate T2
        # Assuming standard binomial coefficients for k=0, 1, 2
        import math
        T2 = 0
        for k in range(3):
            # Using math.comb(m, k) assuming m is an integer >= 2
            weight = math.comb(int(m), k) * (p**k) * ((1 - p)**(m - k))
            T2 += weight * kl_div(theta[k], q_p)
            
        # Assemble F
        term1 = np.sqrt(2 * sigma_art**2 * T1)
        term2 = np.sqrt((2 * sigma_art**2 * T2) / m)
        term3 = np.sqrt((2 * sigma**2 * (T3 + np.log(np.sqrt(n)) / epsilon)) / (n - 1))
        
        return term1 + term2 + term3

    # Optimize bounded strictly between (0, 1) to avoid log domain errors
    result = minimize_scalar(objective_F, bounds=(1e-5, 1 - 1e-5), method='bounded')
    
    if result.success:
        return result.x, result.fun
    else:
        raise ValueError("Optimization failed.")

# Example usage:
optimal_p, min_F = optimize_p(c=0.8, sigma_art=0.1, m=20, sigma=1.0, n=20, epsilon=0.1)

print(f"Optimal p: {optimal_p}, Minimum F: {min_F}")