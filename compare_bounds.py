import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar

def bound1(n, m, alpha, beta, X, delta):
    """
    Bound 1 (Joint):
    sqrt(alpha/(n-1) + beta/(m-1)) * sqrt(X + log(sqrt(n*m)/delta))
    Confidence: 1 - delta
    """
    term_sample = alpha / (n - 1.0) + beta / (m - 1.0)
    term_log = X + np.log(np.sqrt(n * m) / delta)
    return np.sqrt(term_sample) * np.sqrt(term_log)

def bound2(n, m, alpha, beta, X, eps, eps_prime):
    """
    Bound 2 (Decomposed):
    sqrt(alpha/(n-1) * (X + log(sqrt(n)/eps))) + sqrt(beta/(m-1) * (X + log(sqrt(m)/eps_prime)))
    Confidence: (1 - eps)*(1 - eps_prime) = 1 - (eps + eps_prime - eps*eps_prime)
    """
    t1 = np.sqrt((alpha / (n - 1.0)) * (X + np.log(np.sqrt(n) / eps)))
    t2 = np.sqrt((beta / (m - 1.0)) * (X + np.log(np.sqrt(m) / eps_prime)))
    return t1 + t2

def get_eps_prime(eps, delta):
    """
    Solves (1 - eps)(1 - eps') = 1 - delta => 1 - eps' = (1 - delta)/(1 - eps)
    => eps' = 1 - (1 - delta)/(1 - eps) = (delta - eps) / (1 - eps)
    Requires eps < delta.
    """
    return (delta - eps) / (1.0 - eps)

def main():
    alpha = 1.0
    beta = 1.0
    X = 2.0
    delta = 0.05  # Total failure probability budget (5%)

    # Sample size range for m
    m_range = np.logspace(1, 6, 500)

    # Epsilon allocations for Bound 2
    eps_equal = 0.5 * delta
    eps_prime_equal = get_eps_prime(eps_equal, delta)

    eps_smart_m = 0.99 * delta
    eps_prime_smart_m = get_eps_prime(eps_smart_m, delta)

    # Configure publication quality parameters for IEEE single column figure (width ~3.5 inches)
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 7,
        'axes.labelsize': 8,
        'axes.titlesize': 8.5,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'legend.fontsize': 6.5,
        'lines.linewidth': 1.2,
        'figure.dpi': 300,
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.5, 2.2), sharey=False)

    # Subplot 1: n = 10 (Imbalanced case)
    n1 = 10.0
    b1_n10 = bound1(n1, m_range, alpha, beta, X, delta)
    b2_eq_n10 = bound2(n1, m_range, alpha, beta, X, eps_equal, eps_prime_equal)
    b2_sm_n10 = bound2(n1, m_range, alpha, beta, X, eps_smart_m, eps_prime_smart_m)

    ax1.semilogx(m_range, b1_n10, color='#d62728', linestyle='-', label='Joint')
    ax1.semilogx(m_range, b2_eq_n10, color='#1f77b4', linestyle='--', label='Decoupled (Uniform budget)')
    ax1.semilogx(m_range, b2_sm_n10, color='#2ca02c', linestyle='-', label='Decoupled (Adaptive budget)')

    # Find cross point for n = 10 (smart split)
    def obj_smart(m):
        return bound1(n1, m, alpha, beta, X, delta) - bound2(n1, m, alpha, beta, X, eps_smart_m, eps_prime_smart_m)
    
    def obj_equal(m):
        return bound1(n1, m, alpha, beta, X, delta) - bound2(n1, m, alpha, beta, X, eps_equal, eps_prime_equal)
    
    res = root_scalar(obj_smart, bracket=[10, 10000], method='brentq')
    m_cross = res.root
    y_cross = bound1(n1, m_cross, alpha, beta, X, delta)
    res_equal = root_scalar(obj_equal, bracket=[10, 10000], method='brentq')
    m_cross_equal = res_equal.root
    y_cross_equal = bound1(n1, m_cross_equal, alpha, beta, X, delta)

    # Mark the cross point with a clear annotation in open space
    ax1.plot(m_cross, y_cross, 'ro', markersize=4, zorder=5)
    ax1.annotate(f'Cross point\n$m={int(round(m_cross))}$', 
                 xy=(m_cross, y_cross), 
                 xytext=(m_cross * 8.0, y_cross * 1.45),
                 arrowprops=dict(arrowstyle='->', lw=0.7, color='black'),
                 fontsize=6, ha='center')

    ax1.set_xlabel(r'$m$', labelpad=1)
    ax1.set_ylabel('Complexity Value')
    ax1.grid(True, which='both', linestyle=':', alpha=0.5, lw=0.5)
    ax1.text(0.5, -0.3, r'(a) $n = 10$', transform=ax1.transAxes, 
             fontsize=8, fontweight='bold', ha='center', va='top')

    # Subplot 2: n = m (Balanced case)
    b1_nn = bound1(m_range, m_range, alpha, beta, X, delta)
    b2_eq_nn = bound2(m_range, m_range, alpha, beta, X, eps_equal, eps_prime_equal)
    b2_sm_nn = bound2(m_range, m_range, alpha, beta, X, eps_smart_m, eps_prime_smart_m)

    ax2.semilogx(m_range, b1_nn, color='#d62728', linestyle='-')
    ax2.semilogx(m_range, b2_eq_nn, color='#1f77b4', linestyle='--')
    ax2.semilogx(m_range, b2_sm_nn, color='#2ca02c', linestyle='-')

    ax2.set_xlabel(r'$m$', labelpad=1)
    ax2.grid(True, which='both', linestyle=':', alpha=0.5, lw=0.5)
    ax2.text(0.5, -0.3, r'(b) $n = m$', transform=ax2.transAxes, 
             fontsize=8, fontweight='bold', ha='center', va='top')

    # Single top legend for clarity and space saving
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.99), 
               ncol=3, frameon=False, fontsize=6.5, handlelength=1.5)

    plt.tight_layout()
    plt.subplots_adjust(top=0.86, bottom=0.30, wspace=0.35)

    for output_path in ["bound_comparison.pdf", "bound_comparison.png"]:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')

if __name__ == '__main__':
    main()

