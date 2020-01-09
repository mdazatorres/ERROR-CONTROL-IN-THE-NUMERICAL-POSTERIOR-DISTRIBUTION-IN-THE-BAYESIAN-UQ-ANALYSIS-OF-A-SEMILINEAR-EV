# =============================================================================
# Comparison posterior samples: numeric vs theorical FM
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma
from pytwalk import IAT

theta = 4  # True parameter
# Define prior parameters:
alp_r = 5
bet_r = alp_r / 6

T = 350000                                # Chain length
start = int(.3 * T)                       # Burning
Out_t = np.loadtxt('outputs/samples_ex4_true.out')     # Output with the exact Forward Map
Out_n = np.loadtxt('outputs/samples_ex4_numeric.out')  # Output with the numeric Forward Map

IAT_t = int(IAT(Out_t, cols=0))         # Tining for the exact Forward Map
IAT_n = int(IAT(Out_n, cols=0))         # Tining for the numeric Forward Map

# Exact Forward Map
hpd_index_t = np.argsort(Out_t[:, -1])  # Maximum a posteriori Index
MAP_t = Out_n[hpd_index_t[0], 0]        # MAP
# Numeric Forward Map
hpd_index_n = np.argsort(Out_n[:, -1])  # Index of the maximum a posteriori
MAP_n = Out_n[hpd_index_n[0], 0]        # MAP

Out_r_t = Out_t[start::IAT_t, 0]        # Output without burning and tining
Out_r_n = Out_n[start::IAT_n, 0]        # Output without burning and tining


def hist_area_comparison_r(save):
    """
    Function to plot the histogram of the MCMC simulations with the numeric and exact FM
    """
    plt.figure()
    rt = plt.hist(Out_r_n, bins=30, label='Numeric', density=True)
    plt.hist(Out_r_t, bins=rt[1], histtype='step', lw=1.5, stacked=True, fill=False, color='m', label='Exact',density=True)
    xalph = np.linspace(Out_r_n.min(), Out_r_n.max(), 100)
    plt.plot(xalph, gamma.pdf(xalph, alp_r, scale=1 / bet_r), 'g-', lw=1.5, alpha=0.6,label="Prior")
    plt.axvline(theta, ymax=rt[0].max() * 0.02, color='r', label='True')
    plt.xlabel('r')
    plt.ylabel('Density')
    plt.legend()
    if save:
        plt.savefig("Comparison_D.eps")


# -------- Figure for Example 4 ---------
# To plot Figure 3b uncommented the following line,
# hist_area_comparison_r(False)
