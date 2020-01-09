import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma
from pytwalk import IAT

theta = np.array([4.5, 5.5])    # True parameter
# Define prior parameters:
alp_r = 3.5
bet_r = alp_r / 4.5
alp_s = 3.5
bet_s = alp_s / 5.5

T = 100000                                      # Chain length
start = int(.3 * T)                             # Burning
Out_t = np.loadtxt('outputs/samples_ex6_true.out')      # Output with the exact Forward Map
Out_n = np.loadtxt('outputs/samples_ex6_numeric.out')   # Output with the numeric Forward Map

IAT_t = 20                                      # Tining for the exact Forward Map
IAT_n = 20                                      # Tining for the numeric Forward Map

Out_r_t = Out_t[start::IAT_t, :2]          # Output without burning and tining
Out_r_n = Out_n[start::IAT_n, :2]          # Output without burning and tining

# Exact Forward Map
hpd_index_t = np.argsort(Out_r_t[:, -1])  # Maximum a posteriori Index
MAP_t = Out_r_t[hpd_index_t[0], :2]       # MAP
# Numeric Forward Map
hpd_index_n = np.argsort(Out_r_n[:, -1])  # Index of the maximum a posteriori
MAP_n = Out_r_n[hpd_index_n[0], :2]       # MAP


def hist_area_comparison_r(save, which=0, eti="r"):
    plt.figure()
    rt = plt.hist(Out_r_n[:, which], bins=20, label='Numeric', density=True)
    plt.hist(Out_r_t[:, which], bins=rt[1], histtype='step', lw=1.5, stacked=True, fill=False, color='m', label='Exact', density=True)
    xalph = np.linspace(Out_r_n.min(), Out_r_n.max(), 100)
    plt.plot(xalph, gamma.pdf(xalph, alp_r, scale=1 / bet_r), 'g-', lw=1.5, alpha=0.6, label="Prior")
    plt.axvline(theta[which], ymax=rt[0].max() * 0.3, color='r', label='True')
    plt.xlabel(eti)
    plt.ylabel('Density')
    plt.legend()
    if save:
        plt.savefig("Comparison_D_%s.eps"%(which))


# -------- Figure for Example 6 ---------
# To plot Figures 5b-c uncommented the following lines,
# hist_area_comparison_r(False, which=1, eti="s")
# hist_area_comparison_r(False, which=0)

