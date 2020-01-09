"""
In this program we implemented of Algorithm 1  for the Burgers-Fisher equation. Also,
we plot all the figures of section 2 of the paper for this example.

@author: J.Cricelio Montesinos-López and Maria L. Daza-Torres
"""

import numpy as np
from numba import jit
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.size'] = 11.5

# =============================================================================
# Parameters used for the Example 3
# =============================================================================

n_obs = 8           # Number of observation (even) n = nobs-1
t0 = 0              # Initial time
tau = 1             # Final time

# Boundary
x0 = 0
xN = 1

n_m = 16                    # number of times that the observation grid is divided
nx = n_m * (n_obs + 2)      # Number of nodes in x
dx = 1 / nx                 # Spatial step size
alpha = 4 / 3               # Parameter for the stability (CFL condition)
dt = dx ** 2 / alpha        # Temporal step size

grid_t = np.arange(t0, tau + dt, dt)            # Time grid
grid_x = np.arange(x0, xN, dx)                  # Spatial grid
M = len(grid_t)                                 # Number of time steps
N = len(grid_x[1:])                             # Dimension of state variables

index = M // 8
t_obs = grid_t[index]         # Time at which the data are observed
x_obs = grid_x[::n_m][1:-1]   # Location where data are observed

theta = np.array([4.5, 5.5])  # Parameters in the Burgers-Fisher equation

# Auxiliary parameters
p = 1 / dx ** 2
x_l = 0.00001
x_r = 6
s_l = 0.00001
s_r = 6

@jit(nopython=True)
def u_true(x, t, theta):
    """
        Exact solution for the Burgers-Fisher equation

        Parameters
        ---------
        x: float
           Spatial point to evaluate
        t: float
           Time to evaluate
        theta: array
           Parameters in Burgers-Fisher equation

        Returns
        -------
        sol: float
           Evaluated equation in (x,t) and the vector of parameters theta
    """
    r = theta[0]
    s = theta[1]
    aa = 0.5*s + r*2/s
    return 0.5 + 0.5*np.tanh(-s/4*(x-aa*t))


@jit(nopython=True)
def F(w, t, N, dx_, theta):
    """
       Function to compute the semi-discrete differential equation in the paper (eq. (8))
        for the Burgers-Fisher equation

       Parameters
       ----------
       w: array
           The vector solution in the iteration t-1
       t: float
           Time to evaluate
       N: int
           Dimension of state variables
       dx: float
           Spatial step size
       theta: float
           Parameters in Burgers-Fisher equation

       Returns
       -------
       F: array
           The right side of the semi-discrete differential equation in the paper (eq. (8)).
           dU/dt = AU + F(t, U)
       """
    r, s = theta
    p = 1 / dx_ ** 2
    q = 1 / dx_
    v_hat = np.zeros(N)
    v_hat[0] = u_true(x0, t, theta)
    v_hat[-1] = u_true(xN, t, theta)
    v = v_hat.copy()
    v[1:] = w[:-1]
    v_hat[:-1] = w[1:]
    F = p*(v_hat - 2*w + v) - s * q * (w - v) * w + r * w + v * (-(2 * r + s) * w + (r+s)*v)
    return F

# =============================================================================
# Runge-Kutta  parameters
# =============================================================================
sr = 6
c = np.array([0.0, 1.0 / 5, 3.0 / 10, 3.0 / 5, 1.0, 7.0 / 8])
b = np.array([37.0 / 378, 0.0, 250.0 / 621, 125.0 / 594, 0.0, 512.0 / 1771])
a = np.array([[0., 0., 0., 0., 0.,], [1.0 / 5, 0., 0., 0., 0.],
              [3.0 / 40, 9.0 / 40, 0., 0., 0.],
              [3.0 / 10, -9.0 / 10, 6.0 / 5,0.,0.],
              [-11.0 / 54, 5.0 / 2, -70.0 / 27, 35.0 / 27,0.],
              [1631.0 / 55296, 175.0 / 512, 575.0 / 13824, 44275.0 / 110592, 253.0 / 4096]])
Er = np.array([-0.00429377, 0.0, 0.01866859, -0.03415503, -0.01932199, 0.0391022])

@jit(nopython=True)
def rungekuttaodeint(theta,X0,grid,dx,dt):
    """
    Function to compute the numerical solution using RK45 for
    a linear system

    Parameters
    ----------
    theta: float
        Parameter in the Burgers-Fisher (Fisher) equation
    X0: array
        Initial condition
    grid: array
        Temporal grid
    dx: float
        Spatial step size
    dt: float
        Temporal step size

    Returns
    -------
    X: array
        The numeric solution for the system dU/dt = AU + F(t, U)
    E: array
        Global truncation error
    """
    M = len(grid)                 # Number of time steps
    N = len(X0)                   # Dimension of state variables
    k = np.zeros((sr, N))         # Size of the Butcher Tableau for the RK method
    X = np.zeros((M, N))          # Array to hold results
    X[0,:] = X0                   # Initial value
    E = np.zeros((M, N))          # Array to hold the estimated local truncation error
    for i in range(1, M):
        k[0,:] = F(X[i-1, :], grid[i] + dt*c[0], N, dx, theta)
        for j in range(1, sr):
            tmp = np.dot(k[:j, :].T, a[j][:j])
            k[j,:] = F(X[i-1, :] + dt*tmp, grid[i] + dt*c[j], N, dx, theta)
        tmp = np.dot(k.T, b)
        X[i,:] = X[i-1, :] + dt*tmp
        # Local Truncation error estimation at the time points
        tmp = np.dot(k.T, Er) + dx
        # The local truncation error is dt*tmp
        E[i, :] = E[i-1, :] + dt*tmp  # We accumulate the sum of all the local truncation errors

    return X, E


def U_xt(dx, dt, theta):
    """
     Function to compute the numeric solution for a specific spatial step size dx and
     temporal step size dt using the rungekuttaodeint() function

     Parameters
     ----------
     theta: float
         Parameter in the Burgers-Fisher equation
     dx: float
         Spatial step size
     dt: float
         Temporal step size

     Returns
     -------
     X: array
         The numeric solution for the system dU/dt = AU + F(t, U)
     Gerr: array
         Global truncation error
    """
    grid_x = np.arange(x0,xN,dx)
    grid_t = np.arange(t0, tau + dt, dt)
    X0 = u_true(grid_x[1:], t0, theta)
    X, Gerr = rungekuttaodeint(theta, X0, grid_t, dx, dt)
    return X, Gerr


def plot_num_vs_exac_t_x(theta):
    """
      This function plot the numerical solution and the exact solution
       for the Fisher equation
      Parameters
      ----------
      theta: float
          Parameter in the Burgers-Fisher equation
     """
    X, Gerr = U_xt(dx, dt, theta)
    X_num = X[:, ::n_m][:, 1:-1][:, ::2]
    x_nu = grid_x[::n_m][1:-1][::2]
    xx, tt = np.meshgrid(x_nu, grid_t)
    X_true = u_true(xx, tt, theta)

    plt.figure()
    for xi in range(0, n_obs // 2):
        if (xi == (n_obs // 2 - 2)):
            plt.plot(grid_t, X_true[:, xi], "black", label='Exact')
            plt.plot(grid_t, X_num[:, xi], "--r", label='Numeric')
        else:
            plt.plot(grid_t, X_true[:, xi], "black")
            plt.plot(grid_t, X_num[:, xi], "--r")
    plt.xlabel('t')
    plt.ylabel('u(t,x)')
    plt.legend()
   # plt.savefig("Error_ex3.eps")  # Uncomment the line for saving the figure


def Error_exact(X, grid_x, grid_t, theta):

    """
     Function to compute the exact error for the numerical solution
     of the Burgers-Fisher equation.

     Parameters
     ----------
     X: array
         Numerical solution for a grid t x grid x.
     theta: float
          Parameter in the Fisher equation
     grid_t: float
         Temporal step size
     grid_x: float
         Temporal step size

     Returns
     -------
     Error_Rk4: array
         Exact error for the RK45 numerical solution
    """
    xx, tt = np.meshgrid(grid_x[1:], grid_t)
    X_true = u_true(xx, tt, theta)
    Error_RK4 = np.max(np.abs(X - X_true))
    return Error_RK4


def Error_est(dx, theta):
    """
     Function to compute the error estimation proposed in the paper and the exact error

     Parameters
     ----------
     dx: float
         Spatial step size
     theta: float
          Parameters in the Burgers-Fisher equation

     Returns
     -------
     Est_Error_max: array
                Estimate error

     e_exact_max_h: array
                Exact error
     [array, array]
    """
    dx2 = 2 * dx
    dt = dx ** 2 / alpha

    grid_t = np.arange(t0, tau + dt, dt)  # Time grid
    grid_x = np.arange(x0, xN, dx)        # Spatial grid

    uh = U_xt(dx, dt, theta)
    u2h = U_xt(dx2, dt, theta)
    Uh = uh[0][:, 1::2]
    U2h = u2h[0]

    eRK_h = uh[1][:, 1::2]
    eRK_2h = u2h[1]
    eRK_max_h = np.max(np.abs(uh[1]))
    C_ = (np.max(np.abs(Uh - U2h)) + np.max(np.abs(eRK_2h - eRK_h))) / (3 * dx ** 2)
    e_exact_max_h = Error_exact(uh[0], grid_x, grid_t, theta)
    Est_Error_max = eRK_max_h + C_ * dx ** 2
    return Est_Error_max, e_exact_max_h


def error_order(h_x):
    """
     Function to compute the order for the numerical solution

     Parameters
     ----------
     h_x: float
        Spatial step size

     Returns
     -------
     ordh2k5: array
        numerical order
    """
    dt_x = h_x ** 2 / alpha
    ordh2 = h_x
    ordk5 = dt_x ** 4
    ordh2k5 =  .1*ordh2*(1+ordk5)
    return ordh2k5


def plot_error_ex_est(dx, theta, num=4):
    """
      This function plot the exact error vs estimate error
      Parameters
      ----------
      dx: float
          Spatial step size
      theta: float
          Parameter in the Fisher equation

      Returns
      -------
      e_s_max: array
                Estimate error

      e_exact_max: array
                Exact error
      H: array
        step size vector
      [array, array, array]
     """

    e_s_max = []
    e_exact_max = []
    H = []
    for i in range(num):
        h = 2 ** i * dx
        H.append(h)
        Est_Error_max, e_exact_max_h = Error_est(h, theta)
        e_s_max.append(Est_Error_max)
        e_exact_max.append(e_exact_max_h)

    fig, ax = plt.subplots()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.loglog(H, e_exact_max, "black", marker="v", label='$|u-u_h|_{l_{\infty}}$')
    ax.loglog(H, e_s_max,"r-.", marker="s", label='Estimated Error')

    h_x = np.linspace(H[0], H[-1], 100); er_h = error_order(h_x)
    h_1 = np.linspace(h_x[25], h_x[35], 2)
    ord_h1 = error_order(h_1)
    codes = [Path.MOVETO] + [Path.LINETO] * 2 + [Path.CLOSEPOLY]
    vertices = [(h_1[0], ord_h1[0]), (h_1[-1], ord_h1[-1]), (h_1[-1], ord_h1[0]), (0, 0)]
    vertices = np.array(vertices, float)
    path = Path(vertices, codes)
    pathpatch = PathPatch(path, facecolor='None', edgecolor='black')
    ax.add_patch(pathpatch)

    # Results for example 1
    sali = ([7.875716361598596e-05,
             0.00031504702422968883,
             0.0012604819074808447,
             0.0050466191871248855],
            [3.6146042070084317e-05,
             0.00014449013256268017,
             0.0005771794490772031,
             0.0023019738066867856],
            [0.00625, 0.0125, 0.025, 0.05])
    vertices2 = np.array([[0.01729798, 0.00014961], [0.02171717, 0.00023582], [0.02171717, 0.00014961], [0., 0.]])
    path2 = Path(vertices2, codes)
    pathpatch2 = PathPatch(path2, facecolor='None', edgecolor='black')
    ax.add_patch(pathpatch2)
    ax.loglog(H, sali[1], "black", marker="v")
    ax.loglog(H, sali[0],"r-.", marker="s")

    which = 3
    y = (e_exact_max[which] + e_s_max[which]) / 3
    xy = H[which]
    ax.text(xy+.1*xy, y, "Example 3", ha="center", va="center", size=10, rotation=0, bbox=dict(facecolor='grey', alpha=0.))

    y1 = (sali[0][which]+sali[1][which])/2.2
    ax.text(xy+.1*xy, y1, "Example 1", ha="center", va="center", size=10, rotation=0, bbox=dict(facecolor='grey', alpha=0.))#rotation=24

    plt.xlabel('h')
    plt.ylabel('Error')
    plt.legend()
    # plt.savefig("Error_Ex3.eps")
    return e_s_max, e_exact_max, H


jit(nopython=True)
def FindClosestIndex(tt, d):
    """
    Find the index of the closest item (time, array) in tt to d[i] (obs. time).

    Parameters
    ----------
    tt: array
    d: array

    Returns
    -------
    index: array
         List of indexes.

    """
    index = np.array([0] * len(d))
    for j, t in enumerate(d):
        mn = np.min(np.abs(tt - t))
        for i, df in enumerate(abs(tt - t)):
            if (df == mn):
                index[j] = i
    return index


def U_xt_obs(dx, dt, theta, nm):
    """
     Function to compute the numeric solution in the observational grid using the rungekuttaodeint() function

     Parameters
     ----------
     theta: float
         Parameter in the Fisher equation
     dx: float
         Spatial step size
     dt: float
         Temporal step size
     nm: int
         parameter for choosing the grid to evaluate

     Returns
     -------
     sol_obs:  array
            Solution in observational points
     err_obs: array
            Error in observational points
    """
    grid_t_ = np.arange(t0, tau + dt, dt)
    ind = FindClosestIndex(grid_t_, np.array([t_obs]))
    X, Gerr = U_xt(dx, dt, theta)
    sol_obs = X[ind[0]][(nm - 1)::nm][:-1]
    err_obs = Gerr[ind[0]][(nm - 1)::nm][:-1]
    return sol_obs, err_obs


def Error_order(theta):
    """
     Function to compute the numerical order for the approximate solution

     Parameters
     ----------
     h_x: array
     Returns
     -------
     order: array
            Numerical order
    """
    hs = []; order = []
    plt.figure()
    for n_m in range(4, 14, 4):
        nx = n_m * (n_obs + 2)
        dx = 1 / nx
        hs.append(dx)
        dt = dx ** 2 / alpha
        uh = U_xt_obs(dx, dt, theta, nm=n_m)
        u2h = U_xt_obs(2 * dx, dt, theta, nm=n_m // 2)
        u4h = U_xt_obs(4 * dx, dt, theta, nm=n_m // 4)
        Uh = uh[0]
        U2h = u2h[0]
        U4h = u4h[0]
        order.append(np.max(np.abs(U4h - U2h))/np.max(np.abs(U2h - Uh)))
    plt.plot(hs, order)
    return order

# -------- Figures for Example 3 ---------
# To plot Figures 1c, 2a and the Table 1 (column 3) uncommented the following lines,


# plot_num_vs_exac_t_x(theta)          # Fig 1c
# plot_error_ex_est(dx, theta, num=4)  # Fig 2a
# Error_order(theta)                   # Table 1 (column 4)

