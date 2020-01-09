# Simulation codes for the article: 

ERROR CONTROL IN THE NUMERICAL POSTERIOR DISTRIBUTION IN THE BAYESIAN UQ ANALYSIS OF
A SEMILINEAR EVOLUTION PDE. Daza, M. L., Montesinos-López, J. Cricelio, Capistrán, M. A., Christen, J. A., & Haario, H. (2019).

In the following programs we implemented the Algorithm 1 for examples 1, 2, 3. Also, we plot all the figures of section 2.

The available files are:

- FM_Ex1.py
    Algorithm 1 for the Fisher equation and a function to plot Figures 1a, 2a. 

- FM_Ex2.py
    Algorithm 1 for the Fitzhugh-Nagumo equation and a function to plot Figures 1b, 2b

- FM_Ex3.py
    Algorithm 1 for the Burgers-Fisher equation and a function to plot Figures 1c, 2a


In the following programs we implemented the Algorithm 2 for examples 4, 5, 6.

The available files are: 

- MCMC_Ex4.py
    Algorithm 2 implemented to inverse problem for Fisher's equation.

- MCMC_Ex5.py          
    Algorithm 2 implemented to inverse problem for Fitzhugh-Nagumo's equation.

- MCMC_Ex6.py
    Algorithm 2 implemented to inverse problem for Burgers-Fisher's equation.


In the following programs we plot all the figures of section 3.

The available files are:

- Post_NumVsTeo_Ex4.py
    Function to plot Figure 3b

- Post_NumVsTeo_Ex5.py
    Function to plot Figure 4b

- Post_NumVsTeo_Ex6.py
    Function to plot Figures 5b-c

The folder outputs contains the mcmc simulations for examples 4, 5 and 6.

- pytwalk.py
Library for the t-walk MCMC algorithm. For more details about this library see https://www.cimat.mx/~jac/twalk/
