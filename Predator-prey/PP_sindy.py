#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 00:03:19 2021

@author: mohamedazizbhouri
"""

################################################################################
    
import numpy as onp
import matplotlib.pyplot as plt

import jax.numpy as np
import jax.random as random
from jax.experimental.ode import odeint as odeint_jax
from jax.config import config
config.update("jax_enable_x64", True)

import pysindy as ps

onp.random.seed(1234)

plt.rcParams.update(plt.rcParamsDefault)
plt.rc('font', family='serif')
plt.rcParams.update({'font.size': 16,
                     'lines.linewidth': 2,
                     'axes.labelsize': 20,  # fontsize for x and y labels
                     'axes.titlesize': 20,
                     'xtick.labelsize': 16,
                     'ytick.labelsize': 16,
                     'legend.fontsize': 20,
                     'axes.linewidth': 2,
                     "pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
                     "text.usetex": True,                # use LaTeX to write all text
                     })
dpiv = 100

################################################################################

def LV(x, t, alpha, beta, gamma, delta):
    x1, x2 = x
    dxdt = [alpha*x1+beta*x1*x2, delta*x1*x2+gamma*x2]
    return dxdt

key = random.PRNGKey(123)
D = 2
alpha = 1.0
beta = -0.1
gamma = -1.5
delta = 0.75
IC = 5.0

noise = 0.1
N_fine = 1100
Tf = 16.5
Tf_test = 30
x0_onp = [5.0, IC]
x0 = np.array(x0_onp)

# Test data
t_star = np.linspace(0, Tf_test, 2*N_fine+1)
t_grid_test = onp.array(t_star)
data_test = onp.array( odeint_jax(LV, x0, t_grid_test, alpha, beta, gamma, delta) )

library_functions = [
    lambda x : x,
    lambda x,y : x*y,
    lambda x : x**2,
    lambda x : x**3
]

library_function_names = [    
    lambda x : x,
    lambda x,y : x + '.' + y,
    lambda x : x + '^2',
    lambda x : x + '^3'
]
custom_library = ps.CustomLibrary(
    library_functions=library_functions, function_names=library_function_names
)

################################################################################

N = 55 
N_fine = 1100

# Training data
t_fine = np.linspace(0, Tf, N_fine+1)
X_fine = odeint_jax(LV, x0, t_fine, alpha, beta, gamma, delta)
X_fine_noise = X_fine + noise*X_fine.std(0)*random.normal(key, X_fine.shape)
t = t_fine[onp.array( list(range(0, N_fine+1, N_fine//N)) )]
X_train = X_fine_noise[list(range(0, N_fine+1, N_fine//N)),:]

gap = 4
ind_t = np.array([0])
ind_t = np.concatenate([ind_t[:,None],np.arange(gap+1,N+1)[:,None]])
ind_t = ind_t[:,0]

t_grid = onp.array(t[ind_t])
print('case_1_as_GP_NODE',t_grid,t_grid.shape)
model_GP_ODE = ps.SINDy(feature_library=custom_library)
model_GP_ODE.fit(onp.array(X_train[ind_t,:]), t=t_grid)
print('case_1_as_GP_NODE:')
model_GP_ODE.print()
x_test_sim = model_GP_ODE.simulate(x0_onp, t_grid_test)

plt.figure(figsize=(12,6.5))
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.plot(t_grid_test, data_test[:,0],'r-', label = "True trajectory of $x_1(t)$")
plt.plot(t_grid, X_train[ind_t,0], 'ro', label = "Training data of $x_1(t)$")
plt.plot(t_grid_test, x_test_sim[:,0],'g--', label = "SINDy prediction of $x_1(t)$")
plt.xlabel('$t$',fontsize=26)
plt.ylabel('$x_1(t)$',fontsize=26)
plt.ylim((0.0, 8.0)) 
plt.legend(loc='upper right', frameon=False, prop={'size': 20})
plt.savefig('plots_sindy/case_1_as_GP_NODE_x1.png', dpi = dpiv)  

plt.figure(figsize=(12,6.5))
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.plot(t_grid_test, data_test[:,1],'r-', label = "True trajectory of $x_2(t)$")
plt.plot(t_grid, X_train[ind_t,1], 'ro', label = "Training data of $x_2(t)$")
plt.plot(t_grid_test, x_test_sim[:,1],'g--', label = "SINDy prediction of $x_2(t)$")
plt.xlabel('$t$',fontsize=26)
plt.ylabel('$x_2(t)$',fontsize=26)
plt.ylim((0.0, 45.0)) 
plt.legend(loc='upper right', frameon=False, prop={'size': 20})
plt.savefig('plots_sindy/case_1_as_GP_NODE_x2.png', dpi = dpiv)  

################################################################################

N = 7040 # dt = 0.00234375
N_fine = 35200 

# Training data
t_fine = np.linspace(0, Tf, N_fine+1)
X_fine = odeint_jax(LV, x0, t_fine, alpha, beta, gamma, delta)
X_fine_noise = X_fine + noise*X_fine.std(0)*random.normal(key, X_fine.shape)
t = t_fine[onp.array( list(range(0, N_fine+1, N_fine//N)) )]
X_train = X_fine_noise[list(range(0, N_fine+1, N_fine//N)),:]

gap = 639
ind_t = np.array([0])
ind_t = np.concatenate([ind_t[:,None],np.arange(gap+1,N+1)[:,None]])
ind_t = ind_t[:,0]

t_grid = onp.array(t[ind_t])
print('case_2_fine_dt',t_grid,t_grid.shape)
model_GP_ODE_fine_dt = ps.SINDy(feature_library=custom_library)
model_GP_ODE_fine_dt.fit(onp.array(X_train[ind_t,:]), t=t_grid)
print('case_2_fine_dt:')
model_GP_ODE_fine_dt.print()
x_test_sim = model_GP_ODE_fine_dt.simulate(x0_onp, t_grid_test)

plt.figure(figsize=(12,6.5))
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.plot(t_grid_test, data_test[:,0],'r-', label = "True trajectory of $x_1(t)$")
plt.plot(t_grid, X_train[ind_t,0], 'ro', label = "Training data of $x_1(t)$")
plt.plot(t_grid_test, x_test_sim[:,0],'g--', label = "SINDy prediction of $x_1(t)$")
plt.xlabel('$t$',fontsize=26)
plt.ylabel('$x_1(t)$',fontsize=26)
plt.ylim((0.0, 8.0)) 
plt.legend(loc='upper right', frameon=False, prop={'size': 20})
plt.savefig('plots_sindy/case_2_fine_dt_x1.png', dpi = dpiv)  

plt.figure(figsize=(12,6.5))
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.plot(t_grid_test, data_test[:,1],'r-', label = "True trajectory of $x_2(t)$")
plt.plot(t_grid, X_train[ind_t,1], 'ro', label = "Training data of $x_2(t)$")
plt.plot(t_grid_test, x_test_sim[:,1],'g--', label = "SINDy prediction of $x_2(t)$")
plt.xlabel('$t$',fontsize=26)
plt.ylabel('$x_2(t)$',fontsize=26)
plt.ylim((0.0, 45.0)) 
plt.legend(loc='upper right', frameon=False, prop={'size': 20})
plt.savefig('plots_sindy/case_2_fine_dt_x2.png', dpi = dpiv)  

################################################################################

N = 7040 # dt = 0.00234375
N_fine = 35200 

# Training data
t_fine = np.linspace(0, Tf, N_fine+1)
X_fine = odeint_jax(LV, x0, t_fine, alpha, beta, gamma, delta)
X_fine_noise = X_fine + noise*X_fine.std(0)*random.normal(key, X_fine.shape)

t = t_fine[onp.array( list(range(0, N_fine+1, N_fine//N)) )]
X_train = X_fine_noise[list(range(0, N_fine+1, N_fine//N)),:]

gap = 639
ind_t = np.array([0])
ind_t = np.concatenate([ind_t[:,None],np.arange(gap+1,N+1)[:,None]])
ind_t = ind_t[:,0]

t_grid = onp.array(t[ind_t])
print('case_3_fine_dt_no_noise',t_grid,t_grid.shape)
model_GP_ODE_fine_dt_no_noise = ps.SINDy(feature_library=custom_library)
model_GP_ODE_fine_dt_no_noise.fit(onp.array(X_train[ind_t,:]), t=t_grid)
print('case_3_fine_dt_no_noise:')
model_GP_ODE_fine_dt_no_noise.print()
x_test_sim = model_GP_ODE_fine_dt_no_noise.simulate(x0_onp, t_grid_test)

plt.figure(figsize=(12,6.5))
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.plot(t_grid_test, data_test[:,0],'r-', label = "True trajectory of $x_1(t)$")
plt.plot(t_grid, X_train[ind_t,0], 'ro', label = "Training data of $x_1(t)$")
plt.plot(t_grid_test, x_test_sim[:,0],'g--', label = "SINDy prediction of $x_1(t)$")
plt.xlabel('$t$',fontsize=26)
plt.ylabel('$x_1(t)$',fontsize=26)
plt.ylim((0.0, 8.0)) 
plt.legend(loc='upper right', frameon=False, prop={'size': 20})
plt.savefig('plots_sindy/case_3_fine_dt_no_noise_x1.png', dpi = dpiv)  

plt.figure(figsize=(12,6.5))
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.plot(t_grid_test, data_test[:,1],'r-', label = "True trajectory of $x_2(t)$")
plt.plot(t_grid, X_train[ind_t,1], 'ro', label = "Training data of $x_2(t)$")
plt.plot(t_grid_test, x_test_sim[:,1],'g--', label = "SINDy prediction of $x_2(t)$")
plt.xlabel('$t$',fontsize=26)
plt.ylabel('$x_2(t)$',fontsize=26)
plt.ylim((0.0, 45.0)) 
plt.legend(loc='upper right', frameon=False, prop={'size': 20})
plt.savefig('plots_sindy/case_3_fine_dt_no_noise_x2.png', dpi = dpiv)  

################################################################################

N = 110
t_grid = onp.linspace(0, Tf, N//2+1)
data = odeint_jax(LV, x0, t_grid, alpha, beta, gamma, delta)
data = onp.array( data + noise*data.std(0)*random.normal(key, data.shape) )

print('case_4_no_t_gap_large_dt',t_grid,t_grid.shape)
model_no_t_gap_large_dt = ps.SINDy(feature_library=custom_library)
model_no_t_gap_large_dt.fit(data, t=t_grid) # data Nt x D
print('case_4_no_t_gap_large_dt:')
model_no_t_gap_large_dt.print()
x_test_sim = model_no_t_gap_large_dt.simulate(x0_onp, t_grid_test)

plt.figure(figsize=(12,6.5))
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.plot(t_grid_test, data_test[:,0],'r-', label = "True trajectory of $x_1(t)$")
plt.plot(t_grid, data[:,0], 'ro', label = "Training data of $x_1(t)$")
plt.plot(t_grid_test, x_test_sim[:,0],'g--', label = "SINDy prediction of $x_1(t)$")
plt.xlabel('$t$',fontsize=26)
plt.ylabel('$x_1(t)$',fontsize=26)
plt.ylim((0.0, 8.0)) 
plt.legend(loc='upper right', frameon=False, prop={'size': 20})
plt.savefig('plots_sindy/case_4_no_t_gap_large_dt_x1.png', dpi = dpiv)  

plt.figure(figsize=(12,6.5))
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.plot(t_grid_test, data_test[:,1],'r-', label = "True trajectory of $x_2(t)$")
plt.plot(t_grid, data[:,1], 'ro', label = "Training data of $x_2(t)$")
plt.plot(t_grid_test, x_test_sim[:,1],'g--', label = "SINDy prediction of $x_2(t)$")
plt.xlabel('$t$',fontsize=26)
plt.ylabel('$x_2(t)$',fontsize=26)
plt.ylim((0.0, 45.0)) 
plt.legend(loc='upper right', frameon=False, prop={'size': 20})
plt.savefig('plots_sindy/case_4_no_t_gap_large_dt_x2.png', dpi = dpiv)  

################################################################################
################################################################################
################################################################################
################################################################################

N = 110
t_grid = onp.linspace(0, Tf, 4*N+1) # N//2+1) # 441 dt = 0.0375
data = odeint_jax(LV, x0, t_grid, alpha, beta, gamma, delta)
data = onp.array( data + noise*data.std(0)*random.normal(key, data.shape) )

print('case_5_no_t_gap_small_dt',t_grid,t_grid.shape)
model_no_t_gap_small_dt = ps.SINDy(feature_library=custom_library)
model_no_t_gap_small_dt.fit(data, t=t_grid) # data Nt x D
print('case_5_no_t_gap_small_dt:')
model_no_t_gap_small_dt.print()
x_test_sim = model_no_t_gap_small_dt.simulate(x0_onp, t_grid_test)

plt.figure(figsize=(12,6.5))
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.plot(t_grid_test, data_test[:,0],'r-', label = "True trajectory of $x_1(t)$")
plt.plot(t_grid, data[:,0], 'ro', label = "Training data of $x_1(t)$")
plt.plot(t_grid_test, x_test_sim[:,0],'g--', label = "SINDy prediction of $x_1(t)$")
plt.xlabel('$t$',fontsize=26)
plt.ylabel('$x_1(t)$',fontsize=26)
plt.ylim((0.0, 8.)) 
plt.legend(loc='upper right', frameon=False, prop={'size': 20})
plt.savefig('plots_sindy/case_5_no_t_gap_small_dt_x1.png', dpi = dpiv)  

plt.figure(figsize=(12,6.5))
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.plot(t_grid_test, data_test[:,1],'r-', label = "True trajectory of $x_2(t)$")
plt.plot(t_grid, data[:,1], 'ro', label = "Training data of $x_2(t)$")
plt.plot(t_grid_test, x_test_sim[:,1],'g--', label = "SINDy prediction of $x_2(t)$")
plt.xlabel('$t$',fontsize=26)
plt.ylabel('$x_2(t)$',fontsize=26)
plt.ylim((0.0, 45.0)) 
plt.legend(loc='upper right', frameon=False, prop={'size': 20})
plt.savefig('plots_sindy/case_5_no_t_gap_small_dt_x2.png', dpi = dpiv)  
