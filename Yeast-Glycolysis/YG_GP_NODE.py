#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 15:42:18 2020

@author: mohamedazizbhouri
"""

import math

import jax.numpy as np
import jax.random as random
from jax import vmap, jit
from jax.experimental.ode import odeint
from jax.config import config
config.update("jax_enable_x64", True)

from numpyro import sample
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

import numpy as onp
import matplotlib.pyplot as plt
from functools import partial
import time

class ODE_GP:
    # Initialize the class
    def __init__(self, t, i_t, X_train_ff, x0, dxdt, ind):    
        # Normalization 
        self.t = t
        self.x0 = x0
        self.i_t = i_t
        self.dxdt = dxdt      
        self.jitter = 1e-8
        self.ind = ind
        
        self.max_t = t.max(0)
        
        self.max_X = []
        self.X = []
        self.N = []
        self.D = len(i_t)
        self.t_t = []
        for i in range(len(i_t)):
            self.max_X.append(np.abs(X_train_ff[i]).max(0))
            self.X.append(X_train_ff[i] / self.max_X[i])
            self.N.append(X_train_ff[i].shape[0])
            self.t_t.append(t[self.i_t[i]]/self.max_t)
        
    @partial(jit, static_argnums=(0,))     
    def RBF(self,x1, x2, params):
        diffs = (x1 / params).T - x2 / params
        return np.exp(-0.5 * diffs**2)

    def model(self, t, X):        

        noise = sample('noise', dist.LogNormal(0.0, 1.0), sample_shape=(self.D,))
        hyp = sample('hyp', dist.Gamma(1.0, 0.5), sample_shape=(self.D,))
        W = sample('W', dist.LogNormal(0.0, 1.0), sample_shape=(self.D,))
        
        J0 = sample('J0', dist.Uniform(1.0, 10.0)) # 2.5
        k1 = sample('k1', dist.Uniform(80., 120.0)) # 100.
        k2 = sample('k2', dist.Uniform(1., 10.0)) # 6.
        k3 = sample('k3', dist.Uniform(2., 20.0)) # 16.
        k4 = sample('k4', dist.Uniform(80., 120.0)) # 100.
        k5 = sample('k5', dist.Uniform(0.1, 2.0)) # 1.28
        k6 = sample('k6', dist.Uniform(2., 20.0)) # 12.
        k = sample('k', dist.Uniform(0.1, 2.0)) # 1.8
        ka = sample('ka', dist.Uniform(2., 20.0)) # 13.
        q = sample('q', dist.Uniform(1., 10.0)) # 4.
        KI = sample('KI', dist.Uniform(0.1, 2.0)) # 0.52
        phi = sample('phi', dist.Uniform(0.05, 1.0)) # 0.1
        Np = sample('Np', dist.Uniform(0.1, 2.0)) # 1.
        A = sample('A', dist.Uniform(1., 10.0)) #4.
        
        IC = sample('IC', dist.Uniform(0, 1))
        
        # compute kernel
        K_11 = W[0]*self.RBF(self.t_t[0], self.t_t[0], hyp[0]) + np.eye(self.N[0])*(noise[0] + self.jitter)
        K_22 = W[1]*self.RBF(self.t_t[1], self.t_t[1], hyp[1]) + np.eye(self.N[1])*(noise[1] + self.jitter)
        K_33 = W[2]*self.RBF(self.t_t[2], self.t_t[2], hyp[2]) + np.eye(self.N[2])*(noise[2] + self.jitter)
        K = np.concatenate([np.concatenate([K_11, np.zeros((self.N[0], self.N[1])), np.zeros((self.N[0], self.N[2]))], axis = 1),
                            np.concatenate([np.zeros((self.N[1], self.N[0])), K_22, np.zeros((self.N[1], self.N[2]))], axis = 1),
                            np.concatenate([np.zeros((self.N[2], self.N[0])), np.zeros((self.N[2], self.N[1])), K_33], axis = 1)], axis = 0)
        
        # compute mean
        x0 = np.array([0.5, 1.9, 0.18, 0.15, IC, 0.1, 0.064])
        mut = odeint(self.dxdt, x0, self.t.flatten(), J0, k1, k2, k3, k4, k5, k6, k, ka, q, KI, phi, Np, A)
        mu1 = mut[self.i_t[0],ind[0]] / self.max_X[0]
        mu2 = mut[self.i_t[1],ind[1]] / self.max_X[1]
        mu3 = mut[self.i_t[2],ind[2]] / self.max_X[2]
        mu = np.concatenate((mu1,mu2,mu3),axis=0)
        
        # Concat data
        mu = mu.flatten('F')
        X = np.concatenate((self.X[0],self.X[1],self.X[2]),axis=0)
        X = X.flatten('F')

        # sample X according to the standard gaussian process formula
        sample("X", dist.MultivariateNormal(loc=mu, covariance_matrix=K), obs=X)
        
    # helper function for doing hmc inference
    def train(self, settings, rng_key):
        start = time.time()
        kernel = NUTS(self.model, 
                      target_accept_prob = settings['target_accept_prob'])
        mcmc = MCMC(kernel, 
                    num_warmup = settings['num_warmup'], 
                    num_samples = settings['num_samples'],
                    num_chains = settings['num_chains'],
                    progress_bar=True,
                    jit_model_args=True)
        mcmc.run(rng_key, self.t, self.X)
        mcmc.print_summary()
        elapsed = time.time() - start
        print('\nMCMC elapsed time: %.2f seconds' % (elapsed))
        return mcmc.get_samples()
    
    @partial(jit, static_argnums=(0,))     
    def predict(self, t_star, J0, k1, k2, k3, k4, k5, k6, k, ka, q, KI, phi, Np, A, N20):
        x0_l = np.array([0.5, 1.9, 0.18, 0.15, N20, 0.1, 0.064])
        X = odeint(self.dxdt, x0_l, t_star, J0, k1, k2, k3, k4, k5, k6, k, ka, q, KI, phi, Np, A)
        return X

plt.rcParams.update(plt.rcParamsDefault)
plt.rc('font', family='serif')
plt.rcParams.update({'font.size': 16,
                     'lines.linewidth': 2,
                     'axes.labelsize': 20,
                     'axes.titlesize': 20,
                     'xtick.labelsize': 16,
                     'ytick.labelsize': 16,
                     'legend.fontsize': 20,
                     'axes.linewidth': 2,
                     "pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
                     "text.usetex": True,                # use LaTeX to write all text
                     })
    
def glyc(x, t, J0, k1, k2, k3, k4, k5, k6, k, ka, q, KI, phi, Np, A):
    S1, S2, S3, S4, N2, A3, S4ex = x
    J = ka*(S4-S4ex)
    N1 = Np-N2
    A2 = A-A3
    v1 = k1*S1*A3/(1+(A3/KI)**q)
    v2 = k2*S2*N1
    v3 = k3*S3*A2
    v4 = k4*S4*N2
    v5 = k5*A3
    v6 = k6*S2*N2
    v7 = k*S4ex
    dxdt = [J0-v1, 2*v1-v2-v6, v2-v3, v3-v4-J, v2-v4-v6, -2*v1+2*v3-v5, phi*J-v7]
    return dxdt

key = random.PRNGKey(1234)
D = 7
J0 = 2.5
k1 = 100.
k2 = 6.
k3 = 16.
k4 = 100.
k5 = 1.28
k6 = 12.
k = 1.8
ka = 13.
q = 4.
KI = 0.52
phi = 0.1
Np = 1.
A = 4.
IC = 0.16

noise = 0.1
N = 120
N_fine = 1200
Tf = 3
Tf_test = 6
x0 = np.array([0.5, 1.9, 0.18, 0.15, IC, 0.1, 0.064])

# Training data
t_fine = np.linspace(0, Tf, N_fine+1)
X_fine = odeint(glyc, x0, t_fine, J0, k1, k2, k3, k4, k5, k6, k, ka, q, KI, phi, Np, A)
X_fine_noise = X_fine + noise*X_fine.std(0)*random.normal(key, X_fine.shape)

t = t_fine[[list(range(0, N_fine+1, N_fine//N))]]
X_train = X_fine_noise[list(range(0, N_fine+1, N_fine//N)),:]

# Test data
t_star = np.linspace(0, Tf_test, 2*N_fine+1)
X_star = odeint(glyc, x0, t_star, J0, k1, k2, k3, k4, k5, k6, k, ka, q, KI, phi, Np, A)

gap = 19
ind_t = np.array([0])
ind_t = np.concatenate([ind_t[:,None],np.arange(gap+1,N+1)[:,None]])
ind_t = ind_t[:,0]

j1 = list(range(0, ind_t.shape[0]+1, 2))
j2 = list([0]) + list( range(1, ind_t.shape[0]+1, 2) )

i1 = ind_t[j1]
i1 = i1[1:]
i2 = ind_t[j2]
i3 = ind_t[j1]

i_t = []
i_t.append(i1)
i_t.append(i2)
i_t.append(i3)

ind = [4,5,6]

X1_train = X_train[i_t[0],ind[0]]
X2_train = X_train[i_t[1],ind[1]]
X3_train = X_train[i_t[2],ind[2]]

X_train_ff = []
X_train_ff.append(X1_train)
X_train_ff.append(X2_train)
X_train_ff.append(X3_train)

model = ODE_GP(t[:,None], i_t, X_train_ff, x0, glyc, ind)
rng_key_train, rng_key_predict = random.split(random.PRNGKey(0))

num_warmup = 4000
num_samples = 8000
num_chains = 1
target_accept_prob = 0.85
settings = {'num_warmup': num_warmup,
            'num_samples': num_samples,
            'num_chains': num_chains,
            'target_accept_prob': target_accept_prob}
samples = model.train(settings, rng_key_train)  
print('True values: J0 = %f, k1 = %f, k2 = %f, k3 = %f, k4 = %f, k5 = %f, k6 = %f, k = %f, ka = %f, q = %f, KI = %f, phi = %f, Np = %f, A = %f, IC = %f' % (J0, k1, k2, k3, k4, k5, k6, k, ka, q, KI, phi, Np, A, IC))

vmap_args = (samples['J0'], samples['k1'], samples['k2'], samples['k3'], samples['k4'], samples['k5'], samples['k6'], samples['k'], samples['ka'], samples['q'], samples['KI'], samples['phi'], samples['Np'], samples['A'], samples['IC'])

np.save('data/par_and_IC',np.array(vmap_args))
np.save('data/noise',np.array(samples['noise']))
np.save('data/hyp',np.array(samples['hyp']))
np.save('data/W',np.array(samples['W']))

def RBF(x1, x2, params):
    diffs = (x1 / params).T - x2 / params
    return np.exp(-0.5 * diffs**2)

Nt = N+1
N_fine = 100
t_test = np.linspace(0, Tf_test, 2*N_fine+1)
Nt_test = t_test.shape[0]

t_tr = t[:,None] /model.max_t
t_te = t_test[:,None] /model.max_t

pred_X_tr_i = lambda J0, k1, k2, k3, k4, k5, k6, k, ka, q, KI, phi, Np, A, N20: model.predict(t, J0, k1, k2, k3, k4, k5, k6, k, ka, q, KI, phi, Np, A, N20)
X_tr_i = vmap(pred_X_tr_i)(*vmap_args)

pred_X_ode_i = lambda J0, k1, k2, k3, k4, k5, k6, k, ka, q, KI, phi, Np, A, N20: model.predict(t_test, J0, k1, k2, k3, k4, k5, k6, k, ka, q, KI, phi, Np, A, N20)
X_ode_i = vmap(pred_X_ode_i)(*vmap_args)

X_pred_GP = []
Npred_GP_f = 0
for i in range(num_samples):
    if i % 500 == 0:
        print(i)
    K1_tr = samples['W'][i,0]*RBF(model.t_t[0], model.t_t[0], samples['hyp'][i,0]) + np.eye(model.N[0])*(samples['noise'][i,0] + model.jitter)
    K2_tr = samples['W'][i,1]*RBF(model.t_t[1], model.t_t[1], samples['hyp'][i,1]) + np.eye(model.N[1])*(samples['noise'][i,1] + model.jitter)
    K3_tr = samples['W'][i,2]*RBF(model.t_t[2], model.t_t[2], samples['hyp'][i,2]) + np.eye(model.N[2])*(samples['noise'][i,2] + model.jitter)
    K_tr = np.concatenate([np.concatenate([K1_tr, np.zeros((model.N[0], model.N[1])), np.zeros((model.N[0], model.N[2]))], axis = 1),
                           np.concatenate([np.zeros((model.N[1], model.N[0])), K2_tr, np.zeros((model.N[1], model.N[2]))], axis = 1),
                           np.concatenate([np.zeros((model.N[2], model.N[0])), np.zeros((model.N[2], model.N[1])), K3_tr], axis = 1)], axis = 0)
    K1_trte = samples['W'][i,0]*RBF(t_te, model.t_t[0], samples['hyp'][i,0])
    K2_trte = samples['W'][i,1]*RBF(t_te, model.t_t[1], samples['hyp'][i,1])
    K3_trte = samples['W'][i,2]*RBF(t_te, model.t_t[2], samples['hyp'][i,2])
    K_trte = np.concatenate([np.concatenate([K1_trte, np.zeros((model.N[0],Nt_test)), np.zeros((model.N[0],Nt_test))], axis = 1),
                             np.concatenate([np.zeros((model.N[1],Nt_test)), K2_trte, np.zeros((model.N[1],Nt_test))], axis = 1),
                             np.concatenate([np.zeros((model.N[2],Nt_test)), np.zeros((model.N[2],Nt_test)), K3_trte], axis = 1)], axis = 0)
    K1_te = samples['W'][i,0]*RBF(t_te, t_te, samples['hyp'][i,0])
    K2_te = samples['W'][i,1]*RBF(t_te, t_te, samples['hyp'][i,1])
    K3_te = samples['W'][i,2]*RBF(t_te, t_te, samples['hyp'][i,2])
    K_te = np.concatenate([np.concatenate([K1_te, np.zeros((Nt_test,Nt_test)), np.zeros((Nt_test,Nt_test))], axis = 1),
                           np.concatenate([np.zeros((Nt_test,Nt_test)), K2_te, np.zeros((Nt_test,Nt_test))], axis = 1),
                           np.concatenate([np.zeros((Nt_test,Nt_test)), np.zeros((Nt_test,Nt_test)), K3_te], axis = 1)], axis = 0)
    x0_l = np.array([0.5, 1.9, 0.18, 0.15, samples['IC'][i], 0.1, 0.064])
#    X_tr_i = odeint(glyc, x0_l, t, samples['J0'][i], samples['k1'][i], samples['k2'][i], samples['k3'][i], samples['k4'][i], samples['k5'][i], samples['k6'][i], samples['k'][i], samples['ka'][i], samples['q'][i], samples['KI'][i], samples['phi'][i], samples['Np'][i], samples['A'][i])
    X_tr1 = X_tr_i[i,i_t[0],ind[0]] / model.max_X[0]
    X_tr2 = X_tr_i[i,i_t[1],ind[1]] / model.max_X[1]
    X_tr3 = X_tr_i[i,i_t[2],ind[2]] / model.max_X[2]
    X_tr = np.concatenate((X_tr1,X_tr2,X_tr3),axis=0)
    
    L = np.linalg.cholesky(K_tr) 
    X_train_f = np.concatenate((model.X[0],model.X[1],model.X[2]),axis=0)
    X_train_f = X_train_f.flatten('F')
    dX = np.matmul( K_trte.T, np.linalg.solve(np.transpose(L), np.linalg.solve(L,X_train_f.flatten('F')-X_tr.flatten('F'))) )
#    X_ode_i = odeint(glyc, x0_l, t_test, samples['J0'][i], samples['k1'][i], samples['k2'][i], samples['k3'][i], samples['k4'][i], samples['k5'][i], samples['k6'][i], samples['k'][i], samples['ka'][i], samples['q'][i], samples['KI'][i], samples['phi'][i], samples['Np'][i], samples['A'][i])
    X_ode1 = X_ode_i[i,:,ind[0]] / model.max_X[0]
    X_ode2 = X_ode_i[i,:,ind[1]] / model.max_X[1]
    X_ode3 = X_ode_i[i,:,ind[2]] / model.max_X[2]
    X_ode = np.concatenate((X_ode1,X_ode2,X_ode3),axis=0)
        
    mu = X_ode.flatten('F') + dX
    K = K_te - np.matmul(K_trte.T, np.linalg.solve(np.transpose(L), np.linalg.solve(L,K_trte)))
    pred0 = onp.random.multivariate_normal(mu, K)
    pred0 = pred0.reshape((len(ind), Nt_test)).T
    pred0[:,0:1] = model.max_X[0] * pred0[:,0:1]
    pred0[:,1:2] = model.max_X[1] * pred0[:,1:2]
    pred0[:,2:3] = model.max_X[2] * pred0[:,2:3]
    pred = onp.array(X_ode_i[i,:,:])
    pred[:,ind] = pred0
    if not math.isnan( np.amax(np.abs(pred)) ):
        Npred_GP_f += 1
        X_pred_GP.append( pred )

X_pred_GP = np.array(X_pred_GP)

mean_prediction_GP, std_prediction_GP = np.mean(X_pred_GP, axis=0), np.std(X_pred_GP, axis=0)
lower_GP = mean_prediction_GP - 2.0*std_prediction_GP
upper_GP = mean_prediction_GP + 2.0*std_prediction_GP

var_n = ['S_1','S_2','S_3','S_4','N_2','A_3','S_4^{ex}']
i_tr = -1
for i in range(D):
    plt.figure(figsize = (12,6))
    plt.plot(t_star, X_star[:,i], 'r-', label = "True Trajectory of $"+var_n[i]+"(t)$")
    if i in ind :
        i_tr += 1
        plt.plot(t[i_t[i_tr]], X_train[i_t[i_tr],i], 'ro', label = "Training data of $"+var_n[i]+"(t)$")
    else:
        plt.plot(t[0], X_train[0,i], 'ro', label = "Training data of $"+var_n[i]+"(t)$")
    plt.plot(t_test, mean_prediction_GP[:,i], 'g--', label = "MAP Trajectory of $"+var_n[i]+"(t)$")
    plt.fill_between(t_test, lower_GP[:,i], upper_GP[:,i], facecolor='orange', alpha=0.5, label="Two std band")
    plt.xlabel('$t$',fontsize=26)
    plt.ylabel('$'+var_n[i]+'(t)$',fontsize=26)
    plt.legend(loc='upper right', frameon=False, prop={'size': 20})
    if i==1:
        plt.ylim(top= 1.8*X_star[:,i].max(0)) 
    if i==2 or i==5:
        plt.ylim(top= 2.0*X_star[:,i].max(0)) 
    if i==4:
        plt.ylim(top= 1.9*X_star[:,i].max(0)) 
    if i==0 or i==3:
        plt.ylim(top= 1.7*X_star[:,i].max(0)) 
    if i==6:
        plt.ylim(top= 1.5*X_star[:,i].max(0)) 
    tt = 'plots/x_' + str(i+1) + ".png"
    plt.savefig(tt, dpi = 100) 
print(Npred_GP_f)

x0 = np.array([onp.random.uniform(0.15,1.60),onp.random.uniform(0.19,2.10),onp.random.uniform(0.04,0.20),onp.random.uniform(0.10,0.35),onp.random.uniform(0.08,0.30),onp.random.uniform(0.14,2.67),onp.random.uniform(0.05,0.10)])
X_star = odeint(glyc, x0, t_star, J0, k1, k2, k3, k4, k5, k6, k, ka, q, KI, phi, Np, A)
vmap_args_diff_x0 = (samples['J0'], samples['k1'], samples['k2'], samples['k3'], samples['k4'], samples['k5'], samples['k6'], samples['k'], samples['ka'], samples['q'], samples['KI'], samples['phi'], samples['Np'], samples['A'])

@partial(jit, static_argnums=(0,))     
def predict_diff_x0( t_star, J0, k1, k2, k3, k4, k5, k6, k, ka, q, KI, phi, Np, A):
    X = odeint(glyc, x0, t_star, J0, k1, k2, k3, k4, k5, k6, k, ka, q, KI, phi, Np, A)
    return X
    
pred_X_tr_i = lambda J0, k1, k2, k3, k4, k5, k6, k, ka, q, KI, phi, Np, A: predict_diff_x0(t, J0, k1, k2, k3, k4, k5, k6, k, ka, q, KI, phi, Np, A)
X_tr_i = vmap(pred_X_tr_i)(*vmap_args_diff_x0)

pred_X_ode_i = lambda J0, k1, k2, k3, k4, k5, k6, k, ka, q, KI, phi, Np, A: predict_diff_x0(t_test, J0, k1, k2, k3, k4, k5, k6, k, ka, q, KI, phi, Np, A)
X_ode_i = vmap(pred_X_ode_i)(*vmap_args_diff_x0)

X_pred_GP = []
Npred_GP_f = 0
for i in range(num_samples):
    if i % 500 == 0:
        print(i)
    K1_tr = samples['W'][i,0]*RBF(model.t_t[0], model.t_t[0], samples['hyp'][i,0]) + np.eye(model.N[0])*(samples['noise'][i,0] + model.jitter)
    K2_tr = samples['W'][i,1]*RBF(model.t_t[1], model.t_t[1], samples['hyp'][i,1]) + np.eye(model.N[1])*(samples['noise'][i,1] + model.jitter)
    K3_tr = samples['W'][i,2]*RBF(model.t_t[2], model.t_t[2], samples['hyp'][i,2]) + np.eye(model.N[2])*(samples['noise'][i,2] + model.jitter)
    K_tr = np.concatenate([np.concatenate([K1_tr, np.zeros((model.N[0], model.N[1])), np.zeros((model.N[0], model.N[2]))], axis = 1),
                           np.concatenate([np.zeros((model.N[1], model.N[0])), K2_tr, np.zeros((model.N[1], model.N[2]))], axis = 1),
                           np.concatenate([np.zeros((model.N[2], model.N[0])), np.zeros((model.N[2], model.N[1])), K3_tr], axis = 1)], axis = 0)
    K1_trte = samples['W'][i,0]*RBF(t_te, model.t_t[0], samples['hyp'][i,0])
    K2_trte = samples['W'][i,1]*RBF(t_te, model.t_t[1], samples['hyp'][i,1])
    K3_trte = samples['W'][i,2]*RBF(t_te, model.t_t[2], samples['hyp'][i,2])
    K_trte = np.concatenate([np.concatenate([K1_trte, np.zeros((model.N[0],Nt_test)), np.zeros((model.N[0],Nt_test))], axis = 1),
                             np.concatenate([np.zeros((model.N[1],Nt_test)), K2_trte, np.zeros((model.N[1],Nt_test))], axis = 1),
                             np.concatenate([np.zeros((model.N[2],Nt_test)), np.zeros((model.N[2],Nt_test)), K3_trte], axis = 1)], axis = 0)
    K1_te = samples['W'][i,0]*RBF(t_te, t_te, samples['hyp'][i,0])
    K2_te = samples['W'][i,1]*RBF(t_te, t_te, samples['hyp'][i,1])
    K3_te = samples['W'][i,2]*RBF(t_te, t_te, samples['hyp'][i,2])
    K_te = np.concatenate([np.concatenate([K1_te, np.zeros((Nt_test,Nt_test)), np.zeros((Nt_test,Nt_test))], axis = 1),
                           np.concatenate([np.zeros((Nt_test,Nt_test)), K2_te, np.zeros((Nt_test,Nt_test))], axis = 1),
                           np.concatenate([np.zeros((Nt_test,Nt_test)), np.zeros((Nt_test,Nt_test)), K3_te], axis = 1)], axis = 0)
    X_tr1 = X_tr_i[i,i_t[0],ind[0]] / model.max_X[0]
    X_tr2 = X_tr_i[i,i_t[1],ind[1]] / model.max_X[1]
    X_tr3 = X_tr_i[i,i_t[2],ind[2]] / model.max_X[2]
    X_tr = np.concatenate((X_tr1,X_tr2,X_tr3),axis=0)
    
    L = np.linalg.cholesky(K_tr) 
    X_train_f = np.concatenate((model.X[0],model.X[1],model.X[2]),axis=0)
    X_train_f = X_train_f.flatten('F')
    dX = np.matmul( K_trte.T, np.linalg.solve(np.transpose(L), np.linalg.solve(L,X_train_f.flatten('F')-X_tr.flatten('F'))) )
    X_ode1 = X_ode_i[i,:,ind[0]] / model.max_X[0]
    X_ode2 = X_ode_i[i,:,ind[1]] / model.max_X[1]
    X_ode3 = X_ode_i[i,:,ind[2]] / model.max_X[2]
    X_ode = np.concatenate((X_ode1,X_ode2,X_ode3),axis=0)
        
    mu = X_ode.flatten('F') + dX
    K = K_te - np.matmul(K_trte.T, np.linalg.solve(np.transpose(L), np.linalg.solve(L,K_trte)))
    pred0 = onp.random.multivariate_normal(mu, K)
    pred0 = pred0.reshape((len(ind), Nt_test)).T
    pred0[:,0:1] = model.max_X[0] * pred0[:,0:1]
    pred0[:,1:2] = model.max_X[1] * pred0[:,1:2]
    pred0[:,2:3] = model.max_X[2] * pred0[:,2:3]
    pred = onp.array(X_ode_i[i,:,:])
    pred[:,ind] = pred0
    if not math.isnan( np.amax(np.abs(pred)) ):
        Npred_GP_f += 1
        X_pred_GP.append( pred )

X_pred_GP = np.array(X_pred_GP)
mean_prediction_GP, std_prediction_GP = np.mean(X_pred_GP, axis=0), np.std(X_pred_GP, axis=0)
lower_GP = mean_prediction_GP - 2.0*std_prediction_GP
upper_GP = mean_prediction_GP + 2.0*std_prediction_GP

var_n = ['S_1','S_2','S_3','S_4','N_2','A_3','S_4^{ex}']
i_tr = -1
for i in range(D):
    plt.figure(figsize = (12,6))
    plt.plot(t_star, X_star[:,i], 'r-', label = "True Trajectory of $"+var_n[i]+"(t)$")
    plt.plot(t_test, mean_prediction_GP[:,i], 'g--', label = "MAP Trajectory of $"+var_n[i]+"(t)$")
    plt.fill_between(t_test, lower_GP[:,i], upper_GP[:,i], facecolor='orange', alpha=0.5, label="Two std band")
    plt.xlabel('$t$',fontsize=26)
    plt.ylabel('$'+var_n[i]+'(t)$',fontsize=26)
    plt.legend(loc='upper right', frameon=False, prop={'size': 20})
    if i==1:
        plt.ylim(top= 1.8*X_star[:,i].max(0)) 
    if i==2 or i==5:
        plt.ylim(top= 2.0*X_star[:,i].max(0)) 
    if i==4:
        plt.ylim(top= 1.9*X_star[:,i].max(0)) 
    if i==0 or i==3:
        plt.ylim(top= 1.7*X_star[:,i].max(0)) 
    if i==6:
        plt.ylim(top= 1.5*X_star[:,i].max(0)) 
    tt = 'plots/random_x0_x_' + str(i+1) + ".png"
    plt.savefig(tt, dpi = 100)
    
print(Npred_GP_f)

import matplotlib as mpl

def figsize(scale, nplots = 1):
    fig_width_pt = 390.0                          # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0/72.27                       # Convert pt to inch
    golden_mean = (np.sqrt(5.0)-1.0)/2.0            # Aesthetic ratio (you could change this)
    fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
    fig_height = nplots*fig_width*golden_mean              # height in inches
    fig_size = [fig_width,fig_height]
    return fig_size

pgf_with_latex = {                      # setup matplotlib to use latex for output
    "pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
    "text.usetex": True,                # use LaTeX to write all text
    "font.family": "serif",
    "font.serif": [],                   # blank entries should cause plots to inherit fonts from the document
    "font.sans-serif": [],
    "font.monospace": [],
    "axes.labelsize": 20,               # LaTeX default is 10pt font.
    "axes.titlesize": 20,
    "axes.linewidth": 2,
    "font.size": 16,
    "lines.linewidth": 2,
    "legend.fontsize": 20,               # Make the legend/label fonts a little smaller
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "figure.figsize": figsize(1.0),     # default fig size of 0.9 textwidth
    "pgf.preamble": [
        r"\usepackage[utf8x]{inputenc}",    # use utf8 fonts becasue your computer can handle it :)
        r"\usepackage[T1]{fontenc}",        # plots will be generated using this preamble
        ]
    }
mpl.rcParams.update(pgf_with_latex)

def newfig(width, nplots = 1):
    fig = plt.figure(figsize=figsize(width, nplots))
    ax = fig.add_subplot(111)
    return fig, ax

def savefig(filename, crop = True):
    if crop == True:
        plt.savefig('{}.png'.format(filename), bbox_inches='tight', pad_inches=0 , dpi = 100)
    else:
        plt.savefig('{}.png'.format(filename) , dpi = 100)

Data = [samples['J0'], samples['k1'], samples['k2'], samples['k3'], samples['k4'], samples['k5'], samples['k6'], samples['k'], samples['ka'], samples['q'], samples['KI'], samples['phi'], samples['Np'], samples['A'], samples['IC']]

true = [2.5, 100, 6, 16, 100, 1.28, 12, 1.8, 13, 4, 0.52, 0.1, 1, 4, 0.16]
name = [r'$J_0$', r'$k_1$', r'$k_2$', r'$k_3$', r'$k_4$', r'$k_5$', r'$k_6$', r'$k$',r'$\kappa$', r'$q$', r'$K_1$', r'$\varphi$', r'$N$', r'$A$', r'$N_{2,0}$']

X = np.linspace(0.94, 1.06, 100)[:,None]
fig = plt.figure()
fig.set_size_inches(15, 10)
for i in range(2):
    for j in range(7+i):
        ind = i*7 + j
        ax = plt.subplot2grid((2,8), (i,j))
        ax.boxplot([Data[ind]], showfliers=False)
        ax.plot(X, true[ind]*np.ones(X.shape[0]), 'b-')
        ax.set_xticklabels([])
        ax.set_xticks([])
        plt.title(name[ind], y=-0.12)
            
fig.tight_layout()

savefig('plots/box_plot', False)
