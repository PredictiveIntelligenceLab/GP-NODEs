#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 17:43:50 2020

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

from scipy import io

class ODE_GP:
    # Initialize the class
    def __init__(self, t, i_t, X_train_ff, x0, dxdt, case):    
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
        
        if case == 'A_':
            self.M = 9*3
        if case == 'B_':
            self.M = 10*3
            
    @partial(jit, static_argnums=(0,))     
    def RBF(self,x1, x2, params):
        diffs = (x1 / params).T - x2 / params
        return np.exp(-0.5 * diffs**2)
 
    def model(self, t, X):        
        noise = sample('noise', dist.LogNormal(0.0, 1.0), sample_shape=(self.D,))
        
        hyp = sample('hyp', dist.Gamma(1.0, 0.5), sample_shape=(self.D,))
        W = sample('W', dist.LogNormal(0.0, 1.0), sample_shape=(self.D,))
        
        m0 = self.M-1
        sigma = 1
        
        tau0 = (m0/(self.M-m0) * (sigma/np.sqrt(1.0*sum(self.N))))
        tau_tilde = sample('tau_tilde', dist.HalfCauchy(1.), sample_shape=(self.D,))
        tau = np.repeat(tau0 * tau_tilde,self.M//self.D)
        
        slab_scale=1
        slab_scale2 = slab_scale**2
        
        slab_df=1
        half_slab_df = slab_df/2
        c2_tilde = sample('c2_tilde', dist.InverseGamma(half_slab_df, half_slab_df))
        c2 = slab_scale2 * c2_tilde
        
        lambd = sample('lambd', dist.HalfCauchy(1.), sample_shape=(self.M,))
        lambd_tilde = tau**2 * c2 * lambd**2 / (c2 + tau**2 * lambd**2)
        par = sample('par', dist.MultivariateNormal(np.zeros(self.M,), np.diag(lambd_tilde)))
        
        # compute kernel
        K_11 = W[0]*self.RBF(self.t_t[0], self.t_t[0], hyp[0]) + np.eye(self.N[0])*(noise[0] + self.jitter)
        K_22 = W[1]*self.RBF(self.t_t[1], self.t_t[1], hyp[1]) + np.eye(self.N[1])*(noise[1] + self.jitter)
        K_33 = W[2]*self.RBF(self.t_t[2], self.t_t[2], hyp[2]) + np.eye(self.N[2])*(noise[2] + self.jitter)
        K = np.concatenate([np.concatenate([K_11, np.zeros((self.N[0], self.N[1])), np.zeros((self.N[0], self.N[2]))], axis = 1),
                            np.concatenate([np.zeros((self.N[1], self.N[0])), K_22, np.zeros((self.N[1], self.N[2]))], axis = 1),
                            np.concatenate([np.zeros((self.N[2], self.N[0])), np.zeros((self.N[2], self.N[1])), K_33], axis = 1)], axis = 0)
        
        # compute mean
        mut = odeint(self.dxdt, self.x0, self.t.flatten(), par)        
        mu1 = mut[self.i_t[0],ind[0]] / self.max_X[0]
        mu2 = mut[self.i_t[1],ind[1]] / self.max_X[1]
        mu3 = mut[self.i_t[2],ind[2]] / self.max_X[2]
        mu = np.concatenate((mu1,mu2,mu3),axis=0)
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
    def predict(self, t_star, par):
        X = odeint(self.dxdt, self.x0, t_star, par)
        return X

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

# load data
mat = io.loadmat('data/X.mat')
dat = mat['X']
dat_max = np.abs(dat).max(0)
dat = dat / dat_max

l_dat = io.loadmat('data/v_pca.mat')
v_pca = l_dat['v_pca']
l_dat = io.loadmat('data/u_pca.mat')
u_pca = l_dat['u_pca']
l_dat = io.loadmat('data/Y.mat')
Y = l_dat['Y'] 

case = 'B_' # 'A_' or 'B_'

if case =='A_':
    def robot_dict(x, t, par):
        x1, x2, x3 = x
        dxdt = [par[0] + par[1]*x1 + par[2]*x2 + par[3]*x3 + par[4]*x1*x2 + par[5]*x1*x3 + par[6]*x2*x3, 
                par[7] + par[8]*x1 + par[9]*x2 + par[10]*x3 + par[11]*x1*x2 + par[12]*x1*x3 + par[13]*x2*x3, 
                par[14] + par[15]*x1 + par[16]*x2 + par[17]*x3 + par[18]*x1*x2 + par[19]*x1*x3 + par[20]*x2*x3]
        return dxdt
if case =='B_':
    def robot_dict(x, t, par):
        x1, x2, x3 = x
        dxdt = [par[18] + par[0]*x1 + par[1]*x2 + par[2]*x3 + par[3]*x1*x2 + par[4]*x1*x3 + par[5]*x2*x3 + par[21]*x1**2 + par[22]*x2**2 + par[23]*x3**2, 
                par[19] + par[6]*x1 + par[7]*x2 + par[8]*x3 + par[9]*x1*x2 + par[10]*x1*x3 + par[11]*x2*x3 + par[24]*x1**2 + par[25]*x2**2 + par[26]*x3**2, 
                par[20] + par[12]*x1 + par[13]*x2 + par[14]*x3 + par[15]*x1*x2 + par[16]*x1*x3 + par[17]*x2*x3 + par[27]*x1**2 + par[28]*x2**2 + par[29]*x3**2]
        return dxdt

key = random.PRNGKey(1234)
D = 3

Nt_star = 82

x0 = dat[0,:]

# Test data
t_star = np.linspace(0, 1, Nt_star)
X_star = dat

# Training data
Nt = 62
t = t_star

i1 = 33
i2 = 48
ind_t = np.concatenate([np.arange(0,i1)[:,None],np.arange(i2,82)[:,None]])
ind_t = ind_t[:,0]

ind_t_test = np.arange(i1,i2)

i_t = []
i_t.append(ind_t)
i_t.append(ind_t)
i_t.append(ind_t)

X_train = dat

ind = [0,1,2]

X1_train = X_train[i_t[0],ind[0]]
X2_train = X_train[i_t[1],ind[1]]
X3_train = X_train[i_t[2],ind[2]]

X_train_ff = []
X_train_ff.append(X1_train)
X_train_ff.append(X2_train)
X_train_ff.append(X3_train)

model = ODE_GP(t[:,None], i_t, X_train_ff, x0, robot_dict, case)
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

np.save('data/'+case+'par',np.array(samples['par']))
np.save('data/'+case+'noise',np.array(samples['noise']))
np.save('data/'+case+'hyp',np.array(samples['hyp']))
np.save('data/'+case+'W',np.array(samples['W']))
 
def RBF(x1, x2, params):
    diffs = (x1 / params).T - x2 / params
    return np.exp(-0.5 * diffs**2)
def Ksin(x, xp, period, lengthscale):         
    K = np.exp(-2.0*np.sin(np.pi*np.abs(x.T-xp)/period)**2/lengthscale**2)
    return K
    
N_fine = 100
t_test = np.linspace(0, 1, Nt_star)
Nt_test = t_test.shape[0]

t_tr = t[:,None] /model.max_t
t_te = t_test[:,None] /model.max_t

vmap_args = (samples['par'],)
pred_X_tr_i = lambda a: model.predict(t, a)
X_tr_i = vmap(pred_X_tr_i)(*vmap_args)

pred_X_ode_i = lambda a: model.predict(t_test, a)
X_ode_i = vmap(pred_X_ode_i)(*vmap_args)

X_pred_GP = []
Npred_GP_f = 0

ind_PCA = [26,33,36,38,41,47]
Y_PCA_GP = []
Y_PCA = np.matmul( np.matmul( dat_max*X_star ,np.diag(np.sqrt(v_pca[:,0]))) ,u_pca )

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
    pred = onp.random.multivariate_normal(mu, K)
    if not math.isnan( np.amax(np.abs(pred)) ):
        Npred_GP_f += 1
        X_pred_GP.append( pred.reshape((D, Nt_test)).T )
        Y_PCA_GP.append( np.matmul( np.matmul( dat_max*np.array(model.max_X)*pred.reshape((D, Nt_test)).T,np.diag(np.sqrt(v_pca[:,0]))) ,u_pca )  )

X_pred_GP = np.array(X_pred_GP)
mean_prediction_GP, std_prediction_GP = np.mean(X_pred_GP, axis=0), np.std(X_pred_GP, axis=0)
lower_GP = mean_prediction_GP - 2.0*std_prediction_GP
upper_GP = mean_prediction_GP + 2.0*std_prediction_GP

for i in range(D):
    plt.figure(figsize = (12,6.5))
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.plot(t_star, X_star[:,i], 'r-', label = "True Trajectory of $x_"+str(i+1)+"(t)$")
    
    plt.plot(t[i_t[i]], X_train[i_t[i],i],'ro', label = "Training data of $x_"+str(i+1)+"(t)$")
    plt.plot(t_test, mean_prediction_GP[:,i], 'g--', label = "MAP Trajectory of $x_"+str(i+1)+"(t)$")
    plt.fill_between(t_test, lower_GP[:,i], upper_GP[:,i], facecolor='orange', alpha=0.5, label="Two std band")
    plt.axvspan(t_star[i1-1], t_star[i2], alpha=0.1, color='blue')
    plt.xlabel('$t$',fontsize=26)
    plt.ylabel('$x_'+str(i+1)+'(t)$',fontsize=26)
    plt.legend(loc='upper right', frameon=False, prop={'size': 20})
    plt.ylim(top= 2.2*upper_GP[:,i].max(0)) 
    tt = 'plots/' + case + 'x_' + str(i+1) + ".png"
    plt.savefig(tt, dpi = 300)  
    
Y_PCA_GP = np.array(Y_PCA_GP)
mean_prediction_Y, std_prediction_Y = np.mean(Y_PCA_GP, axis=0), np.std(Y_PCA_GP, axis=0)
lower_Y = mean_prediction_Y - 2.0*std_prediction_Y
upper_Y = mean_prediction_Y + 2.0*std_prediction_Y

for i in range(len(ind_PCA)):
    plt.figure(figsize = (12,6.5))
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.plot(t_star, Y_PCA[:,ind_PCA[i]], 'r-', label = "True Trajectory of PCA-recovered $y_{"+str(ind_PCA[i]+1)+"}(t)$")
    plt.plot(t_test, mean_prediction_Y[:,ind_PCA[i]], 'g--', label = "MAP Trajectory PCA-recovered $y_{"+str(ind_PCA[i]+1)+"}(t)$")
    plt.fill_between(t_test, lower_Y[:,ind_PCA[i]], upper_Y[:,ind_PCA[i]], facecolor='orange', alpha=0.5, label="Two std band")
    plt.axvspan(t_star[i1-1], t_star[i2], alpha=0.1, color='blue')
    plt.xlabel('$t$',fontsize=26)
    plt.ylabel('PCA-recovered $y_{'+str(ind_PCA[i]+1)+'}(t)$',fontsize=26)
    plt.legend(loc='upper right', frameon=False, prop={'size': 20})
    plt.ylim(top= 2.2*upper_Y[:,ind_PCA[i]].max(0)) 
    tt = 'plots/' + case + 'y_' + str(ind_PCA[i]+1) + ".png"
    plt.savefig(tt, dpi = 300) 

err_obs = []
err_miss = []
for i in range(num_samples):
    err_obs.append( np.sqrt( np.sum( (Y_PCA_GP[i,ind_t,:]-Y[ind_t,:])**2 ) / (Y[ind_t,:].shape[0]*Y[ind_t,:].shape[1]) )  )
    err_miss.append( np.sqrt( np.sum( (Y_PCA_GP[i,ind_t_test,:]-Y[ind_t_test,:])**2 ) / (Y[ind_t_test,:].shape[0]*Y[ind_t_test,:].shape[1]) ) )

err_obs = np.array(err_obs)
err_miss = np.array(err_miss)
print("Error for fitting observed data:",np.mean(err_obs),np.std(err_obs))
print("Error for forecasting missing data:",np.mean(err_miss),np.std(err_miss))
  
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
    "xtick.labelsize": 24,
    "ytick.labelsize": 24,
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
        plt.savefig('{}.png'.format(filename), bbox_inches='tight', pad_inches=0 , dpi = 300)
    else:
        plt.savefig('{}.png'.format(filename) , dpi = 300)

if case == 'A_': # case A
    
    a11 = samples['par'][:,0]
    a12 = samples['par'][:,1]
    a13 = samples['par'][:,2]
    a14 = samples['par'][:,3]
    a15 = samples['par'][:,4]
    a16 = samples['par'][:,5]
    a17 = samples['par'][:,6]
    a21 = samples['par'][:,7]
    a22 = samples['par'][:,8]
    a23 = samples['par'][:,9]
    a24 = samples['par'][:,10]
    a25 = samples['par'][:,11]
    a26 = samples['par'][:,12]
    a27 = samples['par'][:,13]
    a31 = samples['par'][:,14]
    a32 = samples['par'][:,15]
    a33 = samples['par'][:,16]
    a34 = samples['par'][:,17]
    a35 = samples['par'][:,18]
    a36 = samples['par'][:,19]
    a37 = samples['par'][:,20]
        
    Data = [a11,a12,a13,a14,a15,a16,a17,a21,a22,a23,a24,a25,a26,a27,a31,a32,a33,a34,a35,a36,a37]
                
    name = [r'$a_{11}$',r'$a_{12}$',r'$a_{13}$',r'$a_{14}$',r'$a_{15}$',r'$a_{16}$',r'$a_{17}$',r'$a_{21}$',r'$a_{22}$',r'$a_{23}$',r'$a_{24}$',r'$a_{25}$',r'$a_{26}$',r'$a_{27}$',r'$a_{31}$',r'$a_{32}$',r'$a_{33}$',r'$a_{34}$',r'$a_{35}$',r'$a_{36}$',r'$a_{37}$']
   
if case == 'B_': # case B
    
    a11 = samples['par'][:,18]
    a12 = samples['par'][:,0]
    a13 = samples['par'][:,1]
    a14 = samples['par'][:,2]
    a15 = samples['par'][:,3]
    a16 = samples['par'][:,4]
    a17 = samples['par'][:,5]
    a18 = samples['par'][:,21]
    a19 = samples['par'][:,22]
    a110 = samples['par'][:,23]
    
    a21 = samples['par'][:,19]
    a22 = samples['par'][:,6]
    a23 = samples['par'][:,7]
    a24 = samples['par'][:,8]
    a25 = samples['par'][:,9]
    a26 = samples['par'][:,10]
    a27 = samples['par'][:,11]
    a28 = samples['par'][:,24]
    a29 = samples['par'][:,25]
    a210 = samples['par'][:,26]
    
    a31 = samples['par'][:,20]
    a32 = samples['par'][:,12]
    a33 = samples['par'][:,13]
    a34 = samples['par'][:,14]
    a35 = samples['par'][:,15]
    a36 = samples['par'][:,16]
    a37 = samples['par'][:,17]
    a38 = samples['par'][:,27]
    a39 = samples['par'][:,28]
    a310 = samples['par'][:,29]
     
    Data = [a11,a12,a13,a14,a15,a16,a17,a18,a19,a110,a21,a22,a23,a24,a25,a26,a27,a28,a29,a210,a31,a32,a33,a34,a35,a36,a37,a38,a39,a310]
            
    name = [r'$a_{11}$',r'$a_{12}$',r'$a_{13}$',r'$a_{14}$',r'$a_{15}$',r'$a_{16}$',r'$a_{17}$',r'$a_{18}$',r'$a_{19}$',r'$a_{110}$',r'$a_{21}$',r'$a_{22}$',r'$a_{23}$',r'$a_{24}$',r'$a_{25}$',r'$a_{26}$',r'$a_{27}$',r'$a_{28}$',r'$a_{29}$',r'$a_{210}$',r'$a_{31}$',r'$a_{32}$',r'$a_{33}$',r'$a_{34}$',r'$a_{35}$',r'$a_{36}$',r'$a_{37}$',r'$a_{38}$',r'$a_{39}$',r'$a_{310}$']

fig7, ax7 = plt.subplots(figsize=(20, 10))
ax7.boxplot(Data, showfliers=False, labels=name)
savefig('plots/' + case + 'box_plot', True)   
