import numpy as np
import matplotlib.pyplot as plt
import pymc3 as pm3

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

data2 = np.load("data/par_and_IC.npy",allow_pickle=True).T
data3 = np.load("data/par_and_IC_2.npy",allow_pickle=True).T

N_var = 15
i0 = 0
ie = 8000
step = 8

J0_1 = data2[i0:ie:step,0:1] # J0 = 2.5
k1_1 = data2[i0:ie:step,1:2] # k1 = 100.
k2_1 = data2[i0:ie:step,2:3] # k2 = 6.
k3_1 = data2[i0:ie:step,3:4] # k3 = 16.
k4_1 = data2[i0:ie:step,4:5] # k4 = 100.
k5_1 = data2[i0:ie:step,5:6] # k5 = 1.28
k6_1 = data2[i0:ie:step,6:7] # k6 = 12.
k_1 = data2[i0:ie:step,7:8] # k = 1.8
ka_1 = data2[i0:ie:step,8:9] # ka = 13.
q_1 = data2[i0:ie:step,9:10] # q = 4.
KI_1 = data2[i0:ie:step,10:11] # KI = 0.52
phi_1 = data2[i0:ie:step,11:12] # phi = 0.1
Np_1 = data2[i0:ie:step,12:13] # Np = 1.
A_1 = data2[i0:ie:step,13:14] # A = 4.
IC_1 = data2[i0:ie:step,14:15] # IC = 0.16

J0_2 = data3[i0:ie:step,0:1] # J0 = 2.5
k1_2 = data3[i0:ie:step,1:2] # k1 = 100.
k2_2 = data3[i0:ie:step,2:3] # k2 = 6.
k3_2 = data3[i0:ie:step,3:4] # k3 = 16.
k4_2 = data3[i0:ie:step,4:5] # k4 = 100.
k5_2 = data3[i0:ie:step,5:6] # k5 = 1.28
k6_2 = data3[i0:ie:step,6:7] # k6 = 12.
k_2 = data3[i0:ie:step,7:8] # k = 1.8
ka_2 = data3[i0:ie:step,8:9] # ka = 13.
q_2 = data3[i0:ie:step,9:10] # q = 4.
KI_2 = data3[i0:ie:step,10:11] # KI = 0.52
phi_2 = data3[i0:ie:step,11:12] # phi = 0.1
Np_2 = data3[i0:ie:step,12:13] # Np = 1.
A_2 = data3[i0:ie:step,13:14] # A = 4.
IC_2 = data3[i0:ie:step,14:15] # IC = 0.16

names = [r'$J_0$', r'$k_1$', r'$k_2$', r'$k_3$', r'$k_4$', r'$k_5$', r'$k_6$', r'$k$',r'$\kappa$', r'$q$', r'$K_1$', r'$\varphi$', r'$N$', r'$A$', r'$N_{2,0}$']

N = J0_1.shape[0]
iteration = np.arange(0,N)


data_chain1 = np.concatenate((J0_1,k1_1,k2_1,k3_1,k4_1,k5_1,k6_1,k_1,ka_1,q_1,KI_1,phi_1,Np_1,A_1,IC_1),axis=-1) # 2000 x 5
data_chain2 = np.concatenate((J0_2,k1_2,k2_2,k3_2,k4_2,k5_2,k6_2,k_2,ka_2,q_2,KI_2,phi_2,Np_2,A_2,IC_2),axis=-1) # 2000 x 5
    
N_per_block = 5
data_traceplot1_1 = {}
data_traceplot1_2 = {}
data_traceplot1_3 = {}
data_traceplot2_1 = {}
data_traceplot2_2 = {}
data_traceplot2_3 = {}

j = 0
for i,name in enumerate(names[j:j+N_per_block]):
    data_traceplot1_1[name] = data_chain1[:,j+i]
    data_traceplot2_1[name] = data_chain2[:,j+i]

j = N_per_block
for i,name in enumerate(names[j:j+N_per_block]):
    data_traceplot1_2[name] = data_chain1[:,j+i]
    data_traceplot2_2[name] = data_chain2[:,j+i]

j = 2*N_per_block
for i,name in enumerate(names[j:j+N_per_block]):
    data_traceplot1_3[name] = data_chain1[:,j+i]
    data_traceplot2_3[name] = data_chain2[:,j+i]

for i in range(N_var): 

    chain1 = data_chain1[:,i:i+1]
    chain2 = data_chain2[:,i:i+1]
    
    burn_in = 0
    length = (ie-i0)//step
    
    n = chain1[burn_in:burn_in+length].shape[0]
    
    W = (chain1[burn_in:burn_in+length].std()**2 + chain2[burn_in:burn_in+length].std()**2)/2
    mean1 = chain1[burn_in:burn_in+length].mean()
    mean2 = chain2[burn_in:burn_in+length].mean()
    mean = (mean1 + mean2)/2
    B = n * ((mean1 - mean)**2 + (mean2 - mean)**2)
    var_theta = (1 - 1/n) * W + 1/n*B
    print("Gelman-Rubin Diagnostic: ", np.sqrt(var_theta/W))

j = 0
corr_plot1_1 = pm3.autocorrplot(data_traceplot1_1,var_names=names[j:j+N_per_block],grid=(1,N_per_block),figsize=(12,6.5),textsize=18,combined=True)
corr_plot1_1 = corr_plot1_1[None,:]
for i in range(N_per_block):
    corr_plot1_1[0, i].set_xlabel('Lag Index',fontsize=26)
corr_plot1_1[0, 0].set_ylabel('Autocorrelation Value',fontsize=26)
plt.savefig("plots/autocorrelation_1.png", bbox_inches='tight', pad_inches=0.01)

j = N_per_block
corr_plot1_2 = pm3.autocorrplot(data_traceplot1_2,var_names=names[j:j+N_per_block],grid=(1,N_per_block),figsize=(12,6.5),textsize=18,combined=True)
corr_plot1_2 = corr_plot1_2[None,:]
for i in range(N_per_block):
    corr_plot1_2[0, i].set_xlabel('Lag Index',fontsize=26)
corr_plot1_2[0, 0].set_ylabel('Autocorrelation Value',fontsize=26)
plt.savefig("plots/autocorrelation_2.png", bbox_inches='tight', pad_inches=0.01)

j = 2*N_per_block
corr_plot1_3 = pm3.autocorrplot(data_traceplot1_3,var_names=names[j:j+N_per_block],grid=(1,N_per_block),figsize=(12,6.5),textsize=18,combined=True)
corr_plot1_3 = corr_plot1_3[None,:]
for i in range(N_per_block):
    corr_plot1_3[0, i].set_xlabel('Lag Index',fontsize=26)
corr_plot1_3[0, 0].set_ylabel('Autocorrelation Value',fontsize=26)
plt.savefig("plots/autocorrelation_3.png", bbox_inches='tight', pad_inches=0.01)

plt.figure(figsize=(12,7))
for i in range(data_chain1.shape[1]): 
    gw_plot = pm3.geweke(data_chain1[:,i],.1,.5,20)
    plt.scatter(gw_plot[:,0],gw_plot[:,1],label="%s"%names[i])
plt.axhline(-1.98, c='r')
plt.axhline(1.98, c='r')
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.xlabel("Subchain sample number",fontsize=26)
plt.ylabel("Geweke z-score",fontsize=26) 
plt.title('Geweke Plot Comparing first 10$\%$ and Slices of the Last 50$\%$ of Chain')

plt.legend(bbox_to_anchor=(1.0, 1.2), loc='upper left')
plt.tight_layout()
plt.show()
plt.savefig("plots/geweke.png", pad_inches=0.01)
