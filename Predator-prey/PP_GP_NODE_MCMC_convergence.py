import numpy as np
import matplotlib.pyplot as plt
import pymc3 as pm3

plt.rcParams.update(plt.rcParamsDefault)
plt.rc('font', family='serif')
plt.rcParams.update({'font.size': 16,
                     'lines.linewidth': 2,
                     'axes.labelsize': 20,  # fontsize for x and y labels (was 10)
                     'axes.titlesize': 20,
                     'xtick.labelsize': 16,
                     'ytick.labelsize': 16,
                     'legend.fontsize': 20,
                     'axes.linewidth': 2,
                     "pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
                     "text.usetex": True,                # use LaTeX to write all text
                     })

data2 = np.load("data/par.npy",allow_pickle=True)
data2_IC = np.load("data/IC.npy",allow_pickle=True)
data3 = np.load("data/par_2.npy",allow_pickle=True)
data3_IC = np.load("data/IC_2.npy",allow_pickle=True)

N_var = 5
i0 = 0
ie = 8000
step = 8

a11_1 = data2[i0:ie:step,1:2] # alpha = 1
a13_1 = data2[i0:ie:step,3:4] # beta = -0.1
a22_1 = data2[i0:ie:step,10:11] # gamma = -1.5
a23_1 = data2[i0:ie:step,11:12] # delta = 0.75
IC_1 = data2_IC[i0:ie:step] # 5.0
IC_1 = IC_1[:,None]

a11_2 = data3[i0:ie:step,1:2] # alpha
a13_2 = data3[i0:ie:step,3:4] # beta
a22_2 = data3[i0:ie:step,10:11] # gamma
a23_2 = data3[i0:ie:step,11:12] # delta
IC_2 = data3_IC[i0:ie:step]
IC_2 = IC_2[:,None]

names = [r'$a_{11} \ (\alpha)$',r'$a_{13} \ (\beta)$',r'$a_{22} \ (\gamma)$',r'$a_{23} \ (\delta)$',r'$x_{1,0}$']

N = a11_1.shape[0]
iteration = np.arange(0,N)


data_chain1 = np.concatenate((a11_1,a13_1,a22_1,a23_1,IC_1),axis=-1)
data_chain2 = np.concatenate((a11_2,a13_2,a22_2,a23_2,IC_2),axis=-1)
    
data_traceplot1 = {}
data_traceplot2 = {}
for i,name in enumerate(names):
    data_traceplot1[name] = data_chain1[:,i]
    data_traceplot2[name] = data_chain2[:,i]

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

corr_plot1 = pm3.autocorrplot(data_traceplot1,var_names=names,grid=(1,N_var),figsize=(12,6.5),textsize=18,combined=True)
corr_plot1 = corr_plot1[None,:]
for i in range(N_var):
    corr_plot1[0, i].set_xlabel('Lag Index',fontsize=26)
corr_plot1[0, 0].set_ylabel('Autocorrelation Value',fontsize=26)
plt.savefig("plots/autocorrelation.png", bbox_inches='tight', pad_inches=0.01)

plt.figure(figsize=(12,6.5))
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

plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
plt.tight_layout()
plt.show()
plt.savefig("plots/geweke.png", pad_inches=0.01)

