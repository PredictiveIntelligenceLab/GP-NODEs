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

case = 'A_' # 'A_' or 'B_'
N_var = 10
i0 = 0
ie = 8000
step = 8
if case == 'A_':
    # A dictionary
    data2 = np.load("data/A_par.npy",allow_pickle=True)
    data3 = np.load("data/A_par_2.npy",allow_pickle=True)
    a13_1 = data2[i0:ie:step,2:3] 
    a14_1 = data2[i0:ie:step,3:4]
    a16_1 = data2[i0:ie:step,5:6]
    a17_1 = data2[i0:ie:step,6:7]
    a22_1 = data2[i0:ie:step,8:9]
    a25_1 = data2[i0:ie:step,11:12]
    a26_1 = data2[i0:ie:step,12:13]
    a34_1 = data2[i0:ie:step,17:18]
    a35_1 = data2[i0:ie:step,18:19]
    a36_1 = data2[i0:ie:step,19:20]
    
    a13_2 = data3[i0:ie:step,2:3] 
    a14_2 = data3[i0:ie:step,3:4]
    a16_2 = data3[i0:ie:step,5:6]
    a17_2 = data3[i0:ie:step,6:7]
    a22_2 = data3[i0:ie:step,8:9]
    a25_2 = data3[i0:ie:step,11:12]
    a26_2 = data3[i0:ie:step,12:13]
    a34_2 = data3[i0:ie:step,17:18]
    a35_2 = data3[i0:ie:step,18:19]
    a36_2 = data3[i0:ie:step,19:20]
    
else:
    # case B
    data2 = np.load("data/B_par.npy",allow_pickle=True)
    data3 = np.load("data/B_par_2.npy",allow_pickle=True)
    a13_1 = data2[i0:ie:step,1:2] 
    a14_1 = data2[i0:ie:step,2:3]
    a16_1 = data2[i0:ie:step,4:5]
    a17_1 = data2[i0:ie:step,5:6]
    a22_1 = data2[i0:ie:step,6:7]
    a25_1 = data2[i0:ie:step,9:10]
    a26_1 = data2[i0:ie:step,10:11]
    a34_1 = data2[i0:ie:step,14:15]
    a35_1 = data2[i0:ie:step,15:16]
    a36_1 = data2[i0:ie:step,16:17]
    
    a13_2 = data3[i0:ie:step,1:2] 
    a14_2 = data3[i0:ie:step,2:3]
    a16_2 = data3[i0:ie:step,4:5]
    a17_2 = data3[i0:ie:step,5:6]
    a22_2 = data3[i0:ie:step,6:7]
    a25_2 = data3[i0:ie:step,9:10]
    a26_2 = data3[i0:ie:step,10:11]
    a34_2 = data3[i0:ie:step,14:15]
    a35_2 = data3[i0:ie:step,15:16]
    a36_2 = data3[i0:ie:step,16:17]

names = [r'$a_{13}$',r'$a_{14}$',r'$a_{16}$',r'$a_{17}$',r'$a_{22}$',r'$a_{25}$',r'$a_{26}$',r'$a_{34}$',r'$a_{35}$',r'$a_{36}$']

data_chain1 = np.concatenate((a13_1,a14_1,a16_1,a17_1,a22_1,a25_1,a26_1,a34_1,a35_1,a36_1),axis=-1) # 2000 x 5
data_chain2 = np.concatenate((a13_2,a14_2,a16_2,a17_2,a22_2,a25_2,a26_2,a34_2,a35_2,a36_2),axis=-1) # 2000 x 5

N = a13_1.shape[0]
iteration = np.arange(0,N)
   
N_per_block = 5
data_traceplot1_1 = {}
data_traceplot1_2 = {}
data_traceplot2_1 = {}
data_traceplot2_2 = {}

j = 0
for i,name in enumerate(names[j:j+N_per_block]):
    data_traceplot1_1[name] = data_chain1[:,j+i]
    data_traceplot2_1[name] = data_chain2[:,j+i]

j = N_per_block
for i,name in enumerate(names[j:j+N_per_block]):
    data_traceplot1_2[name] = data_chain1[:,j+i]
    data_traceplot2_2[name] = data_chain2[:,j+i]

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
plt.savefig("plots/"+case+"autocorrelation_1.png", bbox_inches='tight', pad_inches=0.01)

j = N_per_block
corr_plot1_2 = pm3.autocorrplot(data_traceplot1_2,var_names=names[j:j+N_per_block],grid=(1,N_per_block),figsize=(12,6.5),textsize=18,combined=True)
corr_plot1_2 = corr_plot1_2[None,:]
for i in range(N_per_block):
    corr_plot1_2[0, i].set_xlabel('Lag Index',fontsize=26)
corr_plot1_2[0, 0].set_ylabel('Autocorrelation Value',fontsize=26)
plt.savefig("plots/"+case+"autocorrelation_2.png", bbox_inches='tight', pad_inches=0.01)

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

plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left')
plt.tight_layout()
plt.show()
plt.savefig("plots/"+case+"geweke.png", pad_inches=0.01)

