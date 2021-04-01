GP-NODE Code Guide
@author: mohamedazizbhouri

Rk1: The easiest way to download the codes and data is to use the following Google Drive link to get a zip file of the whole repository: https://drive.google.com/file/d/1c3SsMf5hYpyKEDC7bbPmueI_5Ax0o03l/view?usp=sharing

Rk2: The code was tested using the jax version 0.1.73, the jaxlib version 0.1.51, and the numpyro version 0.3.0

###################################################
############## Predator-prey problem ##############
###################################################

The folder "Predator-prey" contains the implementation of the GP-NODE method for a predator-prey problem with dictionary learning as detailed in the paper "Gaussian processes meet NeuralODEs: A Bayesian framework for learning the dynamics of partially observed systems from scarce and noisy data". 

The code "PP_GP_NODE.py" contains such implementation. It generates the following 5 numpy files which are saved in the folder "data":
* "par.npy" (of size: number of samples by length of the dictionary) contains the trace of samples of the inferred dictionary parameters,
* "IC.npy" (of size: number of samples) contains the trace of samples of the inferred initial condition for the variable x_0 of the Predator-Prey system,
* "noise.npy" (of size: number of samples by observable state dimension (2)) contains the trace of the samples of the Gaussian noise variance,
* "hyp.npy" (of size: number of samples by observable state dimension (2)) contains the trace of the samples of the RBF kernel length,
* "W.npy" (of size: number of samples by observable state dimension (2)) contains the trace of the samples of the RBF kernel variance.

The code "PP_GP_NODE.py" also generates the following 3 plots which are saved in the folder "plots":
* "x_1.png" and "x_2.png" which show the learned dynamics versus the true dynamics and the training data of the variables x_1 and x_2 respectively,
* "box_plot.png" and "box_plot_x0.png" which show the uncertainty estimation of the inferred dictionary parameters and initial condition respectively.

The code "PP_GP_NODE_MCMC_convergence.py" performs:
* the Gelman Rubin tests for the non-zero dictionary parameters and the initial condition,
* the Geweke diagnostic whose results are saved in the file "geweke.png" within the folder "plots",
* the autocorrelation estimation as a function of the lag, and the corresponding results are saved in the file "autocorrelation.png" within the folder "plots".
The Geweke diagnostic and the autocorrelation estimation are performed using the traces saved in the folder "data" by running the code "PP_GP_NODE.py". In order to perform the Gelman Rubin tests, two chains need to be considered. For instance, the results obtained for the second chain can be saved in the folder "data" with the extension "_2" in the files names such that we would have "par_2.npy", "IC_2.noy", etc as it is in the proposed repository.

The code "PP_sindy.py" contains the implementation of the SINDY framework for the Predator-prey problem. 5 cases are considered depending on the available datasets as detailed in the paper "Gaussian processes meet NeuralODEs: A Bayesian framework for learning the dynamics of partially observed systems from scarce and noisy data". The code outputs the inferred dictionary parameters and generates 10 plots showing the SINDY predictions versus the true dynamics and the training data for each of the two variables and for each for the 5 cases considered. These plots are saved in the folder "plots_sindy".

###################################################
############# Yeast-Glycolysis problem ############
###################################################

The folder "Yeast-Glycolysis" contains the implementation of the GP-NODE method for the Yeast-Glycolysis system as detailed in the paper "Gaussian processes meet NeuralODEs: A Bayesian framework for learning the dynamics of partially observed systems from scarce and noisy data". 

The code "YG_GP_NODE.py" contains such implementation. It generates the following 4 numpy files which are saved in the folder "data":
* "par_and_IC.npy" (of size: number of samples by number of physical parameters (15)) contains the trace of samples of the inferred physical parameters,
* "noise.npy" (of size: number of samples by observable state dimension (3)) contains the trace of the samples of the Gaussian noise variance,
* "hyp.npy" (of size: number of samples by observable  state dimension (3)) contains the trace of the samples of the RBF kernel length,
* "W.npy" (of size: number of samples by observable  state dimension (3)) contains the trace of the samples of the RBF kernel variance.

The code "YG_GP_NODE.py" also generates the following 15 plots which are saved in the folder "plots":
* "x_1.png" ... "x_7.png" which show the learned dynamics versus the true dynamics and the training data of the variables x_1 ... x_7 respectively,
* "random_x0_x_1.png" ... "random_x0_x_7.png" which show the future forecasts versus the true dynamics of the variables x_1 ... x_7 respectively for unseen initial conditions that are randomly sampled,
* "box_plot.png" which shows the uncertainty estimation of the inferred physical parameters.

The code "YG_GP_NODE_MCMC_convergence.py" performs:
* the Gelman Rubin tests for the physical parameters,
* the Geweke diagnostic whose results are saved in the file "geweke.png" within the folder "plots",
* the autocorrelation estimation as a function of the lag, and the corresponding results are saved in the files "autocorrelation_1.png", "autocorrelation_2.png" and "autocorrelation_3.png" within the folder "plots".
The Geweke diagnostic and the autocorrelation estimation are performed using the traces saved in the folder "data" by running the code "YG_GP_NODE.py". In order to perform the Gelman Rubin tests, two chains need to be considered. For instance, the results obtained for the second chain can be saved in the folder "data" with the extension "_2" in the files names such that we would have "par_and_IC_2.npy" as it is in the proposed repository.

###################################################
######## Human motion capture data problem ########
###################################################

The folder "Human motion" contains the implementation of the GP-NODE method for the human motion capture data problem as detailed in the paper "Gaussian processes meet NeuralODEs: A Bayesian framework for learning the dynamics of partially observed systems from scarce and noisy data". 

First, extract the folder "npODE" from the zip file "npODE.zip" which located in the folder "Human motion". Second, run the "demo_cmu_walking.m" code (located in the folder "Human motion/npODE/exps") in order to obtain the fitting and forecasting error of the npODE method and generate the data in .mat format. The latter is saved in different files in the the folder "Human motion/npODE/exps" under the names "X.mat", "Y.mat", "u_pca.mat" and "v_pca.mat". These files need to be copied to the folder "Human motion/data".

The code "HM_GP_NODE.py" contains the implementation of the GP-NODE method to the human motion capture data problem. First, please consider specifying the case to be considered by assigning the value 'A_' to the variable "case" in the code "HM_GP_NODE.py" if you want to consider the A case as detailed in the paper "Gaussian processes meet NeuralODEs: A Bayesian framework for learning the dynamics of partially observed systems from scarce and noisy data". Otherwise, please consider assigning the value 'B_' to the variable "case" in the code "HM_GP_NODE.py" if you want to consider the B case.

For each case (A or B), the code "HM_GP_NODE.py" generates the following 4 numpy files which are saved in the folder "data":
* case+"par.npy" (of size: number of samples by length of the dictionary) contains the trace of samples of the inferred dictionary parameters,
* case+"noise.npy" (of size: number of samples by observable state dimension (3)) contains the trace of the samples of the Gaussian noise variance,
* case+"hyp.npy" (of size: number of samples by observable state dimension (3)) contains the trace of the samples of the RBF kernel length,
* case+"W.npy" (of size: number of samples by observable state dimension (3)) contains the trace of the samples of the RBF kernel variance.

For each case (A or B), the code "HM_GP_NODE.py" also generates the following 10 plots which are saved in the folder "plots":
* case+"x_1.png", case+"x_2.png" and case+"x_3.png" which show the learned dynamics versus the true dynamics and the training data of the variables x_1, x_2 and x_3 respectively,
* case+"y_27.png", case+"y_34.png", case+"y_37.png", case+"y_39.png", case+"y_42.png" and case+"y_48.png" which show the learned dynamics versus the true dynamics of PCA-recovered y_27, y_34, y_37, y_39, y_42 and y_48 respectively,
* "box_plot.png" which shows the uncertainty estimation of the inferred dictionary parameters.

The code "HM_GP_NODE_MCMC_convergence.py" performs:
* the Gelman Rubin tests for the most significant non-zero dictionary parameters,
* the Geweke diagnostic whose results are saved in the file case+"geweke.png" within the folder "plots",
* the autocorrelation estimation as a function of the lag, and the corresponding results are saved in the files case+"autocorrelation_1.png" and case+"autocorrelation_2.png" within the folder "plots".
The Geweke diagnostic and the autocorrelation estimation are performed using the traces saved in the folder "data" by running the code "HM_GP_NODE.py". In order to perform the Gelman Rubin tests, two chains need to be considered. For instance, the results obtained for the second chain can be saved in the folder "data" with the extension "_2" in the files names such that we would have case+"par_2.npy" as it is in the proposed repository.
Please consider specifying the case to be considered by assigning the value 'A_' or 'B_' to the variable "case" in the code "HM_GP_NODE_MCMC_convergence.py".
