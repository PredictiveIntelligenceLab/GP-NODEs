## Gaussian processes meet NeuralODEs

Code and data accompanying the manuscript titled "Gaussian processes meet NeuralODEs: A Bayesian framework for learning the dynamics of partially observed systems from scarce and noisy data", authored by Mohamed Aziz Bhouri and Paris Perdikaris.

## Abstract

This paper presents a machine learning framework for Bayesian systems identification from partial, noisy and irregular observations of nonlinear dynamical systems. The proposed method takes advantage of recent developments in differentiable programming to propagate gradient information through ordinary differential equation solvers and perform Bayesian inference with respect to unknown model parameters using Markov Chain Monte Carlo  and Gaussian Process priors over the observed system states. This allows us to exploit temporal correlations in the observed data, and efficiently infer posterior distributions over plausible models with quantified uncertainty. Moreover, the use of sparsity-promoting priors such as the Finnish Horseshoe for free model parameters enables the discovery of interpretable and parsimonious representations for the underlying latent dynamics. A series of numerical studies is presented to demonstrate the effectiveness of the proposed methods including predator-prey systems, systems biology, and a 50-dimensional human motion dynamical system. Taken together, our findings put forth a novel, flexible and robust workflow for data-driven model discovery under uncertainty.

## Citation

TBA
