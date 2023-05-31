# Automating Model Comparison in Factor Graphs
*By Bart van Erp, Wouter Nuijten, Thijs van de Laar and Bert de Vries*
### Submitted to Entropy
---
**Abstract**
Principled Bayesian approaches often tackle state and parameter estimation, and model comparison as two separate tasks.
Although state and parameter estimation have been automated effectively, this has not yet been the case for model comparison, which therefore still requires error-prone and time-consuming manual derivations.
This paper efficiently automates Bayesian model averaging, selection and combination by message passing on a Forney-style factor graph with a custom mixture node. 
Parameter and state inference, and model comparison can then be executed simultaneously using message passing with scale factors.
This approach shortens the model design cycle and allows for the straightforward extension to hierarchical and temporal model priors to accommodate for modeling complicated time-varying processes.


---
This repository contains all experiments of the paper.
