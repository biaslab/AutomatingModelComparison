# Automating Model Comparison in Factor Graphs
*By Bart van Erp, Wouter Nuijten, Thijs van de Laar and Bert de Vries*
### Published in the special issue on Probabilistic Models in Machine and Human Learning of the Entropy journal (2023).
---
**Abstract**

Bayesian state and parameter estimation are automated effectively in a variety of probabilistic programming languages. The process of model comparison on the other hand, which still requires error-prone and time-consuming manual derivations, is often overlooked despite its importance. This paper efficiently automates Bayesian model averaging, selection, and combination by message passing on a Forney-style factor graph with a custom mixture node. Parameter and state inference, and model comparison can then be executed simultaneously using message passing with scale factors. This approach shortens the model design cycle and allows for the straightforward extension to hierarchical and temporal model priors to accommodate for modeling complicated time-varying processes.


---
This repository contains all experiments of the paper.

## Installation instructions
1. Install [Julia](https://julialang.org/)

2. activate environment (using `]` and backspace you can switch between the regular prompt and package manager)
```julia
>> ] activate .
```

3. instantiate environment (only required once)
```julia
>> ] instantiate
```

4. start Pluto
```julia
>> using Pluto; Pluto.run()
```
