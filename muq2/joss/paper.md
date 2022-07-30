---
title: 'MUQ: The MIT Uncertainty Quantification Library'
tags:
  - Python
  - C++
  - Bayesian Inference
  - Inverse Problems
  - Uncertainty Quantification
authors:
  - name: Matthew Parno^[Corresponding Author.]
    orcid: 0000-0002-9419-2693
    affiliation: 1 # (Multiple affiliations must be quoted)
  - name: Andrew Davis
    orcid: 0000-0002-6023-0989
    affiliation: 2
  - name: Linus Seelinger
    orcid: 0000-0001-8632-8493
    affiliation: 3
affiliations:
 - name: Department of Mathematics, Dartmouth College, Hanover, NH USA
   index: 1
 - name: Courant Institute of Mathematical Sciences, New York University, New York, NY USA
   index: 2
 - name: Institute for Scientific Computing, Heidelberg University, Heidelberg, Germany
   index: 3
date: 22 February 2021
bibliography: paper.bib
---

# Summary

Scientists and engineers frequently rely on mathematical and numerical models to interpret observational data, forecast system behavior, and make decisions. However, unknown and neglected physics, limited and noisy data, and numerical error result in uncertain model predictions. The MIT Uncertainty Quantification library (MUQ) is a modular software framework for defining and solving uncertainty quantification problems involving complex models.  MUQ is written in C++ but uses pybind11 [@pybind11] to provide a nearly comprehensive Python interface.  Users can access nearly all of MUQ's capabilities from either language.

MUQ provides users many commonly used UQ tools and its modular design allows developers to easily modify, extend, and advance existing algorithms. For example, MUQ allows exact sampling of non-Gaussian distributions (e.g., Markov chain Monte Carlo and importance sampling), approximating computationally intensive forward models (e.g., polynomial chaos expansions and Gaussian process regression), working with integral covariance operators (e.g., Gaussian processes and Karhunen-Lo&egrave;ve decompositions), and characterizing predictive uncertainties. The software is designed to support algorithm developers who want to easily construct new algorithms by exploiting a wide variety of existing algorithmic building blocks. Many UQ algorithms are model agnostic: Different physics-based or statistical models can be substituted into the algorithm based on the application. Therefore, MUQ enables users to quickly implement new models and exploit state-of-the art UQ algorithms.  

A suite of documented examples, including Gaussian process regression of Mauna Loa C02 observations, global sensitivity analysis of an Euler-Bernoulli beam, and a hierarchical Bayesian model of groundwater pump-test data, are provided to guide users through the process of implementing their own models and leveraging MUQ's UQ algorithms on quasi-realistic applications.


# Statement of need

Scientists and engineers are increasingly using physical and statistical models to inform policy, system design, and experiments. Although useful tools, models are inherently error prone and assessing predictive capabilities and robustness requires rigorous uncertainty quantification (UQ). The last decade has seen an explosion in the number and complexity of algorithms for characterizing various sources of uncertainty (e.g, @kennedy2001, @conrad2013adaptive, @MLMC, @sargsyan2019embedded, @cotter2013, @cui2016, @han2018stein, @detommaso2018stein). The complexity of many recent advancements makes it difficult to rigorously compare new algorithms against the current state-of-the-art and for users to leverage these new tools on practical applications. Likewise, many interesting models are developed, but due to a lack of a common interface they are often not widely used. MUQ aims to reduce the gap between algorithmic research and application by providing a software pipeline between the algorithmic development community and UQ practitioners.  The goal is to reduce the costly and error prone practice of reimplementing state-of-the-art techniques, lower the barriers preventing widespread use of cutting-edge techniques, and provide an algorithm-agnostic model interface.  MUQ leverages well-known packages such as Eigen3 [@eigenweb], Stan Math [@carpenter2015stan], NLOpt [@nlopt], Sundials [@hindmarsh2005sundials], Nanoflann [@blanco2014nanoflann], and boost [@boost], to help make this possible.

![MUQ allows for complicated models to be constructed by connecting model components on a graph.  Here is a possible graph for a Bayesian inverse problem built on a model for groundwater flow.  MUQ treats each box as a black-box, but if all components can provide derivative information individually, e.g., through adjoint methods, then MUQ can compute gradients, Jacobians, and Hessian actions through the entire graph. \label{fig:graph}](Graph.png)

While MUQ is capable of solving both forward and inverse UQ problems, its primary focus is solving of Bayesian inverse problems with computationally expensive models, as demonstrated in [@ParallelMLMCMC, @davis2020rate], and potentially high dimensional parameter spaces. In comparison, other MCMC and UQ packages, such as Stan [@carpenter2017], BUGS [@lunn2009], or JAGS [@plummer2003], are rooted in the statistics community. MUQ is particularly useful when the target distribution depends on complicated physical models that are difficult to implement and computationally expensive. MUQ employs a semi-intrusive "gray-box" approach (see Figure \autoref{fig:graph}) that enables efficient gradient calculations, through techniques like the adjoint methods used in PDE constrained optimization, but does not rely on automatic differentiation and does not place any restrictions on how the model is implemented (language, solver, etc...).  Other sampling packages, such as PyMC3 [@salvatier2016], Stan [@carpenter2017], and tensorflow-probability [@lao2020], have adopted various probabilistic programming approaches that are more intrusive than MUQ's gray-box approach. Although such tools are useful, they make it more difficult for users to expose efficient gradient evaluation techniques for problems that rely on PDE solvers or other complex forward models. MUQ also has a variety of algorithms (e.g., @cotter2013; @cui2016) for tackling discretizations of infinite-dimensional Bayesian inverse problems and several other novel MCMC implementations, including multi-level [@MLMCMC, @MLMCMCRevised] and multi-index MCMC, local approximation algorithms [@conrad2018, @davis2020rate], and adaptive transport map MCMC [@parno2018transport].


# Acknowledgements

We are grateful for the many users and developers in the MUQ community.  In particular, we would like to acknowledge software contributions from Alexandra Datz, Arnold Song, Brendan West, Devin O'Connor, Taylor Hodgdon, Patrick Conrad, Josephine Westermann, Ki-Tae Kim, Max Liu, and Cassie Lumbrazo.   We would also like to acknowledge financial and technical support from Youssef Marzouk, Matthew Farthing, Peter Bastian, and Robert Scheichl.  

This material is based upon work supported by the National Science Foundation under Grant No. ACI-1550487.

This material is based upon work supported by the US Department of Energy, Office of Advanced Scientific Computing Research, SciDAC (Scientific Discovery through Advanced Computing) program under awards DE-SC0007099 and DE-SC0021226, for the QUEST and FASTMath SciDAC Institutes.

# References
