/***
## Overview
The goal of this example is to demonstrate the use of MUQ's MCMC stack by sampling
a simple bivariate Gaussian density.  To keep things as simple as possible, we
employ a Metropolis-Hastings transition kernel with a simple random walk proposal.
The idea is to introduce the MUQ MCMC workflow without additional the additional
complexities that come from more challenging target densities or more complicated
MCMC algorithms.

### Background
Let $x$ denote a random variable taking values in a space $\mathcal{X}$, and let $\pi(x)$ denote the
probability density of $x$.  In many cases, we cannot compute expectations with
respect to $\pi(x)$ analytically, and we need to resort to some sort of numerical
integration.  Typically, such approaches approximate an expectation $\mathbb{E}_x \left[f(x)\right]$,
through some sort of weighted sum that takes the form
$$
\mathbb{E}_x\left[f(x)\right] \approx \sum_{k=1}^K w_k f\left(x^{(k)}\right).
$$
In standard Monte Carlo procedures, the weights are constant $w_k=\frac{1}{K}$ and
points $x^{(k)}$ are independent samples of $\pi(x)$.   However, generating
independent samples is not always possible for general $\pi(x)$.  Markov chain
Monte Carlo is one way to get around this.   The basic idea is to create a sequence
of points (e.g., a chain) that are correlated, but for large $K$, can still be used in a Monte
Carlo approximation.  We further restrict that the chain is Markov, which simply means that $x^{(k)}$
depends only on the previous step in the chain $x^{(k-1)}$.

#### Transition Kernels
The transition from $x^{(k-1)}$ to $x^{(k)}$ is controlled by a probability distribution
over $\mathcal{X}$ called the transition kernel.  This distribution is consturcted
to ensure that the chain can be used to form a Monte Carlo approximation as $K\rightarrow \infty$.
More precisely, the transition kernel is constructed to ensure that the chain forms
a stationary random process with stationary distribution $\pi(x)$ and is ergodic,
which ensures the sample mean will converge to the true expecation almost surely
as $K\rightarrow \infty$.  Note that establishing a central limit theorem and
studying the variance of the MCMC estimator requires additional technical conditions
on the transition kernel.  See <a href=https://www.springer.com/us/book/9780387212395>Robert and Casella, <i>Monte Carlo Statistical Methods</i></a>
for more details.

#### Metropolis-Hastings Rule
While not the only option, the Metropolis-Hastings rule is one of the most common
methods for constructing an appropriate transition kernel.  The idea is to start
with a proposal distribution $q(x | x^{(k-1)})$ and then "correct" the proposal
with an accept/reject step to ensure ergodicity.  The basic procedure is as follows

1. Generate a sample $x^\prime$ from the proposal distribution
$$
x^\prime \sim q(x | x^{(k-1)}).
$$

2. Compute the acceptance probability $\gamma$
$$
\gamma = \frac{\pi(x^\prime)}{\pi(x^{(k-1)})} \frac{q(x^{(k-1)} | x^\prime)}{q(x^\prime | x^{(k-1)})}.
$$

3.  Accept the proposed step $x^\prime$ as the next state with probability $\gamma$
$$
x^{(k)} = \left\\{ \begin{array}{lll} x^\prime & \text{with probability} & \gamma\\\\ x^{(k-1)} & \text{with probability} & 1.0-\gamma\end{array}\right\\}.
$$

#### Proposal Distributions
Clearly, the proposal density $q(x | x^{(k-1)})$ is a key component of the Metropolis-Hastings
transition kernel.  Fortunately though, there is incredible flexibility in the
choice of proposal; the Metropolis-Hastings correction step ensures, at least
asymptotically, that the resulting chain will be ergodic.   While not
the most efficient, a common choice of proposal is an isotropic Gaussian distribution
centered at $x^{(k-1)}$.  The resulting algorithm is commonly referred to as
the "Random Walk Metropolis" (RWM) algorithm.  Below, we will use the RWM
algorithm in MUQ to sample a simple bivariate Gaussian target density $\pi(x)$.

### MCMC in MUQ
The MCMC classes in MUQ are analogous to the mathematical components of an MCMC
algorithm: there is a base class representing the chain, another base class representing
the transition kernel, and for Metropolis-Hastings, a third base class representing
the proposal distribution.  The RWM algorithm can be constructed by combining an
instance of the "SingleChainMCMC" chain class, with an instance of the "MHKernel"
transition kernel class, and an instance of the "RandomWalk" proposal class.  In
later examples, the flexibility of this structure will become clear, as we construct
algorithms of increasing complexity by simply exchanging each of these components
with alternative chains, kernels, and proposals.

## Problem Description
Use the Random Walk Metropolis algorithm to sample a two-dimensional normal
distribution with mean
$$
\mu = \left[\begin{array}{c} 1.0\\\\ 2.0\end{array}\right],
$$
and covariance
$$
\Sigma = \left[\begin{array}{cc} 1.0 & 0.8 \\\\ 0.8 & 1.5 \end{array}\right].
$$
*/

/***
## Implementation
To sample the Gaussian target, the code needs to do four things:

1. Define the target density and set up a sampling problem.

2. Construct the RWM algorithm.

3. Run the MCMC algorithm.

4. Analyze the results.

### Include statements
Include the necessary header files from MUQ and elsewhere.  Notice our use of the
<a href=https://www.boost.org/doc/libs/1_65_1/doc/html/property_tree.html>boost::property_tree class</a>.
We use property tree's to pass in algorithm parameters, and even to define the
chain, kernel, and proposal themselves.
*/
#include "MUQ/Modeling/Distributions/Gaussian.h"
#include "MUQ/Modeling/Distributions/Density.h"

#include "MUQ/SamplingAlgorithms/SamplingProblem.h"
#include "MUQ/SamplingAlgorithms/SingleChainMCMC.h"
#include "MUQ/SamplingAlgorithms/MCMCFactory.h"
#include "MUQ/SamplingAlgorithms/Diagnostics.h"

#include "MUQ/Utilities/RandomGenerator.h"

#include <boost/property_tree/ptree.hpp>

namespace pt = boost::property_tree;
using namespace muq::Modeling;
using namespace muq::SamplingAlgorithms;
using namespace muq::Utilities;


int main(){

  /***
  ### 1. Define the target density and set up sampling problem
  MUQ has extensive tools for combining many model compoenents into larger
  more complicated models.  The AbstractSamplingProblem base class and its
  children, like the SamplingProblem class, define the interface between
  sampling algorithms like MCMC and the models and densities they work with.

  Here, we create a very simple target density and then construct a SamplingProblem
  directly from the density.
  */
  /***
  Define the Target Density:
  */
  Eigen::VectorXd mu(2);
  mu << 1.0, 2.0;

  Eigen::MatrixXd cov(2,2);
  cov << 1.0, 0.8,
         0.8, 1.5;

  auto targetDensity = std::make_shared<Gaussian>(mu, cov)->AsDensity(); // standard normal Gaussian

  /***
  Create the Sampling Problem:
  */
  auto problem = std::make_shared<SamplingProblem>(targetDensity);

  /***
  ### 2. Construct the RWM algorithm
  One of the easiest ways to define an MCMC algorithm is to put all of the algorithm
  parameters, including the kernel and proposal definitions, in a property tree
  and then let MUQ construct each of the algorithm components: chain, kernel, and
  proposal.

  The boost property tree will have the following entries:

  - NumSamples : 10000
  - KernelList "Kernel1"
  - Kernel1
    * Method : "MHKernel"
    * Proposal : "MyProposal"
    * MyProposal
      + Method : "MHProposal"
      + ProposalVariance : 0.5

At the base level, we specify the number of steps in the chain with the entry "NumSamples".
  Note that this number includes any burnin samples.   The kernel is then defined
  in the "KernelList" entry.  The value, "Kernel1", specifies a block in the
  property tree with the kernel definition.  In the "Kernel1" block, we set the
  kernel to "MHKernel," which specifies that we want to use the Metropolis-Hastings
  kernel.  We also tell MUQ to use the "MyProposal" block to define the proposal.
  The proposal method is specified as "MHProposal", which is the random walk
  proposal used in the RWM algorithm, and the proposal variance is set to 0.5.
  */

  // parameters for the sampler
  pt::ptree pt;
  pt.put("NumSamples", 1e4); // number of MCMC steps
  pt.put("BurnIn", 1e3);
  pt.put("PrintLevel",3);
  pt.put("KernelList", "Kernel1"); // Name of block that defines the transition kernel
  pt.put("Kernel1.Method","MHKernel");  // Name of the transition kernel class
  pt.put("Kernel1.Proposal", "MyProposal"); // Name of block defining the proposal distribution
  pt.put("Kernel1.MyProposal.Method", "MHProposal"); // Name of proposal class
  pt.put("Kernel1.MyProposal.ProposalVariance", 2.5); // Variance of the isotropic MH proposal

  /***
  Once the algorithm parameters are specified, we can pass them to the CreateSingleChain
  function of the MCMCFactory class to create an instance of the MCMC algorithm we defined in the
  property tree.
  */
  Eigen::VectorXd startPt = mu;
  auto mcmc = MCMCFactory::CreateSingleChain(pt, problem);

  /***
  ### 3. Run the MCMC algorithm
  We are now ready to run the MCMC algorithm.  Here we start the chain at the
  target densities mean.   The resulting samples are returned in an instance
  of the SampleCollection class, which internally holds the steps in the MCMC chain
  as a vector of weighted SamplingState's.
  */
  std::shared_ptr<SampleCollection> samps = mcmc->Run(startPt);

  /***
  ### 4. Analyze the results

  When looking at the entries in a SampleCollection, it is important to note that
  the states stored by a SampleCollection are weighted even in the MCMC setting.
  When a proposal $x^\prime$ is rejected, instead of making a copy of $x^{(k-1)}$
  for $x^{(k)}$, the weight on $x^{(k-1)}$ is simply incremented.  This is useful
  for large chains in high dimensional parameter spaces, where storing all duplicates
  could quickly consume available memory.

  The SampleCollection class provides several functions for computing sample moments.
  For example, here we compute the mean, variance, and third central moment.
  While the third moment is actually a tensor, here we only return the marginal
  values, i.e., $\mathbb{E}_x[(x_i-\mu_i)^3]$ for each $i$.
  */
  Eigen::VectorXd sampMean = samps->Mean();
  std::cout << "\nSample Mean = \n" << sampMean.transpose() << std::endl;

  Eigen::VectorXd sampVar = samps->Variance();
  std::cout << "\nSample Variance = \n" << sampVar.transpose() << std::endl;

  Eigen::MatrixXd sampCov = samps->Covariance();
  std::cout << "\nSample Covariance = \n" << sampCov << std::endl;

  Eigen::VectorXd sampMom3 = samps->CentralMoment(3);
  std::cout << "\nSample Third Moment = \n" << sampMom3 << std::endl << std::endl;

  /***
    #### Statistical Accuracy
    In addition to looking at moments and other expectations with respect to the target distribution, we can also look at the 
    statistical accuracy of the mean estimate.  Specifically, we can look at the Monte Carlo Standard Error (MCSE) and effective
    sample size of the chain.   There are multiple ways of computing these quantities and MUQ provides implementations of both
    batch and spectral methods.   Batch methods use the means of different subsets of the chain to estimate the estimator variance
    whereas spectral methods look at the autocorrelation of the chain.   MUQ uses the spectral method described in
    [Monte Carlo errors with less error](https://doi.org/10.1016/S0010-4655(03)00467-3).  
  */

  Eigen::VectorXd batchESS = samps->ESS("Batch");
  Eigen::VectorXd batchMCSE = samps->StandardError("Batch");

  Eigen::VectorXd spectralESS = samps->ESS("Wolff");
  Eigen::VectorXd spectralMCSE = samps->StandardError("Wolff");

  std::cout << "ESS:\n";
  std::cout << "  Batch:    " << batchESS.transpose() << std::endl;
  std::cout << "  Spectral: " << spectralESS.transpose() << std::endl;
  std::cout << "MCSE:\n";
  std::cout << "  Batch:    " << batchMCSE.transpose() << std::endl;
  std::cout << "  Spectral: " << spectralMCSE.transpose() << std::endl;

  /***
    ### 5. Compute convergence diagnostics
    To quantitatively assess whether the chain has converged, we need to run multiple
    chains and then compare the results.  Below we run 3 more independent chains (for a total of 4)
    and then analyze convergence using the commonly employed $\hat{R}$ diagnostic.  A value of $\hat{R}$ close to $1$ (e.g., $<1.01$)
    implies that the chains have converged.  More discussion on this point, as well as a description of the split-rank approach used
    in MUQ to estimat $\hat{R}$, can be found in [Rank-normalization, folding, and localization: An improved R for assessing convergence of MCMC](https://arxiv.org/pdf/1903.08008.pdf).

    Notice that a new MCMC sampler is defined each time with a randomly selected starting point.  If we simply called `mcmc.Run()`
    multiple times, the sampler would always pick up where it left off.  For the estimation of $\hat{R}$, it is also important that 
    the initials states of these chains be drawn from a distribution that is more "diffuse" than the target distribution.
  */
  pt.put("PrintLevel",0);
  int numChains = 4;
  std::vector<std::shared_ptr<SampleCollection>> chains(numChains);
  chains.at(0) = samps;

  for(int i=1; i<numChains; ++i){
    std::cout << "Running chain " << i << "..." << std::flush;
    Eigen::VectorXd x0 = startPt + 1.5*RandomGenerator::GetNormal(mu.size()); // Start the Gaussian block at the mean

    mcmc = MCMCFactory::CreateSingleChain(pt, problem);
    chains.at(i) = mcmc->Run(x0);

    std::cout << " done" << std::endl;
  }

  Eigen::VectorXd rhat = Diagnostics::Rhat(chains);
  std::cout << "\nRhat = " << rhat.transpose() << std::endl;


  return 0;
}
