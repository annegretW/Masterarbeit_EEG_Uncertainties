/***

### Overview
This example demonstrates how to manually specify the proposal covariance in a
simple random walk proposal.

### MUQ MCMC Interfaces
There are two ways to specify MCMC samplers in MUQ.  The first, which is
demonstrated in the Example1_Gaussian example, specifies the proposal variance
and other algorithmic parameters through a boost property tree.  A lower level
interface also exists, which allows users to manually specify the proposal
distribution and transition kernel.  In this latter setting, the boost property
tree is then used only for chain-level settings like the number of steps,
burn in, and print levels.


*/
#include "MUQ/Modeling/Distributions/Gaussian.h"
#include "MUQ/Modeling/Distributions/Density.h"

#include "MUQ/SamplingAlgorithms/SamplingProblem.h"
#include "MUQ/SamplingAlgorithms/SingleChainMCMC.h"
#include "MUQ/SamplingAlgorithms/MHProposal.h"
#include "MUQ/SamplingAlgorithms/MHKernel.h"
#include "MUQ/SamplingAlgorithms/Diagnostics.h"

#include "MUQ/Utilities/RandomGenerator.h"

#include <boost/property_tree/ptree.hpp>

namespace pt = boost::property_tree;
using namespace muq::Modeling;
using namespace muq::SamplingAlgorithms;
using namespace muq::Utilities;


int main(){

  /***
  ### 1. Define the target density and set up the sampling problem

  The `Gaussian` class in MUQ provides a definition of a multivariate Gaussian
  distribution.  The distribution is completely defined by a mean vector and a
  covariance (or precision) matrix.   In MUQ, modeling components like a Gaussian
  probability density, are represented as children of the `ModPiece` base class.
  However, interpreting the Gaussian as as a `ModPiece` is ambiguous.  The
  Gaussian class could be used to define represent a random variable "ModPiece"
  that returns a random realization of the multivariate Gaussian when evaluated.
  Or the Gaussian class could define a "ModPiece" that evaluates the log
  probability density function of the Gaussian distribution.   To select one of
  these two use cases, the MUQ `Gaussian` class has member functions `AsDensity()`
  and `AsVariable()` that return a shared_ptr to a class that evaluates the log PDF,
  or a class that draws a random sample, respectively.   These functions are
  implemented in the `Distribution` class, which is a parent of the `Gaussian`
  class.

  The AbstractSamplingProblem base class and its
  children, like the SamplingProblem class, define the interface between
  sampling algorithms like MCMC and the models and densities they work with.

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
  ### 3. Construct the RWM proposal
  Let $x_k$ denote the $k^{th}$ state in a Markov chain.   At step $k$,
  the Random Walk Metropolis algorithm draws a random realization of the proposal
  random variable $x^\prime = x_k + z$, where $z\sim q(z)$ is a random
  variable distributed according to the proposal distribution $q(z)$.  The
  proposed point is then accepted or rejected using the Metropolis-Hastings rule.

  The MUQ RWM implementation allows users to specify the proposal distributions
  $q(z)$ either through options in a boost property_tree (see the Example1_Gaussian
  example) or manually (demonstrated below).   Here, we employ the latter approach
  and manually specify an instance of MUQ's `MHProposal` class, which is then
  combined with the `MHKernel` Markov transition kernel to define the RWM algorithm.
  */

  /***
  Define the proposal distribution.
  */
  Eigen::VectorXd propMu = Eigen::VectorXd::Zero(2);

  // Set the proposal covariance to be the optimal scaling of the target covariance
  Eigen::MatrixXd propCov = (2.4*2.4/std::sqrt(2))*cov;

  auto propDist = std::make_shared<Gaussian>(propMu, propCov);

  /***
  Use the Gaussian proposal distribution to define an MCMC proposal class.
  */
  pt::ptree propOpts;
  auto proposal = std::make_shared<MHProposal>(propOpts, problem, propDist);

  /***
  Construct the Metropolis-Hastings (MH) Markov transition kernel using the
  proposal.

  MUQ can perform blockwise updates of target densities that have multiple
  vector-valued inputs (e.g., parameters and hyperparameters).  The
  `SingleChainMCMC` class employed below therefore expects a transition kernel
  for each parameter block.  These kernels are passed to the `SingleChainMCMC`
  constructor as a `std::vector` of transition kernels.  Here, we only have a
  single kernel but we store the kernel in a vector to match the interface
  expected by `SingleChainMCMC`.
  */
  pt::ptree kernOpts;
  std::vector<std::shared_ptr<TransitionKernel>> kernels(1);
  kernels.at(0) = std::make_shared<MHKernel>(kernOpts, problem, proposal);

  /***
  Use the kernel to define a single chain MCMC algorithm.
  */
  pt::ptree chainOpts;
  chainOpts.put("NumSamples", 1e4); // number of MCMC steps
  chainOpts.put("BurnIn", 1e3);
  chainOpts.put("PrintLevel",3);

  auto mcmc = std::make_shared<SingleChainMCMC>(chainOpts,kernels);

  /***
  ### 3. Run the MCMC algorithm
  We are now ready to run the MCMC algorithm.  Here we start the chain at the
  target densities mean.   The resulting samples are returned in an instance
  of the SampleCollection class, which internally holds the steps in the MCMC chain
  as a vector of SamplingStates.
  */
  Eigen::VectorXd startPt = mu;
  std::shared_ptr<SampleCollection> samps = mcmc->Run(startPt);

  /***
  ### 4. Run multiple chains to assess convergence
  
  */
  unsigned int numChains = 4;
  std::vector<std::shared_ptr<SampleCollection>> chains(numChains);
  chains.at(0) = samps;

  for(unsigned int i=1; i<numChains; ++i){

    // Construct a new sampler.
    mcmc = std::make_shared<SingleChainMCMC>(chainOpts,kernels);    
    
    // Generate a "diffuse" starting point for the chain
    startPt = std::sqrt(1.5) * RandomGenerator::GetNormal(2);

    // Run the sampler
    chains.at(i) = mcmc->Run(startPt);
  }

  /***
  The $\hat{R}$ statistic compares the intra- and inter-chain variances to assess 
  whether the chains have converged to the same distribution (the target distribution).
  Here we use the `Diagnostics` class in MUQ to compute $\hat{R}$ following the split and
  rank-transformed techniques described in [[Vehtari et al., 2021](https://arxiv.org/abs/1903.08008)].
  If $\hat{R}$ is close to $1$ (e.g., $\hat{R}<1.01$), then the chains are past burn in and 
  are likely generating samples of the posterior.
  */

  Eigen::VectorXd rhat = Diagnostics::Rhat(chains);
  std::cout << "\nRhat:\n" << rhat.transpose() << std::endl;


  /***
  ### 5. Analyze the results

  The SampleCollection class provides several functions for computing sample moments.
  For example, here we compute the mean, variance, and third central moment.
  While the third moment is actually a tensor, here we only return the marginal
  values, i.e., $\mathbb{E}_x[(x_i-\mu_i)^3]$ for each $i$.
  */
  Eigen::VectorXd sampMean = samps->Mean();
  std::cout << "\nSample Mean:\n" << sampMean.transpose() << std::endl;

  Eigen::VectorXd sampVar = samps->Variance();
  std::cout << "\nSample Variance:\n" << sampVar.transpose() << std::endl;

  Eigen::MatrixXd sampCov = samps->Covariance();
  std::cout << "\nSample Covariance:\n" << sampCov << std::endl;

  Eigen::VectorXd sampMom3 = samps->CentralMoment(3);
  std::cout << "\nSample Third Moment:\n" << sampMom3.transpose() << std::endl << std::endl;

  /***
  ### 6. Inspect the sample meta data.

  In addition to storing the state of the MCMC chain, MUQ may also store extra
  information stored by the transition kernel or proposal.  This "metadata"
  is store in the `SampleCollection` and can be accessed using the `GetMeta`
  function of the `SampleCollection` class.   It is also possible to list what
  metadata is available using the `ListMeta` function.`   Below, we first list
  all of the available metadata and then extract the log of the target density.

  Note that the `GetMeta` function returns a matrix of doubles regardless of the
  size or type of the metadata.   Each column of the matrix corresponds to a sample
  and each row of the matrix to a component of the (possibly) vector-valued
  metadata variable.  For scalar values, like the log density, there will only
  be a single row.

  */

  // List the name of any metadata stored by the samples
  std::cout << "Available MetaData: " << std::endl;
  for(auto& metaKey : samps->ListMeta())
    std::cout << "  \"" << metaKey << "\"" << std::endl;
  std::cout << std::endl;

  // Extract the log-density of the target distribution at each sample
  Eigen::MatrixXd logTargetDens = samps->GetMeta("LogTarget");

  // Compute the maximum log target density and store the sample index where it occured
  double maxLogDens;
  unsigned int maxRow, maxCol;
  maxLogDens = logTargetDens.maxCoeff(&maxRow, &maxCol);

  std::cout << "From MetaData:" << std::endl;
  std::cout << "  p* = max log(p(x)) = " << maxLogDens << std::endl;
  std::cout << "  x* = argmax p(x) = " << samps->at(maxCol)->state.at(0).transpose() << std::endl;

  /***
  ### 7. Extract samples as a matrix for further processing.

  In some cases you will want to extract a matrix of the MCMC states from the
  `SampleCollection` object.   As shown below, that can be easily accomplished
  using the `AsMatrix` function in the `SampleCollection` class.
  */
  Eigen::MatrixXd sampMat = samps->AsMatrix();

  std::cout << "\nMean using eigen = " << sampMat.rowwise().mean().transpose() << std::endl;

  return 0;
}
