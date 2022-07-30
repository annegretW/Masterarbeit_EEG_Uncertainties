/***
## Overview
This example shows how to use the high-level (basic) API to MUQ's Multilevel MCMC algorithms.  The actual sampling
problem is quite simple: we want to draw samples from a multivariate Gaussian distribution with
mean
\f[
  \mu = \left[ \begin{array}{c} 1\\ 2\end{array}\right]
\f]
and covariance
\f[
\Sigma = \left[\begin{array}{cc} 0.7& 0.6\\ 0.6 & 1.0\end{array}\right].
\f]
It's of course possible to sample this distribution directly, but we will use Multilevel MCMC methods in
this example to illustrate their use without introducing unnecessary complexity to the problem definition.

*/

#include "MUQ/SamplingAlgorithms/SLMCMC.h"
#include "MUQ/SamplingAlgorithms/GreedyMLMCMC.h"
#include "MUQ/SamplingAlgorithms/MIMCMC.h"

#include "MUQ/Modeling/Distributions/Gaussian.h"
#include "MUQ/Modeling/Distributions/Density.h"

#include "MUQ/SamplingAlgorithms/MHKernel.h"
#include "MUQ/SamplingAlgorithms/MHProposal.h"
#include "MUQ/SamplingAlgorithms/CrankNicolsonProposal.h"
#include "MUQ/SamplingAlgorithms/SamplingProblem.h"
#include "MUQ/SamplingAlgorithms/SubsamplingMIProposal.h"

#include "MUQ/SamplingAlgorithms/Diagnostics.h"

#include "MUQ/SamplingAlgorithms/MIComponentFactory.h"
#include "MUQ/SamplingAlgorithms/SingleChainMCMC.h"

#include "MUQ/Utilities/RandomGenerator.h"

#include <boost/property_tree/ptree.hpp>
#include <sstream>

namespace pt = boost::property_tree;
using namespace muq::Modeling;
using namespace muq::SamplingAlgorithms;
using namespace muq::Utilities;


/***
  ## Define the Target Distributions
  To apply MLMCMC, we need to define
*/
std::vector<std::shared_ptr<ModPiece>> ConstructDensities()
{
  unsigned int numLevels = 4;
  std::vector<std::shared_ptr<ModPiece>> logDensities(numLevels);

  Eigen::VectorXd tgtMu(2);
  Eigen::MatrixXd tgtCov(2,2);
  tgtMu  << 1.0, 2.0;
  tgtCov << 0.7, 0.2,
            0.2, 1.0;

  // Define the problem at the coarsest level
  Eigen::VectorXd levelMu  = 0.8*tgtMu;
  Eigen::MatrixXd levelCov = 2.0*tgtCov;
  logDensities.at(0) = std::make_shared<Gaussian>(levelMu, levelCov)->AsDensity();

  // Define the second coarsest level
  levelMu  = 0.9*tgtMu;
  levelCov = 1.5*tgtCov;
  logDensities.at(1) = std::make_shared<Gaussian>(levelMu, levelCov)->AsDensity();

  // Define the second finest level
  levelMu = 0.99*tgtMu;
  levelCov = 1.1*tgtCov;
  logDensities.at(2) = std::make_shared<Gaussian>(levelMu, levelCov)->AsDensity();

  // Deifne the finest level.  This should be the target distribution.
  logDensities.at(3) = std::make_shared<Gaussian>(tgtMu, tgtCov)->AsDensity();

  return logDensities;
}

int main(){

  std::vector<std::shared_ptr<ModPiece>> logDensities = ConstructDensities();


  pt::ptree options;

  options.put("NumInitialSamples", 1000); // number of initial samples for greedy MLMCMC
  options.put("GreedyTargetVariance", 0.05); // Target estimator variance to be achieved by greedy algorithm
  options.put("verbosity", 1); // show some output
  options.put("MLMCMC.Subsampling_0", 8);
  options.put("MLMCMC.Subsampling_1", 4);
  options.put("MLMCMC.Subsampling_2", 2);
  options.put("MLMCMC.Subsampling_3", 0);

  options.put("Proposal.Method", "MHProposal");
  options.put("Proposal.ProposalVariance", 16.0);


  unsigned int numChains = 5;
  std::vector<std::shared_ptr<MultiIndexEstimator>> estimators(numChains);

  for(int chainInd=0; chainInd<numChains; ++chainInd){
    Eigen::VectorXd x0 = RandomGenerator::GetNormal(2);

    std::cout << "\n=============================\n";
    std::cout << "Running MLMCMC Chain " << chainInd << ": \n";
    std::cout << "-----------------------------\n";

    GreedyMLMCMC sampler(options, x0, logDensities);
    estimators.at(chainInd) = sampler.Run();

    std::cout << "Chain " << chainInd << " Mean:     " << estimators.at(chainInd)->Mean().transpose() << std::endl;
    std::cout << "Chain " << chainInd << " Variance: " << estimators.at(chainInd)->Variance().transpose() << std::endl;

    std::stringstream filename;
    filename << "MultilevelGaussianSampling_Chain" << chainInd << ".h5";
    sampler.WriteToFile(filename.str());

  }


  unsigned int numCalls = logDensities.back()->GetNumCalls();

  std::cout << "\n=============================\n";
  std::cout << "Multilevel Summary: \n";
  std::cout << "-----------------------------\n";
  std::cout << "  Rhat:               " << Diagnostics::Rhat(estimators).transpose() << std::endl;
  std::cout << "  Mean (chain 0):     " << estimators.at(0)->Mean().transpose() << std::endl;
  std::cout << "  MCSE (chain 0):     " << estimators.at(0)->StandardError().transpose() << std::endl;
  std::cout << "  ESS (chain 0):      " << estimators.at(0)->ESS().transpose() << std::endl;
  std::cout << "  Variance (chain 0): " << estimators.at(0)->Variance().transpose() << std::endl;
  std::cout << "  Finest evals:       " << logDensities.back()->GetNumCalls() << std::endl;
  std::cout << std::endl;



  std::cout << "\n=============================\n";
  std::cout << "Running Single Level Chain" << ": \n";
  std::cout << "-----------------------------\n";

  // For comparison, run a single chain on this problem with the same number of density evaluations
  auto problem = std::make_shared<SamplingProblem>(logDensities.back());

  // parameters for the sampler
  pt::ptree pt;
  pt.put("NumSamples", numCalls); // number of MCMC steps
  pt.put("BurnIn", 0);
  pt.put("PrintLevel",3);
  pt.put("KernelList", "Kernel1"); // Name of block that defines the transition kernel
  pt.put("Kernel1.Method","MHKernel");  // Name of the transition kernel class
  pt.put("Kernel1.Proposal", "MyProposal"); // Name of block defining the proposal distribution
  pt.put("Kernel1.MyProposal.Method", "MHProposal"); // Name of proposal class
  pt.put("Kernel1.MyProposal.ProposalVariance", 4.0); // Variance of the isotropic MH proposal

  Eigen::VectorXd startPt(2);
  startPt << 1.0, 2.0;
  auto mcmc = std::make_shared<SingleChainMCMC>(pt, problem);

  auto samps = mcmc->Run(startPt);

  std::cout << "\n=============================\n";
  std::cout << "Single Level Summary: \n";
  std::cout << "-----------------------------\n";
  std::cout << "  Mean:               " << samps->Mean().transpose() << std::endl;
  std::cout << "  MCSE:               " << samps->StandardError().transpose() << std::endl;
  std::cout << "  ESS:                " << samps->ESS().transpose() << std::endl;
  std::cout << "  Variance:           " << samps->Variance().transpose() << std::endl;
  std::cout << "  Finest evals:       " << logDensities.back()->GetNumCalls()- numCalls << std::endl;
  std::cout << std::endl;

  return 0;
}
