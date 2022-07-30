/***
## Overview
This example shows how to use the low-level API to MUQ's Multilevel MCMC algorithms.  The actual sampling 
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

#include "MUQ/Utilities/RandomGenerator.h"

#include <boost/property_tree/ptree.hpp>
#include <sstream>

namespace pt = boost::property_tree;
using namespace muq::Modeling;
using namespace muq::SamplingAlgorithms;
using namespace muq::Utilities;



class MySamplingProblem : public AbstractSamplingProblem {
public:
  MySamplingProblem(std::shared_ptr<muq::Modeling::ModPiece> targetIn)
   : AbstractSamplingProblem(Eigen::VectorXi::Constant(1,2), Eigen::VectorXi::Constant(1,2)),
     target(targetIn){}

  virtual ~MySamplingProblem() = default;


  virtual double LogDensity(std::shared_ptr<SamplingState> const& state) override {
    lastState = state;
    return target->Evaluate(state->state).at(0)(0);
  };

  virtual std::shared_ptr<SamplingState> QOI() override {
    assert (lastState != nullptr);
    return std::make_shared<SamplingState>(lastState->state, 1.0);
  }

private:
  std::shared_ptr<SamplingState> lastState = nullptr;

  std::shared_ptr<muq::Modeling::ModPiece> target;

};


class MyInterpolation : public MIInterpolation {
public:
  std::shared_ptr<SamplingState> Interpolate (std::shared_ptr<SamplingState> const& coarseProposal, 
                                              std::shared_ptr<SamplingState> const& fineProposal) {
    return std::make_shared<SamplingState>(coarseProposal->state);
  }
};


class MyMIComponentFactory : public MIComponentFactory {
public:
  MyMIComponentFactory (Eigen::VectorXd const& start, pt::ptree pt)
   : startingPoint(start), pt(pt)
  { }

  virtual std::shared_ptr<MCMCProposal> Proposal (std::shared_ptr<MultiIndex> const& index, std::shared_ptr<AbstractSamplingProblem> const& samplingProblem) override {
    pt::ptree pt;
    pt.put("BlockIndex",0);

    Eigen::VectorXd mu(2);
    mu << 1.0, 2.0;
    Eigen::MatrixXd cov(2,2);
    cov << 0.7, 0.6,
    0.6, 1.0;
    cov *= 20.0;

    auto prior = std::make_shared<Gaussian>(mu, cov);

    return std::make_shared<CrankNicolsonProposal>(pt, samplingProblem, prior);
  }

  virtual std::shared_ptr<MultiIndex> FinestIndex() override {
    auto index = std::make_shared<MultiIndex>(1);
    index->SetValue(0, 3);
    return index;
  }

  virtual std::shared_ptr<MCMCProposal> CoarseProposal (std::shared_ptr<MultiIndex> const& fineIndex,
                                                        std::shared_ptr<MultiIndex> const& coarseIndex,
                                                        std::shared_ptr<AbstractSamplingProblem> const& coarseProblem,
                                                           std::shared_ptr<SingleChainMCMC> const& coarseChain) override {
    pt::ptree ptProposal = pt;
    ptProposal.put("BlockIndex",0);
    return std::make_shared<SubsamplingMIProposal>(ptProposal, coarseProblem, coarseIndex, coarseChain);
  }

  virtual std::shared_ptr<AbstractSamplingProblem> SamplingProblem (std::shared_ptr<MultiIndex> const& index) override {
    Eigen::VectorXd mu(2);
    mu << 1.0, 2.0;
    Eigen::MatrixXd cov(2,2);
    cov << 0.7, 0.6,
           0.6, 1.0;

    if (index->GetValue(0) == 0) {
      mu *= 0.8;
      cov *= 2.0;
    } else if (index->GetValue(0) == 1) {
      mu *= 0.9;
      cov *= 1.5;
    } else if (index->GetValue(0) == 2) {
      mu *= 0.99;
      cov *= 1.1;
    } else if (index->GetValue(0) == 3) {
      mu *= 1.0;
      cov *= 1.0;
    } else {
      std::cerr << "Sampling problem not defined!" << std::endl;
      assert (false);
    }

    auto coarseTargetDensity = std::make_shared<Gaussian>(mu, cov)->AsDensity();
    return std::make_shared<MySamplingProblem>(coarseTargetDensity);
  }

  virtual std::shared_ptr<MIInterpolation> Interpolation (std::shared_ptr<MultiIndex> const& index) override {
    return std::make_shared<MyInterpolation>();
  }

  virtual Eigen::VectorXd StartingPoint (std::shared_ptr<MultiIndex> const& index) override {
    return startingPoint;
  }

private:
  Eigen::VectorXd startingPoint;
  pt::ptree pt;
};




int main(){

  pt::ptree pt;

  pt.put("NumSamples", 1e2); // number of samples for single level
  pt.put("NumInitialSamples", 1e3); // number of initial samples for greedy MLMCMC
  pt.put("GreedyTargetVariance", 0.05); // estimator variance to be achieved by greedy algorithm
  pt.put("verbosity", 1); // show some output
  pt.put("MLMCMC.Subsampling_0", 8);
  pt.put("MLMCMC.Subsampling_1", 4);
  pt.put("MLMCMC.Subsampling_2", 2);
  pt.put("MLMCMC.Subsampling_3", 0);


  unsigned int numChains = 5;
  std::vector<std::shared_ptr<MultiIndexEstimator>> estimators(numChains);

  for(int chainInd=0; chainInd<numChains; ++chainInd){
    Eigen::VectorXd x0 = RandomGenerator::GetNormal(2);
    auto componentFactory = std::make_shared<MyMIComponentFactory>(x0, pt);

    std::cout << "\n=============================\n";
    std::cout << "Running MLMCMC Chain " << chainInd << ": \n";
    std::cout << "-----------------------------\n";

    GreedyMLMCMC greedymlmcmc (pt, componentFactory);
    estimators.at(chainInd) = greedymlmcmc.Run();

    std::cout << "mean QOI: " << estimators.at(chainInd)->Mean().transpose() << std::endl;

    std::stringstream filename;
    filename << "MultilevelGaussianSampling_Chain" << chainInd << ".h5";
    greedymlmcmc.WriteToFile(filename.str());
  }
  
  std::cout << "\n=============================\n";
  std::cout << "Multilevel Summary: \n";
  std::cout << "-----------------------------\n";
  std::cout << "  Rhat:               " << Diagnostics::Rhat(estimators).transpose() << std::endl;
  std::cout << "  Mean (chain 0):     " << estimators.at(0)->Mean().transpose() << std::endl;
  std::cout << "  MCSE (chain 0):     " << estimators.at(0)->StandardError().transpose() << std::endl;
  std::cout << "  ESS (chain 0):      " << estimators.at(0)->ESS().transpose() << std::endl;
  std::cout << "  Variance (chain 0): " << estimators.at(0)->Variance().transpose() << std::endl;
  std::cout << std::endl;


  return 0;
}
