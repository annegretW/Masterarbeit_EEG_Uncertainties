#include "MUQ/SamplingAlgorithms/SLMCMC.h"
#include "MUQ/SamplingAlgorithms/GreedyMLMCMC.h"
#include "MUQ/SamplingAlgorithms/MIMCMC.h"

#include "MUQ/Modeling/Distributions/Gaussian.h"
#include "MUQ/Modeling/Distributions/Density.h"

#include "MUQ/SamplingAlgorithms/DummyKernel.h"
#include "MUQ/SamplingAlgorithms/MHKernel.h"
#include "MUQ/SamplingAlgorithms/MHProposal.h"
#include "MUQ/SamplingAlgorithms/CrankNicolsonProposal.h"
#include "MUQ/SamplingAlgorithms/SamplingProblem.h"
#include "MUQ/SamplingAlgorithms/SubsamplingMIProposal.h"

#include "MUQ/SamplingAlgorithms/MIComponentFactory.h"

#include <boost/property_tree/ptree.hpp>

namespace pt = boost::property_tree;
using namespace muq::Modeling;
using namespace muq::SamplingAlgorithms;
using namespace muq::Utilities;

#include "Problem.h"

int main(){

  pt::ptree pt;

  pt.put("NumSamples", 1e6); // number of samples for single level
  pt.put("verbosity", 1); // show some output

  auto componentFactory = std::make_shared<MyMIComponentFactory>(pt);


  std::cout << std::endl << "*************** multilevel" << std::endl << std::endl;

  MIMCMC mimcmc(pt, componentFactory);
  //GreedyMLMCMC greedymlmcmc (pt, componentFactory);
  mimcmc.Run();
  std::cout << "mean QOI: " << mimcmc.GetQOIs()->Mean().transpose() << std::endl;

  //mimcmc.GetBox(std::make_shared<MultiIndex>(1,0));
  //std::cout << "coarsest: " << mimcmc.GetBox(std::make_shared<MultiIndex>(1,0))->FinestChain()->GetQOIs()->Mean().transpose() << std::endl;
  auto index_zero = std::make_shared<MultiIndex>(1);
  index_zero->SetValue(0, 0);
  std::cout << "coarsest: " << mimcmc.GetBox(index_zero)->FinestChain()->GetQOIs()->Mean().transpose() << std::endl;


  std::cout << std::endl << "*************** single level" << std::endl << std::endl;

{
  auto index = componentFactory->FinestIndex();

  auto problem = componentFactory->SamplingProblem(index);
  auto proposal = componentFactory->Proposal(index, problem);

  std::vector<std::shared_ptr<TransitionKernel>> kernels(1);
  kernels[0] = std::make_shared<DummyKernel>(pt,problem,proposal);

  Eigen::VectorXd startingPoint = componentFactory->StartingPoint(index);

  auto mcmc = std::make_shared<SingleChainMCMC>(pt,kernels);
  mcmc->Run(startingPoint);
  std::cout << "mean QOI: " << mcmc->GetQOIs()->Mean().transpose() << std::endl;
}

  //SLMCMC slmcmc (pt, componentFactory);
  //slmcmc.Run();
  //std::cout << "mean QOI: " << slmcmc.MeanQOI().transpose() << std::endl;

  return 0;
}
