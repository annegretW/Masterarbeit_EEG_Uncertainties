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
#include "MUQ/SamplingAlgorithms/ParallelFixedSamplesMIMCMC.h"

#include <boost/property_tree/ptree.hpp>

namespace pt = boost::property_tree;
using namespace muq::Modeling;
using namespace muq::SamplingAlgorithms;
using namespace muq::Utilities;

#include "ParallelProblem.h"

int main(int argc, char **argv){

  MPI_Init(&argc, &argv);

  pt::ptree pt;

  pt.put("NumSamples_0", 1e4);
  pt.put("NumSamples_1", 5e3);
  pt.put("NumSamples_2", 1e3);
  pt.put("NumSamples_3", 5e2);
  pt.put("MLMCMC.Subsampling", 1);
  pt.put("MCMC.BurnIn", 10); // number of samples for single level
  pt.put("verbosity", 1); // show some output

/*{
  std::cout << std::endl << "*************** multilevel" << std::endl << std::endl;
  auto componentFactory = std::make_shared<MyMIComponentFactory>(pt);

  MIMCMC mimcmc(pt, componentFactory);
  mimcmc.Run();
  std::cout << "mean QOI: " << mimcmc.MeanQOI().transpose() << std::endl;

  auto index_zero = std::make_shared<MultiIndex>(1);
  index_zero->SetValue(0, 0);
  std::cout << "coarsest level mean QOI: " << mimcmc.GetBox(index_zero)->FinestChain()->GetQOIs()->Mean().transpose() << std::endl;
}


{
  std::cout << std::endl << "*************** single level reference" << std::endl << std::endl;
  auto index = componentFactory->FinestIndex();

  auto problem = componentFactory->SamplingProblem(index);
  auto proposal = componentFactory->Proposal(index, problem);

  std::vector<std::shared_ptr<TransitionKernel>> kernels(1);
  kernels[0] = std::make_shared<DummyKernel>(pt,problem,proposal);

  Eigen::VectorXd startingPoint = componentFactory->StartingPoint(index);

  auto mcmc = std::make_shared<SingleChainMCMC>(pt,kernels);
  mcmc->Run(startingPoint);
  std::cout << "mean QOI: " << mcmc->GetQOIs()->Mean().transpose() << std::endl;
}*/

{
  auto comm = std::make_shared<parcer::Communicator>();

  auto componentFactory = std::make_shared<MyMIComponentFactory>(pt);
  StaticLoadBalancingMIMCMC parallelMIMCMC (pt, componentFactory);

  if (comm->GetRank() == 0) {
    parallelMIMCMC.Run();
    Eigen::VectorXd meanQOI = parallelMIMCMC.MeanQOI();
    std::cout << "mean QOI: " << meanQOI.transpose() << std::endl;
    parallelMIMCMC.WriteToFile("samples.h5");
  }
  parallelMIMCMC.Finalize();

}

  MPI_Finalize();

  return 0;
}
