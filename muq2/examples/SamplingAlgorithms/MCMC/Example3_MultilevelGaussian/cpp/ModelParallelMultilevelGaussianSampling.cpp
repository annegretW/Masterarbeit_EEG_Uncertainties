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

#include "MUQ/SamplingAlgorithms/ParallelMIComponentFactory.h"

#include <boost/property_tree/ptree.hpp>

namespace pt = boost::property_tree;
using namespace muq::Modeling;
using namespace muq::SamplingAlgorithms;
using namespace muq::Utilities;

#include "ParallelProblem.h"


int main(int argc, char **argv){

  MPI_Init(&argc, &argv);

  pt::ptree pt;

  pt.put("NumSamples", 1e3); // number of samples for single level
  pt.put("NumInitialSamples", 1e3); // number of initial samples for greedy MLMCMC
  pt.put("GreedyTargetVariance", 0.05); // estimator variance to be achieved by greedy algorithm
  pt.put("verbosity", 1); // show some output
  pt.put("MLMCMC.Subsampling_0", 8);
  pt.put("MLMCMC.Subsampling_1", 4);
  pt.put("MLMCMC.Subsampling_2", 2);
  pt.put("MLMCMC.Subsampling_3", 0);

  auto localFactory = std::make_shared<MyMIComponentFactory>(pt);

  std::cout << std::endl << "*************** greedy multillevel chain" << std::endl << std::endl;

  auto comm = std::make_shared<parcer::Communicator>();
  localFactory->SetComm(comm);
  auto componentFactory = std::make_shared<ParallelMIComponentFactory>(comm, comm, localFactory);

  if (comm->GetRank() == 0) {
    GreedyMLMCMC greedymlmcmc (pt, componentFactory);
    greedymlmcmc.Run();
    std::cout << "mean QOI: " << greedymlmcmc.GetQOIs()->Mean().transpose() << std::endl;
  }


  if (comm->GetRank() == 0) {
    std::cout << std::endl << "*************** single chain reference" << std::endl << std::endl;

    SLMCMC slmcmc (pt, componentFactory);
    slmcmc.Run();
    componentFactory->finalize();
    std::cout << "mean QOI: " << slmcmc.GetQOIs()->Mean().transpose() << std::endl;
  }


  MPI_Finalize();
  return 0;
}
