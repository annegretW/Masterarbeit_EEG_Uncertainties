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

#include "MUQ/SamplingAlgorithms/MIComponentFactory.h"

#include <boost/property_tree/ptree.hpp>

namespace pt = boost::property_tree;
using namespace muq::Modeling;
using namespace muq::SamplingAlgorithms;
using namespace muq::Utilities;

#include "ParallelProblem.h"


int main(){

  pt::ptree pt;

  pt.put("NumSamples", 1e3); // number of samples for single level
  pt.put("NumInitialSamples", 1e4); // number of initial samples for greedy MLMCMC
  pt.put("GreedyTargetVariance", 0.05); // estimator variance to be achieved by greedy algorithm
  pt.put("verbosity", 1); // show some output


  for (int subsampling : {0, 5, 10, 25, 100, 1000}) {
    std::cout << "Running with subsampling " << subsampling << std::endl;
    pt.put("MLMCMC.Subsampling", subsampling);


    auto componentFactory = std::make_shared<MyMIComponentFactory>(pt);

    std::cout << std::endl << "*************** greedy multillevel chain" << std::endl << std::endl;

    GreedyMLMCMC greedymlmcmc (pt, componentFactory);
    greedymlmcmc.Run();
    std::cout << "mean QOI: " << greedymlmcmc.GetQOIs()->Mean().transpose() << std::endl;
    greedymlmcmc.Draw(false);

    /*std::cout << std::endl << "*************** single chain reference" << std::endl << std::endl;

    SLMCMC slmcmc (pt, componentFactory);
    slmcmc.Run();
    std::cout << "mean QOI: " << slmcmc.MeanQOI().transpose() << std::endl;*/

  }

  return 0;
}
