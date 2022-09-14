#include "MUQ/SamplingAlgorithms/MLDAKernel.h"

#include "MUQ/Modeling/Distributions/Gaussian.h"
#include "MUQ/Modeling/Distributions/Density.h"
#include "MUQ/Modeling/UMBridge/UMBridgeModPiece.h"

#include "MUQ/SamplingAlgorithms/MHProposal.h"
#include "MUQ/SamplingAlgorithms/MHKernel.h"
#include "MUQ/SamplingAlgorithms/SamplingProblem.h"
#include "MUQ/SamplingAlgorithms/SingleChainMCMC.h"
#include "MUQ/SamplingAlgorithms/MCMCFactory.h"
#include "MUQ/SamplingAlgorithms/Diagnostics.h"

#include "MUQ/Utilities/AnyHelpers.h"
#include "MUQ/Utilities/RandomGenerator.h"

#include <boost/property_tree/ptree.hpp>

namespace pt = boost::property_tree;
using namespace muq::Modeling;
using namespace muq::SamplingAlgorithms;
using namespace muq::Utilities;

void MLDA(std::vector<std::shared_ptr<SamplingProblem>> sampling_problems, int n, Eigen::VectorXd startPt, int num_samples, int burn_in, std::vector<double> proposal_var, std::string results_path){
  /*{ // MLDA
    pt::ptree ptProposal;
    ptProposal.put("Subsampling_0", 10); // Subsampling on level 0
    ptProposal.put("Subsampling_1", 5); // Subsampling on level 1

    ptProposal.put("Proposal_Variance_0", proposal_var[0]); // Proposal Variance on coarsest level
    ptProposal.put("Proposal_Variance_1", proposal_var[1]);
    ptProposal.put("Proposal_Variance_2", proposal_var[2]);
    auto proposal = std::make_shared<MLDAProposal>(ptProposal, sampling_problems.size()-1, sampling_problems);

    pt::ptree ptBlockID;
    ptBlockID.put("BlockIndex",0);
    std::vector<std::shared_ptr<TransitionKernel>> kernel(1);
    // TODO: MLDA kernel here
    kernel[0] = std::make_shared<MLDAKernel>(ptBlockID,sampling_problems.back(),proposal);

    pt::ptree pt;
    pt.put("NumSamples", num_samples); // number of MCMC steps
    pt.put("BurnIn", burn_in);
    pt.put("PrintLevel",3);
    auto chain = std::make_shared<SingleChainMCMC>(pt,kernel);

    std::shared_ptr<SampleCollection> samps = chain->Run(startPt);

    samps->WriteToFile(results_path + "_mlda.h5");
  }*/

  { // Single level MCMC Reference
    for (int level = 0; level < sampling_problems.size(); level++) {
      auto problem = sampling_problems[level];

      pt::ptree ptProposal;
      ptProposal.put("ProposalVariance",proposal_var[level]);
      auto proposal = std::make_shared<MHProposal>(ptProposal, problem);

      int x = problem->blockSizes(0);
      std::cout << x << std::endl;

      pt::ptree ptBlockID;
      ptBlockID.put("BlockIndex",0);
      std::vector<std::shared_ptr<TransitionKernel>> kernel(1);
      kernel[0] = std::make_shared<MHKernel>(ptBlockID,problem,proposal);
      pt::ptree pt;
      pt.put("NumSamples", num_samples); // number of MCMC steps
      pt.put("BurnIn", burn_in);
      pt.put("PrintLevel",3);
      auto chain = std::make_shared<SingleChainMCMC>(pt,kernel);

      std::shared_ptr<SampleCollection> samps = chain->Run(startPt);

      samps->WriteToFile(results_path + "_l" + std::to_string(level) + ".h5");
    }
  }
}

void example1(){
  int n = 3;
  int num_samples = 1e4;
  int burn_in = 0;

  std::vector<double> proposal_var = {5,5,5};

  std::string results_path = "/home/anne/Masterarbeit/masterarbeit/results/samples2";

  std::vector<std::shared_ptr<SamplingProblem>> sampling_problems;
  {
    json config;
    config["level"] = 1;
    sampling_problems.push_back(std::make_shared<SamplingProblem>(std::make_shared<UMBridgeModPiece>("localhost:4243", config)));
  }
  {
    json config;
    config["level"] = 2;
    sampling_problems.push_back(std::make_shared<SamplingProblem>(std::make_shared<UMBridgeModPiece>("localhost:4243", config)));
  }
  {
    json config;
    config["level"] = 3;
    sampling_problems.push_back(std::make_shared<SamplingProblem>(std::make_shared<UMBridgeModPiece>("localhost:4243", config)));
  }


  Eigen::VectorXd startPt(3);
  startPt << 127, 127, 127;
  MLDA(sampling_problems, n, startPt, num_samples, burn_in, proposal_var, results_path);
}


int main(){
  example1();
  return 0;
}
