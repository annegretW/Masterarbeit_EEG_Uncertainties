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

void MLDA(std::vector<std::shared_ptr<SamplingProblem>> sampling_problems, int n, Eigen::VectorXd startPt, int num_samples, int burn_in, double proposal_var, double proposal_var_l0, std::string results_path){
  { // MLDA
    pt::ptree ptProposal;
    ptProposal.put("Subsampling_0", 10); // Subsampling on level 0
    ptProposal.put("Subsampling_1", 3); // Subsampling on level 1
    ptProposal.put("Proposal_Variance_0", proposal_var_l0); // Proposal Variance on coarsest level
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
  }

  { // Single level MCMC Reference
    for (int level = 0; level < sampling_problems.size(); level++) {
      auto problem = sampling_problems[level];

      pt::ptree ptProposal;
      ptProposal.put("ProposalVariance",proposal_var);
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
  int n = 10;
  int num_samples = 1000;
  int burn_in = 0;
  double proposal_var = 0.05;
  double proposal_var_l0 = 0.1;
  std::string results_path = "/home/anne/Masterarbeit/masterarbeit/results/samples_10";

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

  Eigen::VectorXd startPt(n);
  startPt.setConstant(1.0/double(10));
  startPt[0] = 0.9;
  startPt.normalize();

  MLDA(sampling_problems, n, startPt, num_samples, burn_in, proposal_var, proposal_var_l0, results_path);
}

void example2(){
  int n = 3;
  int num_samples = 1e3;
  int burn_in = 1e2;
  double proposal_var = 0.0001;
  double proposal_var_l0 = 0.05;
  std::string results_path = "/home/anne/Masterarbeit/masterarbeit/results/samples";

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
  startPt << 127, 127, 197;
  MLDA(sampling_problems, n, startPt, num_samples, burn_in, proposal_var, proposal_var_l0, results_path);
}


int main(){
  //example1();
  example2();
  return 0;
}
