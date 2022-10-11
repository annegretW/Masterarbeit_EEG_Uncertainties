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
#include <boost/foreach.hpp>

namespace pt = boost::property_tree;
using namespace muq::Modeling;
using namespace muq::SamplingAlgorithms;
using namespace muq::Utilities;

void MLDA(std::vector<std::shared_ptr<SamplingProblem>> sampling_problems, int n, Eigen::VectorXd startPt, int num_samples, int burn_in, std::vector<double> proposal_pos_var, std::vector<double> proposal_mom_var, std::string results_path){
  // MLDA
    pt::ptree ptProposal;
    ptProposal.put("Subsampling_0", 5); // Subsampling on level 0
    ptProposal.put("Subsampling_1", 3); // Subsampling on level 1
    ptProposal.put("Subsampling_2", 1); // Subsampling on level 2

    ptProposal.put("Proposal_Variance_Pos_0", proposal_pos_var[0]); // Proposal Variance on coarsest level
    ptProposal.put("Proposal_Variance_Pos_1", proposal_pos_var[1]);
    ptProposal.put("Proposal_Variance_Pos_2", proposal_pos_var[2]);

    ptProposal.put("Proposal_Variance_Mom_0", proposal_mom_var[0]); // Proposal Variance on coarsest level
    ptProposal.put("Proposal_Variance_Mom_1", proposal_mom_var[1]);
    ptProposal.put("Proposal_Variance_Mom_2", proposal_mom_var[2]);

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

void MH(std::vector<std::shared_ptr<SamplingProblem>> sampling_problems, int n, Eigen::VectorXd startPt, int num_samples, int burn_in, std::vector<double> proposal_pos_var, std::vector<double> proposal_mom_var, std::string results_path){
    for (int level = 0; level < sampling_problems.size(); level++) {
      auto problem = sampling_problems[level];

      pt::ptree ptProposal;      
      pt::ptree children;
      pt::ptree child1, child2, child3;

      child1.put("", proposal_pos_var[level]);
      child2.put("", proposal_pos_var[level]);
      child3.put("", proposal_mom_var[level]);

      children.push_back(std::make_pair("", child1));
      children.push_back(std::make_pair("", child2));
      children.push_back(std::make_pair("", child3));

      ptProposal.add_child("ProposalVariance", children);

      //ptProposal.put<double>("ProposalVariance", proposal_pos_var[level]); 

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

void example1(int num_samples, int burn_in, std::vector<double> proposal_pos_var, std::vector<double> proposal_mom_var, std::string results_path){
  int n = 3;

  //std::string results_path = "/home/anne/Masterarbeit/masterarbeit/results/samples2";

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
  {
    json config;
    config["level"] = 4;
    sampling_problems.push_back(std::make_shared<SamplingProblem>(std::make_shared<UMBridgeModPiece>("localhost:4243", config)));
  }


  Eigen::VectorXd startPt(3);
  startPt << 127, 127, 127;
  MLDA(sampling_problems, n, startPt, num_samples, burn_in, proposal_pos_var, proposal_mom_var, results_path);
}

void example2d(std::string mode, int dim, Eigen::VectorXd startPt, int num_samples, int burn_in, std::vector<double> proposal_pos_var, std::vector<double> proposal_mom_var, std::string results_path){
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

  if(mode=="MH"){
    MH(sampling_problems, dim, startPt, num_samples, burn_in, proposal_pos_var, proposal_mom_var, results_path);
  }
  else if(mode=="MLDA"){
    MLDA(sampling_problems, dim, startPt, num_samples, burn_in, proposal_pos_var, proposal_mom_var, results_path);
  }
}

int main(int argc, char *argv[]){
  int dim = atoi(argv[1]);

  int num_samples = atoi(argv[2]);
  int burn_in = atoi(argv[3]);

  std::string results_path = argv[4];

  double a = atof(argv[5]);
  double b = atof(argv[6]);
  double c = atof(argv[7]);

  std::vector<double> proposal_pos_var;
  proposal_pos_var = {a, b, c};

  double d = atof(argv[8]);
  double e = atof(argv[9]);
  double f = atof(argv[10]);

  std::vector<double> proposal_mom_var;
  proposal_mom_var = {d, e, f};

  Eigen::VectorXd startPt(dim);
  if(dim==2){
    startPt << atoi(argv[11]), atoi(argv[12]);
  }
  else{
    startPt << atoi(argv[11]), atoi(argv[12]), atoi(argv[13]);
  }

  std::string mode = atof(argv[14]);
  
  std::cout << "Running MLDA with the following parameters:" << std::endl;
  std::cout << "Iterations: " << num_samples << std::endl;
  std::cout << "Burn In: " << burn_in << std::endl;
  std::cout << "Levels: " << argc-4 << std::endl;

  example2d(mode, dim, startPt, num_samples, burn_in, proposal_pos_var, proposal_mom_var, results_path);
  return 0;
}

