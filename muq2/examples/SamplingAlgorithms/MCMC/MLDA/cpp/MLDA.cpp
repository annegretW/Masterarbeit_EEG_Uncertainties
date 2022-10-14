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
#include <boost/property_tree/json_parser.hpp>

#include <boost/foreach.hpp>

namespace pt = boost::property_tree;
using namespace muq::Modeling;
using namespace muq::SamplingAlgorithms;
using namespace muq::Utilities;

void evaluate_samples(std::shared_ptr<SampleCollection> samps){
  Eigen::VectorXd sampMean = samps->Mean();
  std::cout << "\nSample Mean = \n" << sampMean.transpose() << std::endl;

  Eigen::VectorXd sampVar = samps->Variance();
  std::cout << "\nSample Variance = \n" << sampVar.transpose() << std::endl;

  Eigen::MatrixXd sampCov = samps->Covariance();
  std::cout << "\nSample Covariance = \n" << sampCov << std::endl;

  Eigen::VectorXd sampMom3 = samps->CentralMoment(3);
  std::cout << "\nSample Third Moment = \n" << sampMom3 << std::endl << std::endl;

  Eigen::VectorXd batchESS = samps->ESS("Batch");
  Eigen::VectorXd batchMCSE = samps->StandardError("Batch");
  Eigen::VectorXd batchAutocorrelation = (batchESS/(*samps).size()).cwiseInverse();

  Eigen::VectorXd spectralESS = samps->ESS("Wolff");
  Eigen::VectorXd spectralMCSE = samps->StandardError("Wolff");
  Eigen::VectorXd spectralAutocorrelation = (spectralESS/(*samps).size()).cwiseInverse();

  std::cout << "ESS:\n";
  std::cout << "  Batch:    " << batchESS.transpose() << std::endl;
  std::cout << "  Spectral: " << spectralESS.transpose() << std::endl;
  std::cout << "Autocorrelation:\n";
  std::cout << "  Batch:    " << batchAutocorrelation.transpose() << std::endl;
  std::cout << "  Spectral: " << spectralAutocorrelation.transpose() << std::endl;
  std::cout << "MCSE:\n";
  std::cout << "  Batch:    " << batchMCSE.transpose() << std::endl;
  std::cout << "  Spectral: " << spectralMCSE.transpose() << std::endl;
}

void MLDA(std::vector<std::shared_ptr<SamplingProblem>> sampling_problems, int n, Eigen::VectorXd startPt, int num_samples, int burn_in, std::vector<double> proposal_pos_var, std::vector<double> proposal_mom_var, std::vector<int> subsampling, std::string results_path){
  // MLDA
    pt::ptree ptProposal;
    ptProposal.put("Subsampling_0", subsampling[0]); // Subsampling on level 0
    ptProposal.put("Subsampling_1", subsampling[1]); // Subsampling on level 1

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

    evaluate_samples(samps);
}

void MH(std::shared_ptr<SamplingProblem> sampling_problem, boost::property_tree::ptree config){
    auto problem = sampling_problem;
    pt::ptree ptProposal = config.get_child("Level1.ProposalVariance");
    auto proposal = std::make_shared<MHProposal>(ptProposal, problem);

    pt::ptree ptBlockID;
    ptBlockID.put("BlockIndex",0);
    std::vector<std::shared_ptr<TransitionKernel>> kernel(1);
    kernel[0] = std::make_shared<MHKernel>(ptBlockID,problem,proposal);

    pt::ptree pt;
    pt.put("NumSamples", config.get<int>("Sampling.NumSamples")); // number of MCMC steps
    pt.put("BurnIn", config.get<int>("Sampling.BurnIn"));
    pt.put("PrintLevel",3);
    auto chain = std::make_shared<SingleChainMCMC>(pt,kernel);

    Eigen::VectorXd startPt(2);
    startPt << 127, 127;

    std::shared_ptr<SampleCollection> samps = chain->Run(startPt);

    samps->WriteToFile(config.get<std::string>("Setup.OutputPath") + ".h5");
    evaluate_samples(samps);
}

void example2d(std::string method, boost::property_tree::ptree config, int dim, Eigen::VectorXd startPt, int num_samples, int burn_in, std::vector<double> proposal_pos_var, std::vector<double> proposal_mom_var, std::vector<int> subsampling, std::string results_path){
  std::vector<std::shared_ptr<SamplingProblem>> sampling_problems;

  if(method=="MCMC"){
    json um_config;
    um_config["level"] = 1;
    std::shared_ptr<SamplingProblem> sampling_problem = std::make_shared<SamplingProblem>(std::make_shared<UMBridgeModPiece>("localhost:4243", um_config));
    MH(sampling_problem, config);
  }
  else if(method=="MLDA"){
    MLDA(sampling_problems, dim, startPt, num_samples, burn_in, proposal_pos_var, proposal_mom_var, subsampling, results_path);
  }
  else{
    std::cout << "Method " +  method + " is not implemented." << std::endl;
  }
}

int main(int argc, char *argv[]){
  boost::property_tree::ptree config;
  boost::property_tree::json_parser::read_json("/home/anne/Masterarbeit/masterarbeit/2d/config.json", config);

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

  std::string method = config.get<std::string>("Setup.Method");
  std::cout << method << std::endl;

  int g = atoi(argv[15]);
  int h = atoi(argv[16]);
  std::vector<int> subsampling;
  subsampling = {g, h};

  example2d(method, config, dim, startPt, num_samples, burn_in, proposal_pos_var, proposal_mom_var, subsampling, results_path);
  return 0;
}

