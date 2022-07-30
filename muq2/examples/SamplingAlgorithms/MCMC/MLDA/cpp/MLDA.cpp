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




int main(){

  // Built for the following benchmark problm (run via docker):
  // sudo docker run -it --network=host -p 4243:4243 linusseelinger/benchmark-analytic-banana

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

  { // MLDA

    pt::ptree ptProposal;
    ptProposal.put("Subsampling_0", 10); // Subsampling on level 0
    ptProposal.put("Subsampling_1", 3); // Subsampling on level 1
    auto proposal = std::make_shared<MLDAProposal>(ptProposal, sampling_problems.size()-1, sampling_problems);

    pt::ptree ptBlockID;
    ptBlockID.put("BlockIndex",0);
    std::vector<std::shared_ptr<TransitionKernel>> kernel(1);
    // TODO: MLDA kernel here
    kernel[0] = std::make_shared<MLDAKernel>(ptBlockID,sampling_problems.back(),proposal);

    pt::ptree pt;
    pt.put("NumSamples", 1e4); // number of MCMC steps
    pt.put("BurnIn", 1000);
    pt.put("PrintLevel",3);
    auto chain = std::make_shared<SingleChainMCMC>(pt,kernel);

    Eigen::VectorXd startPt(10);
    startPt.setConstant(0.5);

    std::shared_ptr<SampleCollection> samps = chain->Run(startPt);

    samps->WriteToFile("/home/anne/Masterarbeit/masterarbeit/results/samples_10_mlda.h5");
  }

  { // Single level MCMC Reference
    for (int level = 0; level < sampling_problems.size(); level++) {
      auto problem = sampling_problems[level];

      pt::ptree ptProposal;
      ptProposal.put("ProposalVariance",0.05);
      auto proposal = std::make_shared<MHProposal>(ptProposal, problem);

      int x = problem->blockSizes(0);
      std::cout << x << std::endl;

      pt::ptree ptBlockID;
      ptBlockID.put("BlockIndex",0);
      std::vector<std::shared_ptr<TransitionKernel>> kernel(1);
      kernel[0] = std::make_shared<MHKernel>(ptBlockID,problem,proposal);
      pt::ptree pt;
      pt.put("NumSamples", 1e4); // number of MCMC steps
      pt.put("BurnIn", 1000);
      pt.put("PrintLevel",3);
      auto chain = std::make_shared<SingleChainMCMC>(pt,kernel);

      Eigen::VectorXd startPt(10);
      startPt.setConstant(0.5);
    
      std::shared_ptr<SampleCollection> samps = chain->Run(startPt);

      samps->WriteToFile("/home/anne/Masterarbeit/masterarbeit/results/samples_10_l" + std::to_string(level) + ".h5");
    }
  }

/*
  // parameters for the sampler
  pt::ptree pt;
  pt.put("NumSamples", 1e4); // number of MCMC steps
  pt.put("BurnIn", 1e3);
  pt.put("PrintLevel",3);
  pt.put("KernelList", "Kernel1"); // Name of block that defines the transition kernel
  pt.put("Kernel1.Method","MHKernel");  // Name of the transition kernel class
  pt.put("Kernel1.Proposal", "MyProposal"); // Name of block defining the proposal distribution
  pt.put("Kernel1.MyProposal.Method", "MHProposal"); // Name of proposal class
  pt.put("Kernel1.MyProposal.ProposalVariance", 2.5); // Variance of the isotropic MH proposal

  Eigen::VectorXd startPt = mu;
  auto mcmc = MCMCFactory::CreateSingleChain(pt, problem);

  std::shared_ptr<SampleCollection> samps = mcmc->Run(startPt);

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

  Eigen::VectorXd spectralESS = samps->ESS("Wolff");
  Eigen::VectorXd spectralMCSE = samps->StandardError("Wolff");

  std::cout << "ESS:\n";
  std::cout << "  Batch:    " << batchESS.transpose() << std::endl;
  std::cout << "  Spectral: " << spectralESS.transpose() << std::endl;
  std::cout << "MCSE:\n";
  std::cout << "  Batch:    " << batchMCSE.transpose() << std::endl;
  std::cout << "  Spectral: " << spectralMCSE.transpose() << std::endl;
*/
  return 0;
}
