#include "MUQ/SamplingAlgorithms/MLDAKernel.h"
#include "MUQ/SamplingAlgorithms/MLDAProposal.h"

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
#include <chrono>

#include <cmath>

namespace pt = boost::property_tree;
using namespace muq::Modeling;
using namespace muq::SamplingAlgorithms;
using namespace muq::Utilities;

extern std::vector<int> numberOfSamplesPerLevel;

float getRandom(int lower, int upper)
{
    std::default_random_engine generator;
    std::random_device rd; 
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> distribution(lower,upper);
    float res = distribution(gen);

    return res;
}

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

void MLDA(pt::ptree config){ 
    for(int k=0; k<config.get<int>("Setup.Chains"); k++){
      std::vector<std::shared_ptr<SamplingProblem>> sampling_problems;
      pt::ptree ptProposal;
      pt::ptree general_level_config = config.get_child("GeneralLevelConfig");

      int level = 0;
      int n = config.get_child("Sampling.Levels").size();
      BOOST_FOREACH(const pt::ptree::value_type &v, config.get_child("Sampling.Levels")) {
          pt::ptree level_config = config.get_child(v.second.data());

          json um_config;
          um_config["level"] = v.second.data();
          um_config["chain"] = k;
          sampling_problems.push_back(std::make_shared<SamplingProblem>(std::make_shared<UMBridgeModPiece>("localhost:4243", um_config)));

          if(level == 0){
            if(level_config.get_child_optional("ProposalVariance") == boost::none){
              ptProposal.add_child("ProposalVariance_0", general_level_config.get_child("ProposalVariance")); 
            }
            else{
              ptProposal.add_child("ProposalVariance_0", level_config.get_child("ProposalVariance")); 
            }
          }

          if(level != n-1){
            if(level_config.get_child_optional("Subsampling") == boost::none){
              ptProposal.put("Subsampling_" + std::to_string(level), general_level_config.get<int>("Subsampling"));
            }
            else{
              ptProposal.put("Subsampling_" + std::to_string(level), level_config.get<int>("Subsampling"));
            }

            if(level_config.get_child_optional("SubchainLength") == boost::none){
              ptProposal.put("SubchainLength_" + std::to_string(level), general_level_config.get<std::string>("SubchainLength"));
            }
            else{
              ptProposal.put("SubchainLength_" + std::to_string(level), level_config.get<std::string>("SubchainLength"));
            }
          }
          
          level ++;
      }
      
      auto proposal = std::make_shared<MLDAProposal>(ptProposal, sampling_problems.size()-1, sampling_problems);

      pt::ptree ptBlockID;
      ptBlockID.put("BlockIndex",0);
      std::vector<std::shared_ptr<TransitionKernel>> kernel(1);
      kernel[0] = std::make_shared<MLDAKernel>(ptBlockID,sampling_problems.back(),proposal);

      pt::ptree pt;
      pt.put("NumSamples", config.get<int>("Sampling.NumSamples")); // number of MCMC steps
      pt.put("BurnIn", config.get<int>("Sampling.BurnIn"));
      pt.put("PrintLevel",3);
      auto chain = std::make_shared<SingleChainMCMC>(pt,kernel);

      Eigen::VectorXd startPt(config.get<int>("Geometry.Dim"));
        if (config.get<int>("Geometry.Dim") == 2) {
            if (config.get<std::string>("Setup.Start") == "Random"){
              startPt << getRandom(config.get<int>("Sampling.StartDomain.x_Min"),config.get<int>("Sampling.StartDomain.x_Max")), getRandom(config.get<int>("Sampling.StartDomain.y_Min"),config.get<int>("Sampling.StartDomain.y_Max"));
            }
            else {
              startPt << config.get<int>("Sampling.StartPoint.x"), config.get<int>("Sampling.StartPoint.y");
            }
          }
        else {
            if (config.get<std::string>("Setup.Start") == "Random"){
              startPt <<  getRandom(config.get<int>("Sampling.StartDomain.x_Min"),config.get<int>("Sampling.StartDomain.x_Max")), getRandom(config.get<int>("Sampling.StartDomain.y_Min"),config.get<int>("Sampling.StartDomain.y_Max")), getRandom(0,2*M_PI);
            }
            else {
              startPt << config.get<int>("Sampling.StartPoint.x"), config.get<int>("Sampling.StartPoint.y"), config.get<double>("Sampling.StartPoint.rho");
            }
          }
      
      std::cout << "Start point:" << std::endl;
      std::cout << startPt << std::endl;

      std::shared_ptr<SampleCollection> samps = chain->Run(startPt);

      samps->WriteToFile(config.get<std::string>("Setup.OutputPath") + config.get<std::string>("Sampling.ResultFile")  + "_" + std::to_string(k) + ".h5");
      evaluate_samples(samps);
    }
}

void MH(pt::ptree config){   
    for(int k=0; k<config.get<int>("Setup.Chains"); k++){

      BOOST_FOREACH(const pt::ptree::value_type &v, config.get_child("Sampling.Levels")) {
        pt::ptree level_config = config.get_child(v.second.data());

        json um_config;
        um_config["level"] = v.second.data();
        um_config["chain"] = k;
        std::shared_ptr<SamplingProblem> sampling_problem = std::make_shared<SamplingProblem>(std::make_shared<UMBridgeModPiece>("localhost:4243", um_config));
        
        auto problem = sampling_problem;
        pt::ptree ptProposal;
        ptProposal.add_child("ProposalVariance", level_config.get_child("ProposalVariance"));

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

        Eigen::VectorXd startPt(config.get<int>("Geometry.Dim"));
        if (config.get<int>("Geometry.Dim") == 2) {
            if (config.get<std::string>("Setup.Start") == "Random"){
              startPt << getRandom(config.get<int>("Sampling.StartDomain.x_Min"),config.get<int>("Sampling.StartDomain.x_Max")), getRandom(config.get<int>("Sampling.StartDomain.y_Min"),config.get<int>("Sampling.StartDomain.y_Max"));
            }
            else {
              startPt << config.get<int>("Sampling.StartPoint.x"), config.get<int>("Sampling.StartPoint.y");
            }
          }
        else {
            if (config.get<std::string>("Setup.Start") == "Random"){
              startPt <<  getRandom(config.get<int>("Sampling.StartDomain.x_Min"),config.get<int>("Sampling.StartDomain.x_Max")), getRandom(config.get<int>("Sampling.StartDomain.y_Min"),config.get<int>("Sampling.StartDomain.y_Max")), getRandom(0,2*M_PI);
            }
            else {
              startPt << config.get<int>("Sampling.StartPoint.x"), config.get<int>("Sampling.StartPoint.y"), config.get<double>("Sampling.StartPoint.rho");
            }
          }

        std::cout << "Start point:" << std::endl;
        std::cout << startPt << std::endl;
        std::shared_ptr<SampleCollection> samps = chain->Run(startPt);

        samps->WriteToFile(config.get<std::string>("Setup.OutputPath") + config.get<std::string>("Sampling.ResultFile")  + "_" + std::to_string(k) + ".h5");
        evaluate_samples(samps);
      }
    }
}

int main(int argc, char *argv[]){
  // read configuration from file
  pt::ptree config;
  pt::json_parser::read_json(argv[1], config);

  // choose an algorithm
  std::string method = config.get<std::string>("Setup.Method");

  numberOfSamplesPerLevel.push_back(0);

  // run chosen algorithm using the config
  auto start = std::chrono::high_resolution_clock::now();
  if(method=="MCMC"){
    MH(config);
  }
  else if(method=="MLDA"){
    MLDA(config);
  }
  else{
    std::cout << "Method " +  method + " is not implemented." << std::endl;
  }
  auto stop = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::seconds>(stop - start);
  std::cout << "Duration time: " << duration.count() << "s" << std::endl;

  std::cout << numberOfSamplesPerLevel[0] << std::endl;
  std::cout << numberOfSamplesPerLevel[1] << std::endl;
  std::cout << numberOfSamplesPerLevel[2] << std::endl;

  return 0;
}

