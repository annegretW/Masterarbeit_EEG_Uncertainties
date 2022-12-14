#include "MUQ/SamplingAlgorithms/MLDAProposal.h"
#include "MUQ/SamplingAlgorithms/MLDAKernel.h"

using namespace muq::SamplingAlgorithms;

namespace pt = boost::property_tree;

extern std::vector<int> numberOfSamplesPerLevel;
std::vector<int> numberOfSamplesPerLevel = {};

std::shared_ptr<SamplingState> MLDAProposal::Sample(std::shared_ptr<SamplingState> const& currentState) {
  auto problem = sampling_problems[level-1];
  std::vector<std::shared_ptr<TransitionKernel>> kernel(1);

  if (level-1 == 0) { // Coarsest level: Simple MCMC
    pt::ptree ptProposal;      

    ptProposal.add_child("ProposalVariance", pt.get_child("ProposalVariance_0"));
    auto proposal = std::make_shared<MHProposal>(ptProposal, problem);
    boost::property_tree::ptree ptBlockID;
    ptBlockID.put("BlockIndex",0);
    kernel[0] = std::make_shared<MHKernel>(ptBlockID,problem,proposal);

  } else {
    auto proposal = std::make_shared<MLDAProposal>(pt, level-1, sampling_problems);

    boost::property_tree::ptree ptBlockID;
    ptBlockID.put("BlockIndex",0);
    kernel[0] = std::make_shared<MLDAKernel>(ptBlockID,problem,proposal);
  }

  boost::property_tree::ptree pt_subchain;
  // Subsampling steps on level as defined in MLDA ptree

  int subchain_length;

  if(pt.get<std::string>("SubchainLength_" + std::to_string(level-1))=="Random"){
    std::default_random_engine generator;
    std::random_device rd; 
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> distribution(2,pt.get<int>("Subsampling_" + std::to_string(level-1)));
    subchain_length = distribution(gen);
  }
  else {
    subchain_length = pt.get<int>("Subsampling_" + std::to_string(level-1));
  }

  //std::cout << subchain_length << std::endl;
  //currentState->meta["NumSamples_"+std::to_string(level-1)] = subchain_length;
  pt_subchain.put("NumSamples", subchain_length);
  pt_subchain.put("BurnIn", 0);
  pt_subchain.put("PrintLevel",0);
  auto subchain = std::make_shared<SingleChainMCMC>(pt_subchain,kernel);

  samps = subchain->Run(currentState->state);
  //std::cout << "Level " << level << std::endl;
  //std::cout << subchain->NumSamps() << std::endl;
  //std::vector<int> numberOfSamples;
  numberOfSamplesPerLevel[level-1] += subchain->NumSamps();

  return samps->back();
}