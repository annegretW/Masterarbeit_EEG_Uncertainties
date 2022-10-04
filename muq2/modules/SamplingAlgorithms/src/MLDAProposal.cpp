#include "MUQ/SamplingAlgorithms/MLDAProposal.h"
#include "MUQ/SamplingAlgorithms/MLDAKernel.h"

using namespace muq::SamplingAlgorithms;

namespace pt = boost::property_tree;


std::shared_ptr<SamplingState> MLDAProposal::Sample(std::shared_ptr<SamplingState> const& currentState) {
  auto problem = sampling_problems[level-1];

  std::vector<std::shared_ptr<TransitionKernel>> kernel(1);

  if (level-1 == 0) { // Coarsest level: Simple MCMC
    pt::ptree ptProposal;      
    pt::ptree children;
    pt::ptree child1, child2, child3;

    child1.put("", pt.get<double>("Proposal_Variance_"+ std::to_string(level-1)));
    child2.put("", pt.get<double>("Proposal_Variance_"+ std::to_string(level-1)));
    child3.put("", pt.get<double>("Proposal_Variance_"+ std::to_string(level-1)));

    children.push_back(std::make_pair("", child1));
    children.push_back(std::make_pair("", child2));
    children.push_back(std::make_pair("", child3));

    ptProposal.add_child("ProposalVariance", children);

    //boost::property_tree::ptree ptProposal;
    //ptProposal.put("ProposalVariance",pt.get<double>("Proposal_Variance_0"));
    //ptProposal.put("ProposalVariance",5);
    auto proposal = std::make_shared<MHProposal>(ptProposal, problem);

    boost::property_tree::ptree ptBlockID;
    ptBlockID.put("BlockIndex",0);
    kernel[0] = std::make_shared<MHKernel>(ptBlockID,problem,proposal);
  } else {
    pt::ptree ptProposal;      
    pt::ptree children;
    pt::ptree child1, child2, child3;

    child1.put("", pt.get<double>("Proposal_Variance_"+ std::to_string(level-1)));
    child2.put("", pt.get<double>("Proposal_Variance_"+ std::to_string(level-1)));
    child3.put("", pt.get<double>("Proposal_Variance_"+ std::to_string(level-1)));

    children.push_back(std::make_pair("", child1));
    children.push_back(std::make_pair("", child2));
    children.push_back(std::make_pair("", child3));

    ptProposal.add_child("ProposalVariance", children);

    //boost::property_tree::ptree ptProposal;
    //ptProposal.put("ProposalVariance",pt.get<double>("Proposal_Variance_"+ std::to_string(level-1)));
    //ptProposal.put("ProposalVariance",5);
    auto proposal = std::make_shared<MLDAProposal>(pt, level-1, sampling_problems);

    boost::property_tree::ptree ptBlockID;
    ptBlockID.put("BlockIndex",0);
    kernel[0] = std::make_shared<MLDAKernel>(ptBlockID,problem,proposal);
  }

  boost::property_tree::ptree pt_subchain;
  // Subsampling steps on level as defined in MLDA ptree
  pt_subchain.put("NumSamples", pt.get<int>("Subsampling_" + std::to_string(level-1)));
  pt_subchain.put("BurnIn", 0);
  pt_subchain.put("PrintLevel",0);
  auto subchain = std::make_shared<SingleChainMCMC>(pt_subchain,kernel);

  samps = subchain->Run(currentState->state);

  return samps->back();
}