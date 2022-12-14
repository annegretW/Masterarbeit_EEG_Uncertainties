#include "MUQ/SamplingAlgorithms/MHProposal.h"

#include "MUQ/Modeling/Distributions/Gaussian.h"

#include "MUQ/Utilities/AnyHelpers.h"

#include <boost/foreach.hpp>
#include <cmath>

namespace pt = boost::property_tree;
using namespace muq::Modeling;
using namespace muq::SamplingAlgorithms;
using namespace muq::Utilities;


REGISTER_MCMC_PROPOSAL(MHProposal)

MHProposal::MHProposal(pt::ptree const& pt,
                       std::shared_ptr<AbstractSamplingProblem> prob) :
                       MCMCProposal(pt,prob) {

  unsigned int problemDim = prob->blockSizes(blockInd);

  // compute the (diagonal) covariance for the proposal
  //const Eigen::VectorXd cov = pt.get("ProposalVariance", 1.0)*
  //                            Eigen::VectorXd::Ones(problemDim);

  Eigen::VectorXd cov(problemDim);
  int i = 0;
  BOOST_FOREACH(const boost::property_tree::ptree::value_type &v, pt.get_child("ProposalVariance")) {
          cov(i) = stod(v.second.data());
          i++;
  }

  // created a Gaussian with scaled identity covariance
  proposal = std::make_shared<Gaussian>(Eigen::VectorXd::Zero(problemDim), cov);
}

MHProposal::MHProposal(pt::ptree const& pt,
                       std::shared_ptr<AbstractSamplingProblem> prob,
                       std::shared_ptr<GaussianBase> proposalIn) :
                       MCMCProposal(pt,prob), proposal(proposalIn) {}

std::shared_ptr<SamplingState> MHProposal::Sample(std::shared_ptr<SamplingState> const& currentState) {
  assert(currentState->state.size()>blockInd);

  // the mean of the proposal is the current point
  std::vector<Eigen::VectorXd> props = currentState->state;
  assert(props.size()>blockInd);
  Eigen::VectorXd const& xc = currentState->state.at(blockInd);

  Eigen::VectorXd prop = proposal->Sample();
  // std::cout << xc + prop << std::endl;
  Eigen::VectorXd newState = xc + prop;
  //if(newState[0]<0){newState[0]=0;}
  //else if(newState[0]>256){newState[0]=256;}
  //if(newState[1]<0){newState[1]=0;}
  //else if(newState[1]>256){newState[1]=256;}
  //if(newState[2]<0){newState[2]=0;}
  //else if(newState[2]>=2*M_PI){newState[2]=0;}

  props.at(blockInd) = newState;

  // store the new state in the output
  return std::make_shared<SamplingState>(props, 1.0);
}

double MHProposal::LogDensity(std::shared_ptr<SamplingState> const& currState,
                              std::shared_ptr<SamplingState> const& propState) {

  Eigen::VectorXd diff = propState->state.at(blockInd)-currState->state.at(blockInd);
  return proposal->LogDensity(diff);//, std::pair<boost::any, Gaussian::Mode>(conditioned->state.at(blockInd), Gaussian::Mode::Mean));
}
