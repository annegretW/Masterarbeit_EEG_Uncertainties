#include "MUQ/SamplingAlgorithms/MLDAKernel.h"

using namespace muq::SamplingAlgorithms;

void MLDAKernel::PostStep(unsigned int const t, std::vector<std::shared_ptr<SamplingState>> const& state) {
  proposal->Adapt(t,state);
}

std::vector<std::shared_ptr<SamplingState>> MLDAKernel::Step(unsigned int const t, std::shared_ptr<SamplingState> prevState) {
  using namespace muq::Utilities;

  assert(proposal);


  double prev_logtarget;
  if(prevState->HasMeta("LogTarget") && (prevState->HasMeta("QOI") || problem->numBlocksQOI == 0) && !reeval ){
    prev_logtarget = AnyCast(prevState->meta["LogTarget"]);
  } else {
    prev_logtarget = problem->LogDensity(prevState);
    prevState->meta["LogTarget"] = prev_logtarget;
    if (problem->numBlocksQOI > 0) {
      prevState->meta["QOI"] = problem->QOI();
    }
  }

  std::shared_ptr<SamplingState> coarse_prop_state = proposal->Sample(prevState);
  double coarse_prop_logtarget = AnyCast(coarse_prop_state->meta["LogTarget"]);
  double coarse_prev_logtarget = AnyCast(proposal->GetFirstState()->meta["LogTarget"]);

  std::shared_ptr<SamplingState> prop_state = std::make_shared<SamplingState>(coarse_prop_state->state);

  double prop_logtarget = problem->LogDensity(prop_state);

  const double alpha = std::exp(prop_logtarget - prev_logtarget - (coarse_prop_logtarget - coarse_prev_logtarget));

  // accept/reject
  numCalls++;
  
  if(0){std::cout << std::to_string(prev_logtarget-coarse_prev_logtarget) << std::endl;
  std::cout << std::to_string(prop_logtarget) << std::endl;
  std::cout << std::to_string(prev_logtarget) << std::endl;
  std::cout << std::to_string(coarse_prop_logtarget) << std::endl;
  std::cout << std::to_string(coarse_prev_logtarget) << std::endl;
  std::cout << std::to_string(prop_logtarget - prev_logtarget - (coarse_prop_logtarget - coarse_prev_logtarget)) << std::endl;
  std::cout << std::to_string(std::exp(prop_logtarget - prev_logtarget - (coarse_prop_logtarget - coarse_prev_logtarget))) << std::endl;
  std::cout << "______________________________________" << std::endl;}

  // std::cout << std::to_string(alpha) << std::endl;
  if(RandomGenerator::GetUniform() < alpha) {
    numAccepts++;

    prop_state->meta["LogTarget"] = prop_logtarget;
    if (problem->numBlocksQOI > 0) {
      prop_state->meta["QOI"] = problem->QOI();
    }
    prop_state->meta["IsProposal"] = false;
    return std::vector<std::shared_ptr<SamplingState>>(1, prop_state);
  } else {
    return std::vector<std::shared_ptr<SamplingState>>(1, prevState);
  }
}