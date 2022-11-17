#ifndef MLDAPROPOSAL_H_
#define MLDAPROPOSAL_H_

#include "MUQ/SamplingAlgorithms/MHProposal.h"
#include "MUQ/SamplingAlgorithms/MHKernel.h"
#include "MUQ/SamplingAlgorithms/SingleChainMCMC.h"
#include "MUQ/SamplingAlgorithms/SampleCollection.h"

namespace muq {
  namespace SamplingAlgorithms {
    /** @brief MLDA proposal.
        @details
     */
    class MLDAProposal : public MCMCProposal {
    public:
      MLDAProposal (boost::property_tree::ptree const& pt, int level, std::vector<std::shared_ptr<SamplingProblem>> sampling_problems)
      : MCMCProposal(pt,sampling_problems[level]),
        pt(pt),
        level(level),
        sampling_problems(sampling_problems)
      {
        assert(level > 0 && level < sampling_problems.size());
      }

      ~MLDAProposal() = default;

      virtual std::shared_ptr<SamplingState> Sample(std::shared_ptr<SamplingState> const& currentState) override;

      std::shared_ptr<SamplingState> GetFirstState() {
        assert(samps);
        return samps->front();
      }

      virtual double LogDensity(std::shared_ptr<SamplingState> const& currState,
                                std::shared_ptr<SamplingState> const& propState) override {
        return 0;
      }

    private:
      boost::property_tree::ptree const& pt;
      std::shared_ptr<SampleCollection> samps = nullptr;
      const int level;
      std::vector<std::shared_ptr<SamplingProblem>> sampling_problems;
    };


  }
}

#endif