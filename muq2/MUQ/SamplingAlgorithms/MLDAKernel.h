#ifndef MLDAKERNEL_H_
#define MLDAKERNEL_H_

#include "MUQ/SamplingAlgorithms/TransitionKernel.h"
#include "MUQ/SamplingAlgorithms/MLDAProposal.h"

#include "MUQ/Utilities/AnyHelpers.h"
#include "MUQ/Utilities/RandomGenerator.h"

#include <boost/property_tree/ptree.hpp>
#include <iomanip>

namespace muq {
  namespace SamplingAlgorithms {

    class MLDAKernel : public TransitionKernel {
    public:

      MLDAKernel(boost::property_tree::ptree const& pt,
               std::shared_ptr<AbstractSamplingProblem> problem,
               std::shared_ptr<MLDAProposal> proposal)
      : TransitionKernel(pt, problem),
        proposal(proposal)
        {}

      ~MLDAKernel() = default;

      virtual void PostStep(unsigned int const t, std::vector<std::shared_ptr<SamplingState>> const& state) override;

      virtual void PrintStatus(std::string prefix) const override {
        std::stringstream msg;
        msg << std::setprecision(2);
        msg << prefix << "MLDAKernel acceptance Rate = "  << 100.0*double(numAccepts)/double(numCalls) << "%";

        std::cout << msg.str() << std::endl;
      }

      virtual std::vector<std::shared_ptr<SamplingState>> Step(unsigned int const t, std::shared_ptr<SamplingState> prevState) override;

      virtual double AcceptanceRate() const{return double(numAccepts)/double(numCalls);};

    protected:
      std::shared_ptr<MLDAProposal> proposal;

      unsigned int numCalls = 0;
      unsigned int numAccepts = 0;

    };
  }
}

#endif