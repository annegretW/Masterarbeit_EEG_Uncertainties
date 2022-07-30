import muq.Modeling as mm
import muq.SamplingAlgorithms as ms
import muq.Approximation as ma
import muq.Utilities as mu

import matplotlib.pyplot as plt
import numpy as np

# Import a the model ModPiece from the EllipticInference example
import sys
sys.path.append("../DILI")

from FlowEquation import FlowEquation

# The standard deviation of the additive Gaussian noise
noiseStd = 1e-1

# Number of cells to use in Finite element discretization
numCells = 50
sourceTerm = (200.0/numCells)*np.ones(numCells)
mod = FlowEquation(sourceTerm)

# Create an exponential operator to transform log-conductivity into conductivity
cond = mm.ExpOperator(numCells)

# Combine the two models into a graph
graph = mm.WorkGraph()
graph.AddNode(mm.IdentityOperator(numCells), "Log-Conductivity")

graph.AddNode(cond, "Conductivity")
graph.AddEdge("Log-Conductivity", 0, "Conductivity", 0)

graph.AddNode(mod,"Forward Model")
graph.AddEdge("Conductivity",0,"Forward Model",0)

# Set up the Gaussian process prior on the log-conductivity
gpKern = ma.MaternKernel(1, 1.0, 0.1, 3.0/2.0)
gpMean = ma.ZeroMean(1,1)
prior = ma.GaussianProcess(gpMean,gpKern).Discretize(mod.xs[0:-1].reshape(1,-1))

graph.AddNode(prior.AsDensity(), "Prior")
graph.AddEdge("Log-Conductivity",0,"Prior",0)

# Generate a "true" log conductivity
trueLogK = prior.Sample()

trueHead = mod.Evaluate([np.exp(trueLogK)])[0]
obsHead = trueHead + noiseStd*mu.RandomGenerator.GetNormal(numCells+1)[:,0]

# Set up the likelihood and posterior
likely = mm.Gaussian(trueHead, noiseStd*noiseStd*np.ones((numCells+1,)))

graph.AddNode(likely.AsDensity(),"Likelihood")
graph.AddNode(mm.DensityProduct(2), "Posterior")

graph.AddEdge("Forward Model",0,"Likelihood",0)
graph.AddEdge("Prior",0,"Posterior",0)
graph.AddEdge("Likelihood",0,"Posterior",1)

graph.Visualize('ModelGraph.png')

# Construct the posterior density
postDens = graph.CreateModPiece("Posterior")
problem = ms.SamplingProblem(postDens)

#### MCMC

# Define the sampling problem
problem = ms.SamplingProblem(postDens)

# Define a two-stage delayed rejection sampler constructed from pCN proposals
opts = dict()
opts['NumSamples'] = 250000 # Number of MCMC steps to take in each chain
opts['BurnIn'] = 5000      # Number of steps to throw away at beginning of each chain
opts['PrintLevel'] = 3     # in {0,1,2,3} Verbosity of the output

# Aggressive first proposal
opts['Beta'] = 0.1
prop1 = ms.CrankNicolsonProposal(opts, problem, prior)

# Conservative second proposal
opts['Beta'] = 0.02
prop2 = ms.CrankNicolsonProposal(opts, problem, prior)

# Use the proposal to construct a Metropolis-Hastings kernel
kern = ms.DRKernel(opts, problem, [prop1,prop2])

# Construct multiple MCMC chains with diffuse starting points
numChains = 4
chains = [None]*numChains

for i in range(numChains):
    # Construct the MCMC sampler using this transition kernel
    sampler = ms.SingleChainMCMC(opts, [kern])

    # Run the MCMC sampler, starting from a random draw of the prior distribution
    chains[i] = sampler.Run([prior.Sample()])

# Compute the convergence diagnostic.  If close to 1, the chains have converged
mpsrf = ms.Diagnostics.Rhat(chains, {'Split':True, 'Transform':False, 'Multivariate':True})
print('MPSRF: {:0.3f}'.format(mpsrf[0]))
if(mpsrf[0]>1.1):
    print('  Chains have NOT converged!  Consider increasing the length of each chain.')

# Compute the total effective sample size
ess = np.sum([samps.ESS(method="MultiBatch") for samps in chains])
print('ESS: {:0.1f}'.format(ess))

# Extract the posteior samples as a matrix and compute some posterior statistics
sampMat = np.hstack([samps.AsMatrix() for samps in chains])

postMean = np.mean(sampMat,axis=1)
q05 = np.percentile(sampMat,5,axis=1)
q95 = np.percentile(sampMat,95,axis=1)

# Plot the results
fig, axs = plt.subplots(nrows=3,figsize=(12,8))
axs[0].plot(trueHead, label='True Head')
axs[0].plot(obsHead,'.k',label='Observed Head')
axs[0].legend()
axs[0].set_title('Data')
axs[0].set_xlabel('Position, $x$')
axs[0].set_ylabel('Hydraulic head $h(x)$')

axs[1].fill_between(mod.xs[0:-1], q05, q95, alpha=0.5,label='5%-95% CI')
axs[1].plot(mod.xs[0:-1],postMean, label='Posterior Mean')
axs[1].plot(mod.xs[0:-1],trueLogK,label='Truth')
axs[1].legend()
axs[1].set_title('Posterior on log(K)')
axs[1].set_xlabel('Position, $x$')
axs[1].set_ylabel('Log-Conductivity $log(K(x))$')

axs[2].plot(sampMat[0,:], label='$log K_0$')
axs[2].plot(sampMat[25,:],label='$log K_{25}$')
axs[2].plot(sampMat[45,:],label='$log K_{45}')
axs[2].set_title('Concatenated MCMC Traces')

ymin, ymax = axs[2].get_ylim()
startInd = 0
for chain in chains:
    axs[2].plot([startInd, startInd], [ymin,ymax], '--k')
    startInd += chain.size()
axs[2].set_ylim(ymin,ymax)

axs[2].set_xlabel('MCMC Iteration (after burnin)')
axs[2].set_ylabel('Log-Conductivity')
plt.show()
