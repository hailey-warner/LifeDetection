include("../src/utils.jl")
include("../src/bayesNet.jl")
include("../src/binaryLD.jl")
include("../src/common/plotting.jl")
include("../src/common/simulate.jl")

include("greedy.jl")

using POMDPs
using POMDPTools
using Printf
using SARSOP
using POMDPLinter
using Distributions
using Plots

num_instruments = 3
pomdp = binaryLifeDetectionPOMDP(inst=num_instruments, bn=bn, k=[0.1, 0.8, 0.6], discount=0.9)

if false
    solver = SARSOPSolver(verbose = true, timeout=100)
    policy = solve(solver, pomdp)

    simulate_policy(pomdp, policy)
    plot_alpha_vectors(policy)
end

policy = load_policy(pomdp,"policy.out")
simulate_policy(pomdp, policy, "greedy") # SARSOP or greedy

# @show_requirements POMDPs.solve(solver, pomdp)