include("utils.jl")
include("bayesNet.jl")
include("binaryLD.jl")

bn
# using StaticArrays
# using POMDPs
import POMDPs
using SARSOP
using POMDPLinter
# using DiscreteValueIteration

num_inst = 3
# 5 is setting the number of instruments
pomdp = binaryLifeDetectionPOMDP(inst = num_inst,
                                bn = bn)

solver = SARSOPSolver(verbose = true, timeout=100)
policy = solve(solver, pomdp)

using Plots

function plot_alpha_vectors(policy::AlphaVectorPolicy)
    alpha_vectors = policy.alphas  # Get the alpha vectors
    num_states = size(alpha_vectors, 2)  # Number of states
    num_vectors = size(alpha_vectors, 1)  # Number of alpha vectors
    
    # Create x-axis for belief (assuming 2 states, belief in state 1)
    b = range(0, 1, length=100)  # Belief in state 1 (L=1)

    # Plot each alpha vector
    plt = plot(title="Alpha Vectors (Piecewise-Linear Value Function)",
               xlabel="Belief in L=1", ylabel="Value Function")

    for i in 1:num_vectors
        # Compute the value function for belief state b
        V_b = [alpha_vectors[i][1] * b_i + alpha_vectors[i][2] * (1 - b_i) for b_i in b]
        
        # Plot each alpha vector as a line
        plot!(b, V_b, label="α_$i (a=$(policy.action_map[i]))")
    end
    
    display(plt)
end

plot_alpha_vectors(policy)

# @show_requirements POMDPs.solve(solver, pomdp)
