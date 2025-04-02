include("../src/utils.jl")
include("../src/bayesNet.jl")
include("../src/binaryLD.jl")

using POMDPs
using POMDPTools
using Printf
using SARSOP
using POMDPLinter
using Distributions
using Plots

num_instruments = 3
pomdp = binaryLifeDetectionPOMDP(inst=num_instruments, bn=bn, k=[0.1, 0.8, 0.6], discount=0.9)

solver = SARSOPSolver(verbose = true, timeout=100)
policy = solve(solver, pomdp)

function plot_alpha_vectors(policy::AlphaVectorPolicy)
    alpha_vectors = policy.alphas # get alpha vectors
    num_states = size(alpha_vectors, 2) 
    num_vectors = size(alpha_vectors, 1) # number of alpha vectors
    
    # x-axis represents belief L = 1 (confidence in sample being alive)
    b = range(0, 1, length=100)

    # plot each alpha vector
    plt = plot(title="Alpha Vectors (Piecewise-Linear Value Function)",
               xlabel="Belief in L=1", ylabel="Value Function")

    for i in 1:num_vectors
        # Compute the value function for belief state b
        # V(b) = α₁*b + α₂*(1-b)
        V_b = [alpha_vectors[i][1] * b_i + alpha_vectors[i][2] * (1 - b_i) for b_i in b]
        
        # Plot each alpha vector as a line
        # note : if action i isn't present, it means running instrument i was not optimal.
        plot!(b, V_b, label="α_$i (a=$(policy.action_map[i]))")
    end

    display(plt)
end

function simulate_policy(pomdp, policy, n_episodes=1)
    updater = DiscreteUpdater(pomdp)
    b = initialize_belief(updater, initialstate(pomdp))
    s = rand(initialstate(pomdp))
    
    println("\nPolicy Simulation:")
    println("Step | Action | Observation | Belief(Life)")
    println("-----------------------------------------")
    
    step = 1
    while !isterminal(pomdp, s) && step ≤ 10  # max 10 steps

        # get action, next state, and observation
        a = action(policy, b)
        sp = rand(transition(pomdp, s, a))
        o = rand(observation(pomdp, a, sp))
        
        # format action and observation names
        action_name = a ≤ 2 ? (a == 1 ? "Declare Dead" : "Declare Life") : "Sensor $(a-2)"
        obs_name = o == 1 ? "Negative" : "Positive"
        
        # show step details
        @printf("%3d  | %-12s | %-11s | %.3f\n", 
                step, action_name, obs_name, pdf(b, 2))
        
        # update belief
        b = update(updater, b, a, o)
        s = sp
        step += 1
    end
end

simulate_policy(pomdp, policy)
plot_alpha_vectors(policy)

# @show_requirements POMDPs.solve(solver, pomdp)