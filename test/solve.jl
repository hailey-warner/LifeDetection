include("../src/bayesNet.jl")
include("../src/binaryLD.jl")
include("../src/common/plotting.jl")
include("../src/common/simulate.jl")
include("../src/common/utils.jl")


include("greedy.jl")

using POMDPs
using POMDPTools
using Printf
using SARSOP
using POMDPLinter
using Distributions
using Plots

num_instruments = 3
# Bayes Net:
variable_specs = [(:l, 2), (:a, 2), (:p, 2), (:c, 2)]
dependencies = [(:l, :a), (:l, :p), (:l, :c)]
probability_tables = [
    ([:l], [(l=1,) => 0.5, (l=2,) => 0.5]),
    ([:a, :l], [(a=1, l=1) => 0.9, (a=2, l=1) => 0.1, (a=1, l=2) => 0.1, (a=2, l=2) => 0.9]),
    ([:p, :l], [(p=1, l=1) => 0.7, (p=2, l=1) => 0.3, (p=1, l=2) => 0.3, (p=2, l=2) => 0.7]),
    ([:c, :l], [(c=1, l=1) => 0.75, (c=2, l=1) => 0.25, (c=1, l=2) => 0.25, (c=2, l=2) => 0.75])
]

bn = bayes_net(variable_specs, dependencies, probability_tables)
# println("YA")
# # Example Usage
# num_instruments = 3
# instrument_names = ["sensorA", "sensorB", "sensorC"]
# instrument_probs_alive = [0.9, 0.7, 0.75] # Probabilities of detection given life exists
# instrument_probs_dead = [0.1, 0.3, 0.25]  # Probabilities of false detection given no life

# bn = create_bayes_net(num_instruments, instrument_names, instrument_probs_alive, instrument_probs_dead)


# # Example Usage
# num_instruments = 3
# instrument_probs = [0.9, 0.7, 0.75] # Probabilities of detection given life exists

# bn = create_bayes_net(num_instruments, instrument_probs)


# a = (l=2, i1=1, i2=1, i3=1)
# probability(bn, Assignment(a))
# Example usage
# a = (l=2, a=1, p=1, c=1)
# probability(bn, Assignment(a))


pomdp = binaryLifeDetectionPOMDP(inst=num_instruments, bn=bn, k=[1, 0.05, 0.08], discount=0.9)

if true
    solver = SARSOPSolver(verbose = true, timeout=100)
    policy = solve(solver, pomdp)

    simulate_policy(pomdp, policy)
    plot_alpha_vectors(policy)
end

# policy = load_policy(pomdp,"policy.out")
simulate_policy(pomdp, policy, "greedy",10) # SARSOP or greedy

# @show_requirements POMDPs.solve(solver, pomdp)