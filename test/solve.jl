
using POMDPs
using POMDPTools
using Printf
using SARSOP
using POMDPLinter
using Distributions
using Plots

include("../src/bayesNet.jl")
include("../src/binaryLD.jl")
include("../src/common/plotting.jl")
include("../src/common/simulate.jl")
include("../src/common/utils.jl")
include("greedy.jl")


num_instruments = 4

# Bayes Net:
variable_specs = [(:l, 2), (:a, 2), (:p, 2), (:c, 2), (:h, 2), (:i, 2)]
dependencies = [(:l, :a), (:l, :p), (:l, :c), (:a, :h), (:a, :i), (:p, :i), (:c, :i)]
probability_tables = [
    ([:l], [(l=1,) => 0.5, (l=2,) => 0.5]),
    ([:a, :l], [(a=1, l=1) => 0.9,  (a=2, l=1) => 0.1,  (a=1, l=2) => 0.05, (a=2, l=2) => 0.95]),
    ([:p, :l], [(p=1, l=1) => 0.7,  (p=2, l=1) => 0.3,  (p=1, l=2) => 0.2,  (p=2, l=2) => 0.8 ]),
    ([:c, :l], [(c=1, l=1) => 0.75, (c=2, l=1) => 0.25, (c=1, l=2) => 0.3,  (c=2, l=2) => 0.7 ]),
    ([:h, :a], [(h=1, a=1) => 0.9,  (h=2, a=1) => 0.1,  (h=1, a=2) => 0.15, (h=2, a=2) => 0.85]),
    ([:i, :a], [(i=1, a=1) => 0.7,  (i=2, a=1) => 0.3,  (i=1, a=2) => 0.6,  (i=2, a=2) => 0.4 ]),
    ([:i, :p], [(i=1, p=1) => 0.8,  (i=2, p=1) => 0.2,  (i=1, p=2) => 0.9,  (i=2, p=2) => 0.1 ]),
    ([:i, :c], [(i=1, c=1) => 0.8,  (i=2, c=1) => 0.2,  (i=1, c=2) => 0.7,  (i=2, c=2) => 0.3 ]),
    ]

bn = bayes_net(variable_specs, dependencies, probability_tables)
plot_bayes_net(bn) # draw graph

reward_list = []
acc_list = []
end_λ = 1 # 1000
for λ in range(1, end_λ)
    pomdp = binaryLifeDetectionPOMDP(inst=num_instruments, bn=bn, λ=λ,  k=[0.1, 0.8, 0.6, 0.2], discount=0.9)
    solver = SARSOPSolver(verbose=true, timeout=100)
    policy = solve(solver, pomdp)
    plot_alpha_vectors(policy)
    rewards, acc = simulate_policy(pomdp, policy, "SARSOP", 200,false) # SARSOP or greedy
    push!(reward_list, rewards)
    push!(acc_list, acc)
end

#print(reward_list)
#print(acc_list)

# Prepare the data ranges
x = range(1, end_λ)

# Create two separate plots
p1 = scatter(x, reward_list, color=:blue, xlabel="λ", ylabel="Reward Value", title="Reward", label="rewards")
p2 = scatter(x, acc_list, color=:red, xlabel="λ", ylabel="Accuracy (0 to 1)", title="Accuracy", label="Accuracy")

# Combine them side by side
plot(p1, p2, layout=(1, 2), size=(800, 400), title="Pareto Frontier")
# policy = load_policy(pomdp,"policy.out")
# simulate_policy(pomdp, policy, "greedy",10) # SARSOP or greedy

# @show_requirements POMDPs.solve(solver, pomdp)