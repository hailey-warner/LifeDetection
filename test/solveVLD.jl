
using POMDPs
using POMDPTools
using Printf
using SARSOP
using POMDPLinter
using Distributions
using Plots
# using Statistics, Clustering, Plots


include("../src/common/utils.jl")
include("../src/bayesNet_old.jl")
include("../src/volumeLD.jl")
include("CONOPsOrbiter.jl")

include("../src/common/plotting.jl")
include("../src/common/simulate.jl")

# Bayes Net:
variable_specs = [(:l, 2), (:h, 2), (:s, 2), (:μ, 2), (:E, 2), (:m, 2), (:n, 2)]
dependencies = [(:l, :h), (:l, :s), (:l, :μ), (:l, :E), (:l, :m), (:l, :n)]
probability_tables = [
    ([:l], [(l=1,) => 0.3, (l=2,) => 0.7]),
    ([:h, :l], [(h=1, l=1) => 0.55, (h=2, l=1) => 0.45, (h=1, l=2) => 0.45, (h=2, l=2) => 0.55]),
    ([:s, :l], [(s=1, l=1) => 0.7, (s=2, l=1) => 0.3, (s=1, l=2) => 0.45, (s=2, l=2) => 0.55]),
    ([:μ, :l], [(μ=1, l=1) => 0.65, (μ=2, l=1) => 0.35, (μ=1, l=2) => 0.45, (μ=2, l=2) => 0.55]),
    ([:E, :l], [(E=1, l=1) => 0.6, (E=2, l=1) => 0.4, (E=1, l=2) => 0.49, (E=2, l=2) => 0.51]),
    ([:m, :l], [(m=1, l=1) => 0.61, (m=2, l=1) => 0.39, (m=1, l=2) => 0.5, (m=2, l=2) => 0.5]),
    ([:n, :l], [(n=1, l=1) => 0.99, (n=2, l=1) => 0.01, (n=1, l=2) => 0.35, (n=2, l=2) => 0.65]),
    
]

bn = bayes_net(variable_specs, dependencies, probability_tables)

HRMS = 1 #0 # 0.5e-6 # mL # organic compounds, just going to set to zero its too small
SMS_1 = 20 #400 # μL # 0.4 # mL # amino acid characerization
SMS_2 = 20 #100 # μL # 0.1 mL # Lipid Characterization
SMS = SMS_1 + SMS_2
μCE_LIF = 2 #15 # μL #0.015 # mL # amino acid and lipid characterization
ESA_1 = 2 #15 # μL #0.015  # mL # macronutrients
ESA_2 = 5 #75 # μL #0.075  # mL # micronutrients
ESA_3 = 2 #15 # μL #0.015  # mL # salinity
ESA = ESA_1 + ESA_2 + ESA_3
microscope = 1 # μL # 0.001 # mL # polyelectrolyte
nanopore = 100 #10000 # μL # 10  # mL cell like morphologies
none = 0

# Running CONOPS:
rewards, accuracy = simulate_policyVLD(pomdp, "policy", "conops", n_episodes = 1, verbose = true) # SARSOP or conops or greedy

# Running SARSOP
solver = SARSOPSolver(verbose = true, timeout=100)
@show_requirements POMDPs.solve(solver, pomdp)

policy = solve(solver, pomdp)
plot_alpha_vectors(policy)
rewards, accuracy = simulate_policyVLD(pomdp, policy, "SARSOP", n_episodes = 1, verbose = true) # SARSOP or conops or greedy
plot_alpha_vectors_VLD(policy,pomdp, 0)
