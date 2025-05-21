
using POMDPs
using POMDPTools
using Printf
using SARSOP
using POMDPLinter
using Distributions
using Plots
# using Statistics, Clustering, Plots


# include("../src/common/utils.jl")
# include("../src/bayesNet_old.jl")
include("../src/bayesnet_discrete.jl")
include("../src/volumeLD.jl")
include("CONOPsOrbiter.jl")

include("../src/common/plotting.jl")
include("../src/common/simulate.jl")

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
surfaceAccRate = 10 # 270 μL per day


pomdp = volumeLifeDetectionPOMDP(
            bn=bn, # inside /src/bayesnet_discrete.jl
            λ=0.995, 
            inst=7, # one extra for not choosing anything
            sampleVolume=300, 
            lifeStates = 3,
            surfaceAccRate = surfaceAccRate, 
            sampleUse = [HRMS, SMS, μCE_LIF, ESA, microscope, nanopore, none],
            discount=0.9)

# Running CONOPS:
rewards, accuracy = simulate_policyVLD(pomdp, "policy", "conops", 1, true) # SARSOP or conops or greedy

# Running SARSOP
solver = SARSOPSolver(verbose = true, timeout=100)
@show_requirements POMDPs.solve(solver, pomdp)

policy = solve(solver, pomdp)
plot_alpha_vectors(policy)
rewards, accuracy = simulate_policyVLD(pomdp, policy, "SARSOP", 10, true) # SARSOP or conops or greedy
plot_alpha_vectors_VLD(policy,pomdp, 0)
