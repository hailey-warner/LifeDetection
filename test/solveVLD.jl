
using POMDPs
using POMDPTools
using Printf
using SARSOP
using POMDPLinter
using Distributions
using Plots
# using Statistics, Clustering, Plots


include("../src/common/utils.jl")
include("../src/bayesNet.jl")
include("../src/volumeLD.jl")

include("../src/common/plotting.jl")
include("../src/common/simulate.jl")

# Bayes Net:
variable_specs = [(:l, 2), (:h, 2), (:s, 2), (:μ, 2), (:E, 2), (:m, 2), (:n, 2)]
dependencies = [(:l, :h), (:l, :s), (:l, :μ), (:l, :E), (:l, :m), (:l, :n)]
probability_tables = [
    ([:l], [(l=1,) => 0.5, (l=2,) => 0.5]),
    ([:h, :l], [(h=1, l=1) => 0.7, (h=2, l=1) => 0.3, (h=1, l=2) => 0.3, (h=2, l=2) => 0.7]),
    ([:s, :l], [(s=1, l=1) => 0.75, (s=2, l=1) => 0.25, (s=1, l=2) => 0.25, (s=2, l=2) => 0.75]),
    ([:μ, :l], [(μ=1, l=1) => 0.6, (μ=2, l=1) => 0.4, (μ=1, l=2) => 0.4, (μ=2, l=2) => 0.6]),
    ([:E, :l], [(E=1, l=1) => 0.6, (E=2, l=1) => 0.4, (E=1, l=2) => 0.4, (E=2, l=2) => 0.6]),
    ([:m, :l], [(m=1, l=1) => 0.6, (m=2, l=1) => 0.4, (m=1, l=2) => 0.3, (m=2, l=2) => 0.7]),
    ([:n, :l], [(n=1, l=1) => 0.8, (n=2, l=1) => 0.2, (n=1, l=2) => 0.25, (n=2, l=2) => 0.75]),
    
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

pomdp = volumeLifeDetectionPOMDP(
    bn=bn,
    λ=0.95,
    inst=7,
    sampleVolume=500,
    lifeStates=3,
    surfaceAccRate=270,
    sampleUse=[1,1,1,1,1,1,1],
    discount=0.9
)

# pomdp = volumeLifeDetectionPOMDP(
#             bn=bn, 
#             λ=0.5, 
#             inst=7, # one extra for not choosing anything
#             sampleVolume=300, 
#             lifeStates = 3,
#             surfaceAccRate = 20, 
#             sampleUse = [HRMS, SMS, μCE_LIF, ESA, microscope, nanopore, none],
#             discount=0.9)

# transition(pomdp, 2406, 0)
solver = SARSOPSolver(verbose = true, timeout=100)
@show_requirements POMDPs.solve(solver, pomdp)

policy = solve(solver, pomdp)
plot_alpha_vectors(policy)