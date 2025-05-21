
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
nanopore = 100 #10000 # μL # 10  # mL cell like morphologiess
none = 0
surfaceAccRate = 10 # 270 μL per day

pomdp = volumeLifeDetectionPOMDP(
            bn=bn, 
            λ=0.5, 
            inst=7, # one extra for not choosing anything
            sampleVolume=300, 
            lifeStates = 3,
            surfaceAccRate = surfaceAccRate, 
            sampleUse = [HRMS, SMS, μCE_LIF, ESA, microscope, nanopore, none],
            discount=0.9)
# pomdp = volumeLifeDetectionPOMDP(
#                 bn=bn,
#                 λ=0.95,
#                 inst=7,
#                 sampleVolume=300,
#                 lifeStates=3,
#                 surfaceAccRate=27,
#                 sampleUse=[1,1,1,1,1,1,1],
#                 discount=0.9
#             )  

rewards, accuracy = simulate_policyVLD(pomdp, "policy", "conops", 1, true) # SARSOP or greedy

# # transition(pomdp, 2406, 0)
solver = SARSOPSolver(verbose = true, timeout=100)
@show_requirements POMDPs.solve(solver, pomdp)

policy = solve(solver, pomdp)
plot_alpha_vectors(policy)
rewards, accuracy = simulate_policyVLD(pomdp, policy, "SARSOP", 1, true) # SARSOP or greedy
plot_alpha_vectors_VLD(policy,pomdp, 0)

# get alpha vectors and action map
alpha_vectors = policy.alphas
action_map = policy.action_map

num_vectors = size(alpha_vectors, 1)
b = range(0, 1, length=200)  # Belief in L=1 (life)
# b = range(1, pomdp.sample, length=200)  # Belief in L=1 (life)
# b = range(1, pomdp.sampleVolume*pomdp.lifeStates+pomdp.lifeStates, length=200)  # Belief in L=1 (life)

p = Plots.plot(title="Alpha Vectors at Sample Volume = $sample",
         xlabel="Belief in Life (P(L=1))", ylabel="Value",
         legend=:topright)

for i in 1:num_vectors
    α = alpha_vectors[i]
    # get indices for dead and life at this sample volume
    dead_idx = state_to_stateindex(30, 1)
    life_idx = state_to_stateindex(30, 2)

    # alpha vector values for this sample
    α_dead = α[dead_idx]
    α_life = α[life_idx]

    # compute V(b) = α_life * b + α_dead * (1 - b)
    V_b = [α_life * b_i + α_dead * (1 - b_i) for b_i in b]
    plot!(b, V_b, label="α_$i (a=$(action_map[i]))")#,ylim=(-.5, 0))
end

display(p)
savefig(p, "alpha_vectors_sample_$sample.png")
return p
# # get alpha vectors
# alpha_vectors = policy.alphas
# num_states = size(alpha_vectors, 2) 
# num_vectors = size(alpha_vectors, 1)

# # x-axis represents belief L = 1, or P(life)
# b = range(0, 1, length=100)

# # plot each alpha vector
# p = plot(title="Alpha Vectors (Piecewise-Linear Value Function)",
#             xlabel="Belief in L=1", ylabel="Value Function")

# for i in 1:num_vectors
#     # compute the value function for belief state b
#     # V(b) = α₁*b + α₂*(1-b)
#     V_b = [alpha_vectors[i][1] * b_i + alpha_vectors[i][2] * (1 - b_i) for b_i in b]
    
#     # plot each alpha vector as a line
#     # note : if action i isn't present, it means running instrument i was not optimal.
#     plot!(b, V_b, label="α_$i (a=$(policy.action_map[i]))")
# end

# display(p)
# savefig(p, "alpha_vectors.png")

# # Test state_to_index and index_to_state functions
# for sampleVol in 0:30
#     for lifeState in 1:3
#         index = state_to_stateindex(sampleVol, lifeState)
#         reconstructed_state = stateindex_to_state(index,3)  # Use the function from volumeLD.jl
#         println("Original Comps: $sampleVol, $lifeState, State: $index, Reconstructed Comps: $reconstructed_state")

#     end
# end
# println("SDFSDFSDF")


