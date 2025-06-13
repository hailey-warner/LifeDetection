
using POMDPs
using POMDPTools
using Printf
using SARSOP
using POMDPLinter
using Distributions
using Plots
# using Statistics, Clustering, Plots


include("../src/bayesnet.jl")
include("../src/volumeLD.jl")
include("CONOPsOrbiter.jl")

include("../src/common/plotting.jl")
include("../src/common/simulate.jl")

HRMS = 1   # mL organic compounds, just going to set to zero its too small
SMS_1 = 20 # mL amino acid characerization
SMS_2 = 20 # mL lipid characterization
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


# Instrument Action to sample characteristics:
action_cpds = Dict(
	1 => [:C5, :C7, :C8, :C10],   # HRMS
	2 => [:C5, :C6],              # SMS
	3 => [:C5, :C6],              # μCE_LIF
	4 => [:C7, :C8],              # ESA
	5 => [:C2, :C3],              # microscope
	6 => [:C1],                    # nanopore
)

# Use a specific action
# action = 6
# max_obs = determine_max_obs(action_cpds) 
# life_state = 2
# dist_observations(action_to_cpds, life_state, action, max_obs)

# ci = CartesianIndices(Tuple(domain_sizes))[11248]


pomdp = volumeLifeDetectionPOMDP(
	bn = bn, # inside /src/bayesnet_discrete.jl
	λ = 0.99,
	action_cpds = action_cpds,
	max_obs = determine_max_obs(action_cpds),
	inst = 7, # one extra for not choosing anything
	sample_volume = 300,
	life_states = 3,
	surfaceAccRate = surfaceAccRate,
	sampleUse = [HRMS, SMS, μCE_LIF, ESA, microscope, nanopore, none],
	discount = 0.9)

# Running CONOPS:
rewards, accuracy = simulate_policyVLD(pomdp, "policy", "conops", 1, true) # SARSOP or conops or greedy

# Running SARSOP
solver = SARSOPSolver(verbose = true, timeout = 100)
@show_requirements POMDPs.solve(solver, pomdp)

# policy = load_policy(pomdp,"policy.out")
policy = solve(solver, pomdp)
plot_alpha_vectors(policy)
rewards, accuracy = simulate_policyVLD(pomdp, policy, "SARSOP", 100, false) # SARSOP or conops or greedy
plot_alpha_vectors_VLD(policy, pomdp, 0)



# verbose = true

# if verbose
#     println("--------------------------------START EPISODES---------------------------------")
# end

# total_episode_rewards = []
# accuracy = []

# # for episode in range(1, n_episodes)

# updater = DiscreteUpdater(pomdp)
# b = initialize_belief(updater, initialstate(pomdp))
# s = rand(initialstate(pomdp))

# if verbose
#     println("\nPolicy Simulation: Episode ",episode)
#     # println("Step | Action       | Observation | Belief(Life) | True State | Acc Sample | Total Reward ")

#     println("Step | Action        | Belief(Life) | True State | Acc Sample | Total Reward ")
#     println("-------------------------------------------------------------------------------")
# end
# step = 1
# total_reward = 0
# true_state = 1
# acc = 0
# o_old = 2
# temp_o_old = 0
# modeAcc = true
# prevAction = 0
# belief_life = pdf(b,o_old)
# type = "SARSOP"



# # get action, next state, and observation
# if type == "SARSOP"
#     a = action(policy, b)
# elseif type == "greedy"
#     a = action_greedy_policy(policy,b,step)
# elseif type == "conops"
#     a, modeAcc, prevAction = conopsOrbiter(pomdp, s, modeAcc, prevAction)
# end

# sp = rand(transition(pomdp, s, a))
# o = rand(observation(pomdp, a, sp))

# # Get reward and accumulate total reward
# r = reward(pomdp, s, a)
# total_reward += r

# # format action and observation names
# action_name = a >= pomdp.inst+1 ? (a == pomdp.inst+1 ? "Declare Dead" : "Declare Life") : (a == pomdp.inst ? "Accumulate" : "Sensor $(a)")
# accu , true_state = stateindex_to_state(s, pomdp.life_states)  # Save the current state before transitioning 
# println("Obs: ", o)

# s_check = s
# if o != 0
#     if true_state == 1 && step > 1
#         s_check = s_check + 1
#     end

#     belief_life = pdf(b,s_check)

# end
# # if o != 0
# #     obs_name = o_state == 1 ? "Negative" : "Positive"

# # else
# #     obs_name = "No Sense"
# #     # belief_life = ""
# # end

# # sum(pdf(b, state_to_stateindex(sample, 2)) for sample in 1:pomdp.sample_volume)

# if verbose
#     # show step details
#     # @printf("%3d  | %-12s | %-11s | %.3f        | %d          |  %d         | %.2f         \n", 
#     #         step, action_name, obs_name, belief_life, true_state, accu, total_reward)

#     @printf("%3d  | %-12s | %.3f        | %d          |  %d         | %.2f         \n", 
#             step, action_name, belief_life, true_state, accu, total_reward)
# end

# # update belief
# # if a != pomdp.inst
# # if o != pomdp.sample_volume*pomdp.life_states+pomdp.life_states+1
# b = update(updater, b, a, o)

# # end
# s = sp
# step += 1



# acc = 1-abs(true_state-1-pdf(b,true_state))
# push!(total_episode_rewards, total_reward)
# push!(accuracy, acc)
