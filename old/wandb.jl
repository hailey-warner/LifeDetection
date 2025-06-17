using Pkg
Pkg.activate("wandb")
using POMDPs
using POMDPTools
using Printf
using SARSOP
using POMDPLinter
using Distributions
using Plots

using Wandb
using Logging


include("../src/bayesnet_Wandb.jl")
include("../src/volumeLD.jl")
include("conops.jl")

include("../src/common/plotting_Wandb.jl")
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
ACC_RATE = 10 # 270 μL per day


# Instrument Action to sample characteristics:
ACTION_CPDS = Dict(
	1 => [:C5, :C7, :C8, :C10],   # HRMS ()
	2 => [:C5, :C6],              # SMS
	3 => [:C5, :C6],              # μCE_LIF
	4 => [:C7, :C8],              # ESA
	5 => [:C2, :C3],              # microscope
	6 => [:C1],                    # nanopore
)




#range(0,1,11)
#range(0.99,0.999,10)

for lambda in range(0.2, 1, 9)  #range(0.96,0.99,4) #range(0,1,11)
	pomdp = LifeDetectionPOMDP(
		bn=bn, # inside /src/bayesnet_discrete.jl
		λ=lambda,
		ACTION_CPDS=ACTION_CPDS,
		max_obs=determine_max_obs(ACTION_CPDS),
		inst=7, # one extra for not choosing anything
		sample_volume=100,
		life_states=3,
		ACC_RATE=ACC_RATE,
		sample_use=[HRMS, SMS, μCE_LIF, ESA, microscope, nanopore, none],
		discount=0.9)

	project_name = "Sweep_0.05Penalty_FalseNegatives_lambda_$(pomdp.λ)_sample_$(pomdp.sample_volume)_discount$(pomdp.discount)"

	# using PyCall
	# Start a new wandb run to track this script.
	run = WandbLogger( #Wandb.wandb.init(
		# Set the wandb entity where your project will be logged (generally your team name).
		entity="sherpa-rpa",
		# Set the wandb project where this run will be logged.
		project=project_name,
		# Track hyperparameters and run metadata.
		config=Dict(
			"bayesnet" => pomdp.bn,
			"lambda" => pomdp.λ,
			"ACTION_CPDS" => pomdp.action_cpds,
			"max_obs" => pomdp.max_obs,
			"inst" => pomdp.inst,
			"sample_volume" => pomdp.sample_volume,
			"life_states" => pomdp.life_states,
			"ACC_RATE" => pomdp.acc_rate,
			"sample_use" => pomdp.sample_use,
			"discount" => pomdp.discount,
		),
	)

	# Running CONOPS:
	# rewards, accuracy = simulate_policyVLD(pomdp, "policy", "conops", 1, true) # SARSOP or conops or greedy

	# Running SARSOP
	solver = SARSOPSolver(verbose=true, timeout=100)
	# @show_requirements POMDPs.solve(solver, pomdp)

	# policy = load_policy(pomdp,"policy.out")
	policy = solve(solver, pomdp)
	Wandb.wandb.save("model.pomdpx")
	Wandb.wandb.save("policy.out")
	sleep(1)
	plot_alpha_dots(policy)
	Wandb.wandb.save("figures/plot_alpha_dots.png")

	close(run)

	rewards, accuracy = simulate_policyVLD(pomdp, policy, "SARSOP", 100, true, true, project_name) # SARSOP or conops or greedy

end
# decision_tree(pomdp, policy)

# plot_alpha_vectors(policy)
# plot_alpha_vector_by_state(policy,pomdp)
# plot_pruned_alpha_vectors(policy)
# plot_pruned_alpha_vectors(policy,pomdp,0)
# plot_alpha_vectors_VLD(policy,pomdp, 0)


