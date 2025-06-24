# simulation flags
WANDB = true
POLICY = "SARSOP" # "CONOPS" "GREEDY" "SARSOP
VERBOSE = true
POLICYLOAD = false
EPISODES = 1

# These parameters dictate: range(START, END, SWEEP)
# if you only want to run 1 run, set START and END to same value and SWEEP as 1 

# incorrect penalty
λ_START = .99998
λ_END = .99998
λ_SWEEP = 1
# declare abiotic penalty
τ_START = 0.05
τ_END = 0.05
τ_SWEEP = 1
# discount factor
γ_START = 0.9
γ_END = 0.9
γ_SWEEP = 1

# Parameters to change for additional testing / off nominal testing
ACC_RATE = 10               # 270 μL per day
SAMPLE_MAX_CHAMBER = 100

NUM_INSTRUMENTS = 7 # One extra for accumulate, Wouldn't change unless you change Bayes Net and Action CPDS
LIFE_STATES = 3

##################### General Parameters for Instrument Sample Usage #############################

HRMS = 1                    #0 # 0.5e-6 # mL # organic compounds, just going to set to zero its too small
SMS_1 = 20                  #400 # μL # 0.4 # mL # amino acid characerization
SMS_2 = 20                  #100 # μL # 0.1 mL # Lipid Characterization
SMS = SMS_1 + SMS_2
μCE_LIF = 2                 #15 # μL #0.015 # mL # amino acid and lipid characterization
ESA_1 = 2                   #15 # μL #0.015  # mL # macronutrients
ESA_2 = 5                   #75 # μL #0.075  # mL # micronutrients
ESA_3 = 2                   #15 # μL #0.015  # mL # salinity
ESA = ESA_1 + ESA_2 + ESA_3
microscope = 1              # μL # 0.001 # mL # polyelectrolyte
nanopore = 100              #10000 # μL # 10  # mL cell like morphologies
none = 0                    #non-instrument actions

##################### Mapping Instrument Actions to sample characteristics #####################

ACTION_CPDS = Dict(
	1 => [:C5, :C7, :C8, :C10],   # HRMS ()
	2 => [:C5, :C6],              # SMS
	3 => [:C5, :C6],              # μCE_LIF
	4 => [:C7, :C8],              # ESA
	5 => [:C2, :C3],              # microscope
	6 => [:C1],                   # nanopore
)

##################### Libraries #####################

using Pkg
if WANDB
	Pkg.activate("wandbPkg")
	Pkg.instantiate()
	using Wandb
	using Logging
else
	Pkg.activate("LifeDetectionPkg")
	Pkg.instantiate()
end

# Libraries
using POMDPs
using POMDPTools
using Printf
using SARSOP
using POMDPLinter
using Distributions

##################### Additional Files #####################

# Including Bayesnet & base files for running POMDP and simulation
include("../src/bayes_net.jl")
include("../src/LifeDetectionPOMDP.jl")
include("../src/common/simulate.jl")
include("../src/common/utils.jl")

# Alternative Baseline Policies
include("../src/policies/conops.jl") # TODO: EDIT Doesn't work
include("../src/policies/greedy.jl") # TODO: EDIT Doesn't work

if WANDB
	include("../src/common/plotting_Wandb.jl")
end

##################### Start of the experiment sweeps #####################

for lambda in range(λ_START, λ_END, λ_SWEEP)
	for tau in range(τ_START, τ_END, τ_SWEEP)
		for gamma in range(γ_START, γ_END, γ_SWEEP)

			# TODO: add more for loops for different things we want to test

			# Generate new POMDP for each change in parameters
			pomdp = LifeDetectionPOMDP(
				bn=bn, # inside /src/common/bayesnet_generation.jl
				λ=lambda,
				τ=tau,
				ACTION_CPDS=ACTION_CPDS,
				max_obs=determineMaxObs(ACTION_CPDS, bn),
				inst=NUM_INSTRUMENTS,
				sample_volume=SAMPLE_MAX_CHAMBER,
				life_states=LIFE_STATES,
				ACC_RATE=ACC_RATE,
				sample_use=[HRMS, SMS, μCE_LIF, ESA, microscope, nanopore, none],
				discount=gamma)

			project_name = "Sweep_lambda_$(pomdp.λ)_tau_$(pomdp.τ)_gamma_$(pomdp.discount)_sample_$(pomdp.sample_volume)"

			if WANDB
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
						"tau" => pomdp.τ,
						"ACTION_CPDS" => pomdp.ACTION_CPDS,
						"max_obs" => pomdp.max_obs,
						"inst" => pomdp.inst,
						"sample_volume" => pomdp.sample_volume,
						"life_states" => pomdp.life_states,
						"ACC_RATE" => pomdp.ACC_RATE,
						"sample_use" => pomdp.sample_use,
						"gamma" => pomdp.discount,
					),
				)
			end

			if POLICY == "CONOPS"
				# Running CONOPS:
				rewards, accuracy = simulate_policyVLD(pomdp, "policy", "CONOPS", EPISODES, VERBOSE, WANDB, project_name) # SARSOP or conops or greedy

			elseif POLICY == "SARSOP"

				# Running SARSOP
				solver = SARSOPSolver(verbose=true, timeout=100)
				# @show_requirements POMDPs.solve(solver, pomdp)

				if POLICYLOAD
					policy = load_policy(pomdp, "policy.out")
				else
					policy = solve(solver, pomdp)
				end

				if WANDB
					Wandb.wandb.save("model.pomdpx")
					Wandb.wandb.save("policy.out")
					sleep(1)
					plot_alpha_dots(policy)
					Wandb.wandb.save("figures/plot_alpha_dots.png")

					close(run)
				end
				rewards, accuracy = simulate_policyVLD(pomdp, policy, "SARSOP", EPISODES, VERBOSE, WANDB, project_name) # SARSOP or conops or greedy

			elseif POLICY == "GREEDY"
				rewards, accuracy = simulate_policyVLD(pomdp, "policy", "GREEDY", EPISODES, VERBOSE, WANDB, project_name) # SARSOP or conops or greedy

			else
				println("No Valid Policy Selected")
			end
		end
	end
end


