
##################### Libraries #####################

using Pkg
Pkg.activate("wandbPkg")
Pkg.instantiate()

# Libraries
using POMDPs
using POMDPTools
using Printf
using SARSOP
using POMDPLinter
using Distributions
using Wandb
using Logging
using YAML
using IterTools  # for product()

##################### Read YAML #####################
config_file = get(ARGS, 1, "test/sweep.yaml")
config = YAML.load_file(config_file)

# simulation flags
WANDB = config["evaluations"]["wandb"]
POLICY =  config["evaluations"]["policy"] # "CONOPS" "GREEDY" "SARSOP
# Thresholds for decision
threshold_high =  config["evaluations"]["threshold_high"] #0.99999
threshold_low = config["evaluations"]["threshold_low"] #0.00001
VERBOSE = config["evaluations"]["verbose"]
POLICYLOAD = config["evaluations"]["policy_load"]
EPISODES = config["evaluations"]["episodes"]
proj_name = config["evaluations"]["proj_name"]

# Parameters to change for additional testing / off nominal testing
ACC_RATE = config["scenario"]["acc_rate"]               # 270 μL per day
SAMPLE_MAX_CHAMBER = config["scenario"]["sample_Max_Chamber"]  

params = config["parameters"]

lambdas = params["lambda"]
taus = params["tau"]
gammas = params["gamma"]

##################### General Parameters for Instrument Sample Usage #############################

NUM_INSTRUMENTS = 7 # One extra for accumulate, Wouldn't change unless you change Bayes Net and Action CPDS
LIFE_STATES = 3

HRMS = 1						#0 # 0.5e-6 # mL # organic compounds, just going to set to zero its too small
SMS_1 = 5 						#400 # μL # 0.4 # mL # amino acid characerization
SMS_2 = 1  						 #100 # μL # 0.1 mL # Lipid Characterization
SMS = SMS_1 + SMS_2

μCE_LIF = 1                  	 #15 # μL #0.015 # mL # amino acid and lipid characterization
ESA_1 = 1						#15 # μL #0.015  # mL # macronutrients
ESA_2 = 1   					#75 # μL #0.075  # mL # micronutrients
ESA_3 = 1						#15 # μL #0.015  # mL # macronutrients
ESA = ESA_1 + ESA_2 + ESA_3

microscope = 1					# μL # 0.001 # mL # polyelectrolyte
nanopore = 89					#10000 # μL # 10  # mL cell like morphologies
none = 0						#non-instrument actions

##################### Mapping Instrument Actions to sample characteristics #####################

ACTION_CPDS = Dict(
	1 => [:C5, :C7, :C8, :C10],   # HRMS ()
	2 => [:C5, :C6],              # SMS
	3 => [:C5, :C6],              # μCE_LIF
	4 => [:C7, :C8],              # ESA
	5 => [:C2, :C3],              # microscope
	6 => [:C1],                   # nanopore
)


##################### Additional Files #####################

# Including Bayesnet & base files for running POMDP and simulation
include("../src/bayes_net.jl")
include("../src/LifeDetectionPOMDP.jl")
include("../src/common/simulate.jl")
include("../src/common/utils.jl")

# Alternative Baseline Policies
include("../src/policies/conops.jl")
include("../src/policies/greedy.jl") # TODO: EDIT Doesn't work

if WANDB
	include("../src/common/plotting_wandb.jl")
end


##################### Read in configuration parameters #####################

# Create unique working directory
work_dir = config["evaluations"]["work_dir"]
mkpath(work_dir)

cd(work_dir) do

	println(IterTools.product(lambdas, taus, gammas))
	for (λ, τ, γ) in IterTools.product(lambdas, taus, gammas)
		println("Running with λ=$λ, τ=$τ, γ=$γ")

		# Generate new POMDP for each change in parameters
		pomdp = LifeDetectionPOMDP(
			bn=bn, # inside /src/common/bayesnet_generation.jl
			λ=λ,
			τ=τ,
			ACTION_CPDS=ACTION_CPDS,
			max_obs=determineMaxObs(ACTION_CPDS, bn),
			inst=NUM_INSTRUMENTS,
			sample_volume=SAMPLE_MAX_CHAMBER,
			life_states=LIFE_STATES,
			ACC_RATE=ACC_RATE,
			sample_use=[HRMS, SMS, μCE_LIF, ESA, microscope, nanopore, none],
			discount=γ)

		project_name = "$(proj_name)_lambda_$(pomdp.λ)_tau_$(pomdp.τ)_gamma_$(pomdp.discount)_sample_$(pomdp.sample_volume)"

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
			rewards, accuracy = simulate_policyVLD(pomdp, "policy", "CONOPS", EPISODES, VERBOSE, WANDB, project_name, threshold_high, threshold_low) # SARSOP or conops or greedy
			if WANDB
				sleep(1)
				close(run)
			end
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
				plot_alpha_action_heatmap(policy)
				Wandb.wandb.save("figures/plot_alpha_dots.png")
				Wandb.wandb.save("figures/plot_alpha_action_heatmap.png")

				close(run)
			end
			rewards, accuracy = simulate_policyVLD(pomdp, policy, "SARSOP", EPISODES, VERBOSE, WANDB, project_name, threshold_high, threshold_low) # SARSOP or conops or greedy


		elseif POLICY == "GREEDY"
			rewards, accuracy = simulate_policyVLD(pomdp, "policy", "GREEDY", EPISODES, VERBOSE, WANDB, project_name, threshold_high, threshold_low) # SARSOP or conops or greedy

		else
			println("No Valid Policy Selected")
		end
	end
end

