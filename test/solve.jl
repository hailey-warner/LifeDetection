WANDB = false

# More for simulation
POLICY = "SARSOP" # "CONOPS" "GREEDY" "SARSOP
VERBOSE = true
POLICYLOAD = true
EPISODES = 1

# These parameters dictate: range(0.99,0.999,10)
# if you only want to run 1 run:
# set λ_START and λ_END to same value and λ_SWEEP as 1 
λ_START = 0
λ_END = 0
λ_SWEEP = 1

# Additional Parameter to change in testing
DISCOUNT = 0.9

# Parameters to change for additional testing / off nominal testing
ACC_RATE = 10               # 270 μL per day
SAMPLE_MAX_CHAMBER = 100

NUM_INSTRUMENTS = 7 # One extra for accumulate, Wouldn't change unless you change Bayes Net and Action CPDS
LIFE_STATES = 3

##################### General Parameters for Instrument Sample Usage #####################

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
none = 0

##################### Mapping Instrument Actions to sample characteristics #####################

ACTION_CPDS = Dict(
	1 => [:C5, :C7, :C8, :C10],   # HRMS ()
	2 => [:C5, :C6],              # SMS
	3 => [:C5, :C6],              # μCE_LIF
	4 => [:C7, :C8],              # ESA
	5 => [:C2, :C3],              # microscope
	6 => [:C1],                    # nanopore
)


##################### Libraries #####################

using Pkg
# if WANDB
Pkg.activate("wandbPkg")
Pkg.instantiate()
using Wandb
using Logging
# else
#     Pkg.activate("LifeDetectionPkg")
#     Pkg.instantiate()
# end

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

    # TODO: add more for loops for different things we want to test

    # Generate new POMDP for each change in parameters
	pomdp = LifeDetectionPOMDP(
		bn=bn, # inside /src/bayesnet_discrete.jl
		λ=lambda,
		ACTION_CPDS=ACTION_CPDS,
		max_obs=determineMaxObs(ACTION_CPDS,bn),
		inst= NUM_INSTRUMENTS, 
		sample_volume=SAMPLE_MAX_CHAMBER,
		life_states=LIFE_STATES,
		ACC_RATE=ACC_RATE,
		sample_use=[HRMS, SMS, μCE_LIF, ESA, microscope, nanopore, none],
		discount=DISCOUNT)

	project_name = "Sweep_0.05Penalty_FalseNegatives_lambda_$(pomdp.λ)_sample_$(pomdp.sample_volume)_discount$(pomdp.discount)"

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
                "ACTION_CPDS" => pomdp.ACTION_CPDS,
                "max_obs" => pomdp.max_obs,
                "inst" => pomdp.inst,
                "sample_volume" => pomdp.sample_volume,
                "life_states" => pomdp.life_states,
                "ACC_RATE" => pomdp.ACC_RATE,
                "sample_use" => pomdp.sample_use,
                "discount" => pomdp.discount,
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
            policy = load_policy(pomdp,"policy.out")
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


