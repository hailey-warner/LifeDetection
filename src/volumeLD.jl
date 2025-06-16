using POMDPs # TODO: check if i can delete this
using POMDPTools
include("../src/common/utils.jl")


# Custom constructor to handle dynamic initialization
function volumeLifeDetectionPOMDP(;
	bn::DiscreteBayesNet, # Bayesian Network,
	λ::Float64,
	action_cpds::Dict,
	max_obs::Int64,
	inst::Int64=7, # number of instruments / not using instrument
	sample_volume::Int64=500,
	life_states::Int64=3,
	acc_rate::Int64=270,
	sample_use::Vector{Int64}=[1, 1, 1, 1, 1, 1, 1], # cost of observations
	# k::Vector{Float64} = [HRMS*10e6, SMS*10e6, μCE_LI*10e6, ESA*10e6, microscope*10e6, nanopore*10e6], # cost of observations
	discount::Float64=0.9,
)
	return volumeLifeDetectionPOMDP(
		bn,
		λ,
		action_cpds,
		max_obs,
		inst,
		sample_volume,
		life_states,
		acc_rate,
		sample_use,
		discount,
	)
end

# 1 -> dead
# 2 -> alive
# 3 -> terminal state
POMDPs.states(pomdp::volumeLifeDetectionPOMDP) =
	1:(pomdp.sample_volume*pomdp.life_states+pomdp.life_states) #(pomdp.sample_volume*((2^pomdp.life_states)))+(2^pomdp.life_states)

# run sensor (2+i), where i the ith instrument, and the last instrument is doing nothing
# declare dead (1) declare alive (2)
POMDPs.actions(pomdp::volumeLifeDetectionPOMDP) =
	[1:pomdp.inst..., pomdp.inst+1, pomdp.inst+2]

# Extra observation at end which will be null observation
POMDPs.observations(pomdp::volumeLifeDetectionPOMDP) =
	0:(pomdp.max_obs*(pomdp.sample_volume+1)) # pomdp.sample_volume*pomdp.life_states+pomdp.life_states+1 

POMDPs.stateindex(pomdp::volumeLifeDetectionPOMDP, s::Int) = s
POMDPs.actionindex(pomdp::volumeLifeDetectionPOMDP, a::Int) = a
POMDPs.obsindex(pomdp::volumeLifeDetectionPOMDP, o::Int) = o+1


# TODO: do we want to start with different states?
POMDPs.initialstate(pomdp::volumeLifeDetectionPOMDP) = DiscreteUniform(1, 2) # 50% chance of being alive or dead with no starting sample    
# state_to_stateindex(0, 1) # TODO: change in future, so it starts at any state

function POMDPs.isterminal(pomdp::volumeLifeDetectionPOMDP, s::Int)
	sample_volume, life_state = stateindex_to_state(s, pomdp.life_states)
	if life_state == 3
		return true
	else
		return false
	end
end

POMDPs.discount(pomdp::volumeLifeDetectionPOMDP) = pomdp.discount


function POMDPs.transition(pomdp::volumeLifeDetectionPOMDP, s::Int, a::Int)

	# TODO: Need things for when action is to not turn on instrument
	# TODO: When previous action was using an instrument we are in melt mode 
	# Can't accumulate any more things at that point

	sample_volume, life_state = stateindex_to_state(s, pomdp.life_states)

	# if life_state == 3
	#     life_state = rand(1:2) # Set new life_state at new iteration
	#     sample_volume = 0#pomdp.acc_rate
	# end

	if a <= pomdp.inst

		# # if we choose to not use an instrument, thats the only time sample_volume goes up
		if a == pomdp.inst

			# Trying to make it so that waiting to use an instrument is
			# done multiple times
			# if sample_volume % pomdp.acc_rate != 1
			#     sample_volume = 1 #pomdp.acc_rate
			# end
			# life_state = rand(1:2)
			sample_volume += pomdp.acc_rate

			# Always make sure sample Volume can't exceed certain volume
			if sample_volume > pomdp.sample_volume
				sample_volume = pomdp.sample_volume
			end

			P_yes = bn.cpds[1].distributions[1].p[2]
			s1 = state_to_stateindex(sample_volume, 1)
			s2 = state_to_stateindex(sample_volume, 2)

			return SparseCat([s1, s2], [1 - P_yes, P_yes])

			# if we choose to use an instrument, we take away from sample_volume, assuming we are not choosing to wait for accumulation at this step
		else
			if sample_volume >= pomdp.sample_use[a]
				sample_volume = sample_volume - pomdp.sample_use[a]
			end

		end
		# Always make sure sample Volume can't exceed certain volume
		if sample_volume > pomdp.sample_volume
			sample_volume = pomdp.sample_volume
		end
		return Deterministic(state_to_stateindex(sample_volume, life_state)) # state (alive/dead) wont change while testing sample
	else
		return Deterministic(state_to_stateindex(sample_volume, 3)) # switch to terminal state only when we declare alive/dead
	end

end

function POMDPs.observation(pomdp::volumeLifeDetectionPOMDP, a::Int, sp::Int)

	sample_volume, life_state = stateindex_to_state(sp, pomdp.life_states)

	# if we already declared alive/dead, observation doesn't matter
	# if we declare alive/dead, observation doesn't matter
	# not choosing anything
	# TODO: return null obs if sample_volume == 0?
	if POMDPs.isterminal(pomdp, sp) ||
	   a == pomdp.inst + 1 ||
	   a == pomdp.inst + 2 ||
	   a == pomdp.inst #|| sample_volume == 0
		return Deterministic(0)
	end

	# # Sample first, then do infer to get posterior?

	# # Instrument Action to sample characteristics:
	# action_to_cpds = Dict(
	#     1 => [3, 5, 6, 10],   # HRMS >> Salinity (index 3), Path Complexity Index (5), CHNOPS (index 6), and redox (index 10)
	#     2 => [4, 7],       # SMS >> Chirality (index 4), Amino Acid Abundance (7)
	#     3 => [4, 7],       # μCE_LI >> Chirality (index 4), Amino Acid Abundance (7)
	#     4 => [3, 6],       # ESA >> Salinity (index 3), CHNOPS (index 6)
	#     5 => [8, 9],      # microscope >> Cell Membrane (8), autofluorescence (9)
	#     6 => [2],      # nanopore >> Polyelectrolyte (index 2)
	# )
	# # Get the specific sample indicies for action selected
	# cpd_indices = action_to_cpds[a]
	# evidence_dict = Dict()  # Start with life state as evidence

	# # For each CPD index, sample and add to evidence
	# for idx in cpd_indices
	#     posterior = infer(bn, bn.cpds[idx].target, evidence=Assignment(Dict(:l => life_state)))# Start with life state as evidence
	#     sample = rand(Categorical(convert(DataFrame, posterior)[!,"potential"]))
	#     evidence_dict[bn.cpds[idx].target] = sample
	# end

	# # All this is saying is: bn.cpds[1].target = :l
	# p_life = infer(bn, bn.cpds[1].target, evidence=Assignment(evidence_dict))

	# # Return a probability distribution over the new posterior of life (1 = no, 2 = yes)
	# return SparseCat([ob1, ob2], [p_life[1], p_life[2]])

	return SparseCat(
		obs_sample_volume(sample_volume, pomdp.max_obs),
		dist_observations(pomdp.action_cpds, life_state, a, pomdp.max_obs),
	)
end

function expected_belief_change(pomdp::volumeLifeDetectionPOMDP, a::Int)
	if a >= pomdp.inst  # declare alive/dead
		return 0.0
	end

	# Map action to biosignature node in Bayesian network
	# get probabilities from Bayesian network CPT
	factor = pomdp.bn.factors[idx]
	var_name = factor.vars[1].name

	# P(o|L) and P(o|!L)
	P_o_alive = factor.table[Dict(var_name => 2, :l => 2)]
	P_o_dead = factor.table[Dict(var_name => 2, :l => 1)]

	# compute evidence P(o) 
	P_o = P_o_alive * pomdp.b + P_o_dead * (1 - pomdp.b)

	# compute posterior P(L|o)
	P_L_o = (P_o_alive * pomdp.b) / P_o

	exp_change = abs(pomdp.b - P_L_o)
	#println("exp_change: ", exp_change)

	return exp_change
end


function POMDPs.reward(pomdp::volumeLifeDetectionPOMDP, s::Int, a::Int) #, b::Vector{Float64})
	if a == pomdp.inst + 1  # Declaring "no life"
		return s == 1 ? 0 : -pomdp.λ  # No reward if correct, penalty if wrong
	elseif a == pomdp.inst + 2  # Declaring "life exists"
		return s == 2 ? 0 : -pomdp.λ  # No reward if correct, penalty if wrong
	else  # Sensor action
		#prior_life = b[2]  # Probability of life from current belief state
		#exp_change = expected_belief_change(pomdp, a, prior_life)

		sample_volume, life_state = stateindex_to_state(s, pomdp.life_states)

		if sample_volume < pomdp.sample_use[a]
			return -10000
		end
		# more you're wasting the worse it is
		return -(1 - pomdp.λ)*(sample_volume/pomdp.sample_volume)
		# -(1 - pomdp.λ)*(sample_volume/pomdp.sample_volume)# + exp_change  # Cost of using sensor multipled by lambda factor plus expected information gain
	end
end

function state_to_stateindex(sample_volume::Int, life_states::Int)
	return (sample_volume-1)*3+life_states+3
end

function obs_sample_volume(sample_volume::Int, max_obs::Int)
	return (max_obs*(sample_volume)+1):(max_obs*(sample_volume)+max_obs)
end

function stateindex_to_state(index::Int, n_life_states::Int)
	if index%n_life_states != 0
		sample_volume = div(index, n_life_states)
		life_states = index%n_life_states
	else
		sample_volume = div(index, n_life_states)
		if div(index, n_life_states) != 0
			sample_volume -= 1
		end
		life_states = n_life_states
	end
	return sample_volume, life_states
end


function dist_observations(action_cpds, life_state, action, max_obs)
	"""
	Returns the probabilities of observations for a given action and life state.
	"""
	posterior = infer(bn, action_cpds[action], evidence=(Assignment(:C0 => life_state)))

	probs = Float64[]  # P(obs | l=1)

	# Determine domain sizes (bins per variable)
	domain_sizes = [
		length(bn.cpds[bn.name_to_index[v]].distributions[1].p) for
		v in posterior.dimensions
	]
	obs_indices = collect(CartesianIndices(Tuple(domain_sizes)))

	# -- Inference Loop --
	for idx in obs_indices
		push!(probs, posterior.potential[idx])
	end
	while length(probs) < max_obs
		push!(probs, 0)
	end
	probs ./= sum(probs)

	return probs
end


function determine_max_obs(action_cpds)
	"""
	Determines the maximum number of observations for a given action.
	"""
	max_obs = 0
	for action in collect(keys(action_cpds))
		for life_state in [1, 2]
			probs = Float64[]

			# P(obs | l=1)
			posterior =
				infer(bn, action_cpds[action], evidence=(Assignment(:C0 => life_state)))

			# determine domain sizes (bins per variable)
			domain_sizes = [
				length(bn.cpds[bn.name_to_index[v]].distributions[1].p) for
				v in posterior.dimensions
			]
			obs_indices = collect(CartesianIndices(Tuple(domain_sizes)))

			for i in obs_indices
				push!(probs, posterior.potential[i])
			end
			if length(probs) > max_obs
				max_obs = length(probs)
			end
		end
	end

	return max_obs
end


