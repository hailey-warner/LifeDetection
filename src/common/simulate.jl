function simulate_policyVLD(pomdp, policy, type="SARSOP", n_episodes=1, verbose=true, wandb=false, wandb_Name="")

	if verbose
		println("--------------------------------START EPISODES---------------------------------")
	end


	total_episode_rewards = []
	accuracy = []

	for episode in range(1, n_episodes)

		if wandb
			run = WandbLogger( #Wandb.wandb.init(
				# Set the wandb entity where your project will be logged (generally your team name).
				entity="sherpa-rpa",
				# Set the wandb project where this run will be logged.
				project=wandb_Name,
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

		updater = DiscreteUpdater(pomdp)
		b = initialize_belief(updater, initialstate(pomdp))
		s = rand(initialstate(pomdp))

		if verbose
			println("\nPolicy Simulation: Episode ", episode)
			# println("Step | Action       | Observation | Belief(Life) | True State | Acc Sample | Total Reward ")

			println("Step | Action        | Belief(Life) | True State | Acc Sample | Total Reward ")
			println("-------------------------------------------------------------------------------")
		end
		step = 1
		total_reward = 0
		true_state = 1
		acc = 0
		o_old = 2
		temp_o_old = 0
		modeAcc = true
		prevAction = 0
		belief_life = pdf(b, o_old)
		action_final = 0
		a = 0
		action_name = ""
		belief_life = 0.0
		s = 1
		sp = 1
		total_reward = 0.0

		while !isterminal(pomdp, s) && step ≤ 200  # max 10 steps

			# get action, next state, and observation
			if type == "SARSOP"
				a = action(policy, b)
			elseif type == "greedy"
				a = action_greedy_policy(policy, b, step)
			elseif type == "conops"
				a, modeAcc, prevAction = conops_orbiter(pomdp, s, modeAcc, prevAction)
			end

			sp = rand(transition(pomdp, s, a))
			o = rand(observation(pomdp, a, sp))

			# Get reward and accumulate total reward
			r = reward(pomdp, s, a, sp)
			total_reward += r

			# format action and observation names
			action_name = a >= pomdp.inst+1 ? (a == pomdp.inst+1 ? "Declare Dead" : "Declare Life") : (a == pomdp.inst ? "Accumulate" : "Sensor $(a)")
			accu, true_state = stateindex_to_state(s, pomdp.life_states)  # Save the current state before transitioning 
			println("Obs: ", o)

			s_check = s
			if o != 0
				if true_state == 1 && step > 1
					s_check = s_check + 1
				end

				belief_life = pdf(b, s_check)

			end
			# if o != 0
			#     obs_name = o_state == 1 ? "Negative" : "Positive"

			# else
			#     obs_name = "No Sense"
			#     # belief_life = ""
			# end

			# sum(pdf(b, state_to_stateindex(sample, 2)) for sample in 1:pomdp.sample_volume)

			if verbose
				# show step details
				# @printf("%3d  | %-12s | %-11s | %.3f        | %d          |  %d         | %.2f         \n", 
				#         step, action_name, obs_name, belief_life, true_state, accu, total_reward)

				@printf("%3d  | %-12s | %.3f        | %d          |  %d         | %.2f         \n",
					step, action_name, belief_life, true_state, accu, total_reward)
			end
			if wandb
				###################################
				Wandb.log(
					run,
					Dict(
						"Simulation/step" => step,
						"Simulation/actionName" => action_name,
						"Simulation/beliefLife" => belief_life,
						"Simulation/trueState" => true_state,
						"Simulation/accu" => acc,
						"Simulation/totalReward" => total_reward,
						"Simulation/observation" => o,
						"Simulation/state" => s,
						"Simulation/nextState" => sp,
						"Simulation/belief" => b,
					),
				)

			end

			# update belief
			# if a != pomdp.inst
			# if o != pomdp.sample_volume*pomdp.life_states+pomdp.life_states+1
			b = update(updater, b, a, o)

			# end
			s = sp
			step += 1
			action_final = a-pomdp.inst
		end

		if wandb
			Wandb.wandb.summary["action_final"] = a
			Wandb.wandb.summary["action_final_name"] = action_name
			Wandb.wandb.summary["belief_final"] = belief_life
			Wandb.wandb.summary["s_final"] = true_state
			Wandb.wandb.summary["sp_final"] = sp
			Wandb.wandb.summary["total_reward_final"] = total_reward
			Wandb.wandb.summary["accuracy_final"] = (action_final == true_state ? 1 : 0)

			close(run)
			sleep(0.3)
		end

		println(action_final)
		acc = (action_final == true_state ? 1 : 0)
		push!(total_episode_rewards, total_reward)
		push!(accuracy, acc)
	end

	println("--------------------------------END EPISODES---------------------------------")
	println("Average Rewards:", mean(total_episode_rewards))

	return mean(total_episode_rewards), mean(accuracy)
end

function decision_tree(pomdp, policy; max_depth=3)
	node_labels = String[]
	edge_labels = String[]
	edges = Tuple{Int, Int}[]

	updater = DiscreteUpdater(pomdp)
	b0 = initialize_belief(updater, initialstate(pomdp))

	function traverse(b, parent_index::Union{Int, Nothing}=nothing, edge_label::Union{String, Nothing}=nothing, depth=1)
		# Determine action from the policy
		a = action(policy, b)
		b_val = pdf(b, 1)
		#println("tree action: ", a)
		#println("tree belief: ", b_val)
		action_name = pomdp.inst < a ? (a == 8 ? "Declare Dead" : "Declare Life") : (pomdp.inst == a ? "Accumulate" : "Sensor $(a - 2)")
		label = "Action: $(action_name)\nBelief: $(b_val)"
		push!(node_labels, label)
		current_index = length(node_labels)

		# Record the edge if a parent exists.
		if parent_index !== nothing && edge_label !== nothing
			push!(edge_labels, edge_label)
			push!(edges, (parent_index, current_index))
		end

		# Stop if we've reached max depth or if the action is terminal.
		if depth >= max_depth || a ≤ 2
			return
		end

		# Branch for each observation
		for o in POMDPs.observations(pomdp)
			obs_name = "$(o)" # = o == 1 ? "N" : "Y"
			println(o)
			b_new = update(updater, b, a, o)
			traverse(b_new, current_index, obs_name, depth+1)
		end
	end

	traverse(b0)
	#println("Node Labels: ", node_labels)
	#println("Edge Labels: ", edge_labels)
	#println("Edges: ", edges)
	return (node_labels, edge_labels, edges)
end


