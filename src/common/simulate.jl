using ProgressMeter

function simulate_policyVLD(pomdp, policy, type="SARSOP", n_episodes=1, verbose=true, wandb=false, wandb_Name="")

	if verbose
		println("--------------------------------START EPISODES---------------------------------")
	end


	total_episode_rewards = zeros(Float64,n_episodes)
	accuracy = zeros(Float64,n_episodes)


	# @showprogress Threads.@threads 
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
		b = initialize_belief(updater, initialstateSample(pomdp, 0))
		s = initialstateSample(pomdp, 0)

		if verbose
			println("\nPolicy Simulation: Episode ", episode)
			# println("Step | Action       | Observation | Belief(Life) | True State | Acc Sample | Total Reward ")

			println("Step | Action        | Belief(Life) | True State | Acc Sample | Total Reward ")
			println("-------------------------------------------------------------------------------")
		end

		# Trackers
		step = 1
		true_state = 1 # track s_L (life state)
		a = 0
		s = 1
		sp = 1
		belief_life = 0.0
		action_name = ""
		action_final = 0
		# belief_life = pdf(b, 2) # setting belief for life with no sample volume at first

		# metrics
		total_reward = 0.0
		acc = 0
		correct = Dict(
			"tt" => 0,  # true positive: declared life when life is true
			"tf" => 0,  # false positive: declared life when life is false
			"ft" => 0,  # false negative: declared dead when life is true
			"ff" => 0   # true negative: declared dead when life is false
		)

		# only for conops:
		modeAcc = true
		prevAction = 0

		while step <= 200 # always 200 steps  ### !isterminal(pomdp, s) &&

			if isterminal(pomdp, s)

				if action_final == 2 && true_state == 2
					correct["tt"] += 1  # true positive
				elseif action_final == 1 && true_state == 2
					correct["tf"] += 1  # false positive
				elseif action_final == 2 && true_state == 1
					correct["ft"] += 1  # false negative
				elseif action_final == 1 && true_state == 1
					correct["ff"] += 1  # true negative
				end

				# Get the current sample volume from state index
				sample_volume, _ = stateindex_to_state(s, pomdp.life_states)

				# Sample new life state while keeping sample_volume fixed
				s = rand(initialstateSample(pomdp, sample_volume))

				# Reinitialize belief for same sample_volume
				b = initialize_belief(updater, initialstateSample(pomdp, sample_volume))

                if verbose
                    println("[Resetting] Reached terminal state. Sampling new initial state.")
                end
			end

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

			# Get belief in life at this state
			s_check = s
			if o != 0
				if true_state == 1 && step > 1
					s_check = s_check + 1
				end
				belief_life = pdf(b, s_check)
			end

			if verbose
				@printf("%3d  | %-12s | %.3f        | %d          |  %d         | %.2f         \n",
					step, action_name, belief_life, true_state, accu, total_reward)
			end
			if wandb

				###################################
				Wandb.log(
					run,
					Dict(
						"step" => step,
						"action" => a,
						"beliefLife" => belief_life,
						"trueState" => true_state,
						"accu" => acc,
						"totalReward" => total_reward,
						"observation" => o,
						"state" => s,
						"nextState" => sp,
						"belief" => b,
					),
				)

			end

			b = update(updater, b, a, o)

			# end
			s = sp
			step += 1
			action_final = a-pomdp.inst

			if true_state == 2 && a <=7
				correct["tf"] += 1  # false positive
			end
		end

		println(correct)

		# Calculate accuracy using the correct dictionary
		total = correct["tt"] + correct["tf"] + correct["ft"] + correct["ff"]
		if total > 0
			acc = (correct["tt"] + correct["ff"]) / total
		else
			acc = 0.0
		end
		
		total_episode_rewards[episode] = total_reward
		accuracy[episode] = acc

		if wandb
			Wandb.wandb.summary["tt"] = correct["tt"]
			Wandb.wandb.summary["ft"] = correct["ft"]
			Wandb.wandb.summary["tf"] = correct["tf"]
			Wandb.wandb.summary["ff"] = correct["ff"]
			Wandb.wandb.summary["total_reward_final"] = total_reward
			# Wandb.wandb.summary["action_final"] = a
			# Wandb.wandb.summary["action_final_name"] = action_name
			# Wandb.wandb.summary["belief_final"] = belief_life
			# Wandb.wandb.summary["s_final"] = true_state
			# Wandb.wandb.summary["sp_final"] = sp
			# Wandb.wandb.summary["false_positive_rate"] = (action_final == 1 ? 1 : 0)
			# Wandb.wandb.summary["false_negative_rate"] = (action_final == 2 ? 1 : 0)
			close(run)
			sleep(0.3)
		end

	end

	if verbose
		println("--------------------------------END EPISODES---------------------------------")
		println("Average Rewards:", mean(total_episode_rewards))
		println("Average Accuracy:", mean(accuracy))
		println(accuracy)
	end

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


