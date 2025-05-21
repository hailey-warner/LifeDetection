function simulate_policyVLD(pomdp, policy, type="SARSOP", n_episodes=1,verbose=true)
    
    if verbose
        println("--------------------------------START EPISODES---------------------------------")
    end

    total_episode_rewards = []
    accuracy = []

    for episode in range(1, n_episodes)

        updater = DiscreteUpdater(pomdp)
        b = initialize_belief(updater, initialstate(pomdp))
        s = rand(initialstate(pomdp))
        
        if verbose
            println("\nPolicy Simulation: Episode ",episode)
            println("Step | Action       | Observation | Belief(Life) | True State | Acc Sample | Total Reward ")
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
        belief_life = pdf(b,o_old)

        while !isterminal(pomdp, s) && step ≤ 200  # max 10 steps

            # get action, next state, and observation
            if type == "SARSOP"
                a = action(policy, b)
            elseif type == "greedy"
                a = action_greedy_policy(policy,b,step)
            elseif type == "conops"
                a, modeAcc, prevAction = conopsOrbiter(pomdp, s, modeAcc, prevAction)
            end
            
            sp = rand(transition(pomdp, s, a))
            o = rand(observation(pomdp, a, sp))
            if o != pomdp.sampleVolume*pomdp.lifeStates+pomdp.lifeStates+1
                _ , o_state = stateindex_to_state(o, pomdp.lifeStates)  # Save the current state before transitioning 
            end

            # Get reward and accumulate total reward
            r = reward(pomdp, s, a, sp)
            total_reward += r
            
            # format action and observation names
            action_name = a >= pomdp.inst+1 ? (a == pomdp.inst+1 ? "Declare Dead" : "Declare Life") : (a == pomdp.inst ? "Accumulate" : "Sensor $(a)")
            accu , true_state = stateindex_to_state(s, pomdp.lifeStates)  # Save the current state before transitioning 
            if o_old != pomdp.sampleVolume*pomdp.lifeStates+pomdp.lifeStates+1
                if true_state == 1 && step > 1
                    o_old += 1
                end
            
                belief_life = pdf(b,o_old)
            end
            if o != pomdp.sampleVolume*pomdp.lifeStates+pomdp.lifeStates+1
                obs_name = o_state == 1 ? "Negative" : "Positive"
                
            else
                obs_name = "No Sense"
                # belief_life = ""
            end
            
            # sum(pdf(b, state_to_stateindex(sample, 2)) for sample in 1:pomdp.sampleVolume)

            if verbose
                # show step details
                @printf("%3d  | %-12s | %-11s | %.3f        | %d          |  %d         | %.2f         \n", 
                        step, action_name, obs_name, belief_life, true_state, accu, total_reward)
            end

            # update belief
            # if a != pomdp.inst
            # if o != pomdp.sampleVolume*pomdp.lifeStates+pomdp.lifeStates+1
            b = update(updater, b, a, o)

            if o != pomdp.sampleVolume*pomdp.lifeStates+pomdp.lifeStates+1
                temp_o_old = o
            end
            o_old = o
            
            # end
            s = sp
            step += 1
        end
        acc = 1-abs(true_state-1-pdf(b,temp_o_old))
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
    edges = Tuple{Int,Int}[]
    
    updater = DiscreteUpdater(pomdp)
    b0 = initialize_belief(updater, initialstate(pomdp))
    
    function traverse(b, parent_index::Union{Int,Nothing}=nothing, edge_label::Union{String,Nothing}=nothing, depth=1)
        # Determine action from the policy
        a = action(policy, b)
        b_val = pdf(b, 1)
        #println("tree action: ", a)
        #println("tree belief: ", b_val)
        action_name = a ≤ 2 ? (a == 1 ? "Declare Dead" : "Declare Life") : "Sensor $(a - 2)"
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
            obs_name = o == 1 ? "N" : "Y"
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

function simulate_policy(pomdp, policy, type="SARSOP", n_episodes=1; verbose=true)
    total_episode_rewards = []
    accuracy = []

    if verbose
        println("--------------------------------START EPISODES---------------------------------")
    end

    for episode in range(1, n_episodes)

        if verbose
            println("\nPolicy Simulation: Episode ",episode)
            println("Step | Action       | Observation | Belief(Life) | True State | Total Reward ")
            println("-------------------------------------------------------------------------------")
        end

        updater = DiscreteUpdater(pomdp)
        b = initialize_belief(updater, initialstate(pomdp))
        s = rand(initialstate(pomdp))
        step = 1
        total_reward = 0
        true_state = 1

        while !isterminal(pomdp, s) && step ≤ 10  # max 10 steps

            # get action, next state, and observation
            if type == "SARSOP"
                a = action(policy, b)
            elseif type == "greedy"
                a = action_greedy_policy(policy,b,step)
            end
            
            sp = rand(transition(pomdp, s, a))
            o = rand(observation(pomdp, a, sp))

            # get reward and accumulate total reward
            r = reward(pomdp, s, a, sp)
            total_reward += r
            
            # format action and observation names
            action_name = a ≤ 2 ? (a == 1 ? "Declare Dead" : "Declare Life") : "Sensor $(a-2)"
            obs_name = o == 1 ? "Negative" : "Positive"
            true_state = s  # save the current state before transitioning

            if verbose
                # show step details
                @printf("%3d  | %-12s | %-11s | %.3f        | %d          | %.2f         \n", 
                        step, action_name, obs_name, pdf(b, 2), true_state, total_reward)
            end

            # update belief
            b = update(updater, b, a, o)
            s = sp
            step += 1
        end
        acc = 1-abs(true_state-1-pdf(b, 2))
        push!(total_episode_rewards, total_reward)
        push!(accuracy, acc)
    end

    if verbose
        println("--------------------------------END EPISODES---------------------------------")
        println("Average Rewards:", mean(total_episode_rewards))
    end

    return mean(total_episode_rewards), mean(accuracy)
end



function make_decision_tree(pomdp, policy; max_depth=3)
    node_labels = String[]
    edge_colors = String[]
    edges = Tuple{Int,Int}[]
    
    updater = DiscreteUpdater(pomdp)
    b0 = initialize_belief(updater, initialstate(pomdp))
    
    function traverse(b, parent_index::Union{Int,Nothing}=nothing, edge_color::Union{String,Nothing}=nothing, depth=1)
        a = action(policy, b)
        b_val = pdf(b, 2)
        action_name = a ≤ 2 ? (a == 1 ? "Dead" : "Alive") : "I$(a - 2)"
        label = "$(action_name), P(life)=$(round(b_val, digits=2))"
        push!(node_labels, label)
        current_index = length(node_labels)
        
        # record edge if a parent exists
        if parent_index !== nothing && edge_color !== nothing
            push!(edge_colors, edge_color)
            push!(edges, (parent_index, current_index))
        end
        
        # stop if we've reached max depth or if action is terminal
        if depth >= max_depth || a ≤ 2
            return
        end
        
        # branch for each observation
        for o in POMDPs.observations(pomdp)
            obs_name = o == 1 ? "red" : "green"
            b_new = update(updater, b, a, o)
            traverse(b_new, current_index, obs_name, depth+1)
        end
    end

    traverse(b0)
    edge_colors = Dict(zip(edges, edge_colors))
    return (node_labels, edge_colors, edges)
end