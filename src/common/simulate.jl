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