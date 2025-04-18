function simulate_policy(pomdp, policy, type = "SARSOP", n_episodes=1,verbose = true)
    

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
            println("Step | Action       | Observation | Belief(Life) | True State | Total Reward ")
            println("-------------------------------------------------------------------------------")
        end
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

            # Get reward and accumulate total reward
            r = reward(pomdp, s, a, sp)
            total_reward += r
            
            # format action and observation names
            action_name = a ≤ 2 ? (a == 1 ? "Declare Dead" : "Declare Life") : "Sensor $(a-2)"
            obs_name = o == 1 ? "Negative" : "Positive"
            true_state = s  # Save the current state before transitioning

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
        push!(total_episode_rewards,total_reward)
        acc = 1-abs(true_state-1-pdf(b, 2))
        print(acc)
        push!(accuracy,acc)
    end

    println("--------------------------------END EPISODES---------------------------------")

    println("Average Rewards:", mean(total_episode_rewards))
    return mean(total_episode_rewards), mean(accuracy)
end