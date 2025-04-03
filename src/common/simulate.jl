function simulate_policy(pomdp, policy, type = "SARSOP", n_episodes=1)
    
    updater = DiscreteUpdater(pomdp)
    b = initialize_belief(updater, initialstate(pomdp))
    s = rand(initialstate(pomdp))
    
    println("\nPolicy Simulation:")
    println("Step | Action | Observation | Belief(Life)")
    println("-----------------------------------------")
    
    step = 1
    while !isterminal(pomdp, s) && step ≤ 10  # max 10 steps

        # get action, next state, and observation
        if type == "SARSOP"
            a = action(policy, b)
        elseif type == "greedy"
            a = action_greedy_policy(policy,b,step)
        end
        
        sp = rand(transition(pomdp, s, a))
        o = rand(observation(pomdp, a, sp))
        
        # format action and observation names
        action_name = a ≤ 2 ? (a == 1 ? "Declare Dead" : "Declare Life") : "Sensor $(a-2)"
        obs_name = o == 1 ? "Negative" : "Positive"
        
        # show step details
        @printf("%3d  | %-12s | %-11s | %.3f\n", 
                step, action_name, obs_name, pdf(b, 2))
        
        # update belief
        b = update(updater, b, a, o)
        s = sp
        step += 1
    end
end