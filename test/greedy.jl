function action_greedy_policy(policy, b, step)
    # Only two steps in greedy policy
    # 1. we use the sensor once, (or whatever the initial policy chooses)
    # 2. if the simulation isn't terminated, we declare whether or not it's alive or dead.

    if step == 1
        return action(policy, b)
    else
        return pdf(b, 2) ≤ 0.5 ? 1 : 2
    end
end
