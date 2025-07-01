function conops_orbiter(pomdp, s, mode_acc, prev_action, belief_life, threshold_high, threshold_low)
    # Constants
    declare_dead_action = pomdp.inst + 1  # 8
    declare_life_action = pomdp.inst + 2  # 9
    accumulate_action = pomdp.inst        # 7

    # Get current sample volume
    sample_volume, life_state = stateindex_to_state(s, pomdp.life_states)

    # Declare if confident enough
    if belief_life >= threshold_high
        return (declare_life_action, true, 0)
    elseif belief_life <= threshold_low
        return (declare_dead_action, true, 0)
    end

	if mode_acc
        return sum(pomdp.sample_use[1:5]) > sample_volume ? (7, true, 0) : (1, false, 1)
    end

    if prev_action < 5 && pomdp.sample_use[prev_action+1] < sample_volume
        return (prev_action + 1, false, prev_action + 1)
    end

    return (7, true, 0)
end