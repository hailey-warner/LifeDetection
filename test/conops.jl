function conopsOrbiter(pomdp, o, mode_acc, prev_action)
    sample_volume, _ = stateindex_to_state(o, pomdp.life_states)

    if mode_acc
        return sum(pomdp.sample_use[1:5]) > sample_volume ? (7, true, 0) : (1, false, 1)
    end

    if prev_action < 5 && pomdp.sample_use[prev_action+1] < sample_volume
        return (prev_action + 1, false, prev_action + 1)
    end

    return (7, true, 0)
end
