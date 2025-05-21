function conopsOrbiter(pomdp, o, mode_acc, prev_action)
    sampleVolume, _ = stateindex_to_state(o, pomdp.lifeStates)
    
    if mode_acc
        return sum(pomdp.sampleUse[1:5]) > sampleVolume ? (7, true, 0) : (1, false, 1)
    end

    if prev_action < 5 && pomdp.sampleUse[prev_action+1] < sampleVolume
        return (prev_action + 1, false, prev_action + 1)
    end

    return (7, true, 0)
end