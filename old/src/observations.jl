# the state space is the pomdp itself
POMDPs.observations(pomdp::LifeDetectionPOMDP) = pomdp #0:total_states(pomdp)-1

# cool way of getting state index, inspiration from Rocksample
function obsindex(pomdp::LifeDetectionPOMDP, s::LDState)

    # Step 1: Sample Certainty index
    sample_certainty_index = s.sample_certainty - 1

    # Step 2: Instrument health index (inst_health conversion)
    inst_health_index = 0
    for i = 1:pomdp.NumInst
        inst_health_index += (s.inst_health[i]-1) * pomdp.indices[i]
    end

    # Combine all components
    return sample_certainty_index + inst_health_index
end

function POMDPs.observation(pomdp::LifeDetectionPOMDP, a, sp)

    return Deterministic(sp)

end
