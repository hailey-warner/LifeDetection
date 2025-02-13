function POMDPs.reward(pomdp::LifeDetectionPOMDP, s, a)

    reward = 0.0

    # Negative penalties for lowered instrument health
    for health in s.inst_health
        reward -= pomdp.InstHealthMax-health
    end
    
    reward += pomdp.SampleTrueVal - s.sample_certainty
    
    return reward
end