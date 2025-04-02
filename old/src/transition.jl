

function POMDPs.transition(pomdp::LifeDetectionPOMDP, s, a)


    # Define normal distribution
    normal_dist = Normal(pomdp.SampleTrueVal, pomdp.InstSigma[a])

    # Define truncated distribution
    truncated_dist = Truncated(normal_dist, 1, pomdp.SampleCertaintyMax)

    # Sample from the truncated distribution
    sampled_value = rand(truncated_dist)

    # print(sampled_value)
    # update the quality reading
    s.sample_certainty = round(sampled_value)
    # print(s.sample_certainty)
    # print("HERE!")

    # print(s.sample_certainty)
    # update the inst_health readings
    s.inst_health[a] -= 1 #rand(1:pomdp.InstHealthMax)
    if s.inst_health[a] < 1
        s.inst_health[a] = 1
    end
    # for i in 1:pomdp.NumInst
    #     s.inst_health[i] -= rand(1:pomdp.InstHealthMax)
    #     if s.inst_health[i] < 1
    #         s.inst_health[i] = 1
    #     end
    # end

    print(s.inst_health)
    return Deterministic(s)
end

