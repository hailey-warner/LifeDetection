

function POMDPs.transition(pomdp::LifeDetectionPOMDP, s, a)

    # Whichever instrument we choose, there is a probability of getting a higher likely quality reading
    distribution_inst = Normal(pomdp.SampleTrueVal, pomdp.InstSigma[a])

    # Generate a quality reading
    new_sample_certainty = rand(distribution_inst)

    # update the quality reading
    s.sample_certainty = round(new_sample_certainty)

    # update the inst_health readings
    for i in 1:pomdp.NumInst
        s.inst_health[i] -= rand(1:pomdp.InstHealthMax)
    end

    return Deterministic(s)
end

