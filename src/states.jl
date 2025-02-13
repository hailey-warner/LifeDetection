

function total_states(pomdp::LifeDetectionPOMDP)
    return pomdp.SampleCertaintyMax * pomdp.InstHealthMax^pomdp.NumInst
end

# the state space is the pomdp itself
POMDPs.states(pomdp::LifeDetectionPOMDP) = pomdp #0:total_states(pomdp)-1

# need iterator because states not defined perfectly
function Base.iterate(pomdp::LifeDetectionPOMDP, i::Int=1)
    # do we need this?
    if i > length(pomdp)
        return nothing
    end
    s = state_from_index(pomdp, i)
    return (s, i+1)
end

# need length because states not defined perfectly
Base.length(pomdp::LifeDetectionPOMDP) = total_states(pomdp)

# cool way of getting state index, inspiration from Rocksample
function stateindex(pomdp::LifeDetectionPOMDP, s::LDState)

    # Step 1: Sample Certainty index
    sample_certainty_index = s.sample_certainty - 1

    # Step 2: Instrument health index (inst_health conversion)
    inst_health_index = 0
    for i in 1:pomdp.NumInst
        inst_health_index += (s.inst_health[i]-1) * pomdp.indices[i]
    end

    # Combine all components
    return sample_certainty_index + inst_health_index
end

function state_from_index(pomdp::LifeDetectionPOMDP, index::Int)

    sample_certainty = (index % pomdp.SampleCertaintyMax) + 1
    inst_health = Vector{Int}(zeros(Int, pomdp.NumInst))

    for i in pomdp.NumInst:-1:1
        inst_health[i] = (index ÷ pomdp.indices[i]) % pomdp.InstHealthMax + 1
        index = index % pomdp.indices[i]
    end

    return LDState(sample_certainty, inst_health)
end


function POMDPs.initialstate(pomdp::LifeDetectionPOMDP)

    # Define the number of values
    num_states = total_states(pomdp) - 1 

    # Define the states (the specific values the distribution can take)
    states = 0:num_states  # This can be any set of values

    # Define equal probabilities
    probs = fill(1/num_states, num_states)  
    
    # Return a SparseCat distribution over the initial states
    return  SparseCat(states, probs)
end


