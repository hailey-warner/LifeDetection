
using POMDPs
using POMDPTools

struct volumeLifeDetectionPOMDP <: POMDP{Int, Int, Int}  # POMDP{State, Action, Observation}
    bn::BayesianNetwork # Bayesian Network,
    λ::Float64    
    inst::Int64 # number of instruments (child nodes)
    sampleVolume::Int64
    lifeStates::Int64
    surfaceAccRate::Int64
    sampleUse::Vector{Int64} 
    # k::Vector{Float64} # cost of observations
    discount::Float64
end


# Custom constructor to handle dynamic initialization
function volumeLifeDetectionPOMDP(;
    bn::BayesianNetwork, # Bayesian Network,
    λ::Float64,    
    inst::Int64 = 7, # number of instruments / not using instrument
    sampleVolume::Int64 = 500,
    lifeStates::Int64 = 3,
    surfaceAccRate::Int64 = 270,
    sampleUse::Vector{Int64} = [1,1,1,1,1,1,1], # cost of observations
    # k::Vector{Float64} = [HRMS*10e6, SMS*10e6, μCE_LI*10e6, ESA*10e6, microscope*10e6, nanopore*10e6], # cost of observations
    discount::Float64 = 0.9,
)
    return volumeLifeDetectionPOMDP(bn,λ,inst,sampleVolume,lifeStates,surfaceAccRate,sampleUse,discount) #(inst,bn,λ,k,discount)
end

# 1 -> dead
# 2 -> alive
# 3 -> terminal state
POMDPs.states(pomdp::volumeLifeDetectionPOMDP) =  1:(pomdp.sampleVolume*((2^pomdp.lifeStates)))+(2^pomdp.lifeStates)

# run sensor (2+i), where i the ith instrument
# declare dead (1) declare alive (2)
POMDPs.actions(pomdp::volumeLifeDetectionPOMDP) = [1, 2, 3:2+pomdp.inst...]

# observe biosignature is present (1) or absent (2)
POMDPs.observations(pomdp::volumeLifeDetectionPOMDP) = 1:(pomdp.sampleVolume*((2^pomdp.lifeStates)))+(2^pomdp.lifeStates)

POMDPs.stateindex(pomdp::volumeLifeDetectionPOMDP, s::Int) = s
POMDPs.actionindex(pomdp::volumeLifeDetectionPOMDP,a::Int) = a
POMDPs.obsindex(pomdp::volumeLifeDetectionPOMDP, s::Int)   = s


# TODO: do we want to start with uniform prior (50% alive/dead)?
POMDPs.initialstate(pomdp::volumeLifeDetectionPOMDP) = DiscreteUniform(1, 2) # 50% chance of being alive or dead with no starting sample    
                                                                             # state_to_stateindex(0, 1) # TODO: change in future, so it starts at any state

function POMDPs.isterminal(pomdp::volumeLifeDetectionPOMDP, s::Int) 
    sampleVolume, lifeState = stateindex_to_state(s, pomdp.lifeStates)
    if (lifeState == 3)
        return true
    else
        return false
    end
end

POMDPs.discount(pomdp::volumeLifeDetectionPOMDP) = pomdp.discount

function POMDPs.transition(pomdp::volumeLifeDetectionPOMDP, s::Int, a::Int)

    sampleVolume, lifeState = stateindex_to_state(s, pomdp.lifeStates)
    
    if a > 2

        # if we choose to not use an instrument, thats the only time sampleVolume goes up
        if a-2 == pomdp.inst
            sampleVolume += pomdp.surfaceAccRate

        # if we choose to use an instrument, we take away from sampleVolume, assuming we are not choosing to wait for accumulation at this step
        elseif sampleVolume > pomdp.sampleUse[a-2]
            sampleVolume = sampleVolume - pomdp.sampleUse[a-2]
        end

        # Always make sure sample Volume is larger than certain Volume
        if sampleVolume > pomdp.sampleVolume
            sampleVolume = pomdp.sampleVolume
        end
        return Deterministic(state_to_stateindex(sampleVolume, lifeState)) # state (alive/dead) wont change while testing sample
    else
        return Deterministic(state_to_stateindex(sampleVolume, 3)) # switch to terminal state only when we declare alive/dead
    end

end

function POMDPs.observation(pomdp::volumeLifeDetectionPOMDP, a::Int,  sp::Int)

    sampleVolume, lifeState = stateindex_to_state(sp, pomdp.lifeStates)
    ob1 = state_to_stateindex(sampleVolume, 1)
    ob2 = state_to_stateindex(sampleVolume, 2)

    # if we already declared alive/dead, observation doesn't matter
    if POMDPs.isterminal(pomdp, sp)
        return SparseCat([ob1, ob2], [0.5,0.5])
    end

    # if we declare alive/dead, observation doesn't matter
    if a == 1 || a == 2
        return SparseCat([ob1, ob2], [0.5, 0.5])
    end
    
    # not choosing anything
    if a == pomdp.inst+2
        return SparseCat([ob1, ob2], [0.5, 0.5])
    end

    # map action index to Bayesian network variable (action 3 → A, 4 → P, 5 → C)
    instrument_index = a - 2  # convert action number to variable index in BN
    if instrument_index > length(pomdp.bn.factors)
        error("Invalid instrument action: $a")
    end
    #print(a)
    # get the corresponding conditional probability P(O|L), or P(biosignature present|alive/dead)
    factor = pomdp.bn.factors[instrument_index+1]
    var_name = pomdp.bn.factors[instrument_index+1].vars[1].name
    key = Dict(var_name => 2, :l => lifeState) #sp)  
    #print(key)

    P_yes = factor.table[key] # Probability of "yes" observation for action a (a=2 means detected)
    
    # Return a probability distribution over the two observations (1 = no, 2 = yes)
    return SparseCat([ob1, ob2], [1 - P_yes, P_yes])
end

function expected_belief_change(pomdp::volumeLifeDetectionPOMDP, a::Int, prior_life::Float64)
    if a <= 2  # Not a sensor action
        return 0.0
    end
    
    # Get sensor probabilities from Bayesian network
    instrument_index = a - 2
    factor = pomdp.bn.factors[instrument_index+1]
    var_name = pomdp.bn.factors[instrument_index+1].vars[1].name
    
    # P(sensor|life) and P(sensor|no life)
    P_sensor_given_life = factor.table[Dict(var_name => 2, :l => 2)]
    P_sensor_given_nolife = factor.table[Dict(var_name => 2, :l => 1)]
    
    # Calculate P(sensor) using law of total probability
    P_sensor = P_sensor_given_life * prior_life + P_sensor_given_nolife * (1 - prior_life)
    
    # Calculate P(life|sensor) using Bayes rule
    P_life_given_sensor = (P_sensor_given_life * prior_life) / P_sensor
    
    # Expected change in belief is weighted average of belief changes
    exp_change = P_sensor * abs(P_life_given_sensor - prior_life) + 
                 (1 - P_sensor) * abs((prior_life - P_sensor_given_life * prior_life)/(1 - P_sensor) - prior_life)
    
    return exp_change
end

function POMDPs.reward(pomdp::volumeLifeDetectionPOMDP, s::Int, a::Int) #, b::Vector{Float64})
    if a == 1  # Declaring "no life"
        return s == 1 ? 0 : -pomdp.λ  # No reward if correct, penalty if wrong
    elseif a == 2  # Declaring "life exists"
        return s == 2 ? 0 : -pomdp.λ  # No reward if correct, penalty if wrong
    else  # Sensor action
        #prior_life = b[2]  # Probability of life from current belief state
        #exp_change = expected_belief_change(pomdp, a, prior_life)

        sampleVolume, lifeState = stateindex_to_state(s, pomdp.lifeStates)

        if sampleVolume < pomdp.sampleUse[a-2]
            return -10000
        end
        return -(1 - pomdp.λ)*sampleVolume# + exp_change  # Cost of using sensor multipled by lambda factor plus expected information gain
    end
end

function state_to_stateindex(sampleVolume::Int, lifeStates::Int)
    return (sampleVolume-1)*3+lifeStates
end
function stateindex_to_state(index::Int, n_lifeStates::Int)
    if index%n_lifeStates != 0
        sampleVolume = div(index,n_lifeStates) + 1
        lifeStates = index%n_lifeStates
    else
        sampleVolume = div(index,n_lifeStates)
        lifeStates = n_lifeStates
    end
    return sampleVolume, lifeStates
end
