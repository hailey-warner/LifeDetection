include("utils.jl")
include("bayesNet.jl")

using POMDPs
using POMDPTools

struct binaryLifeDetectionPOMDP <: POMDP{Int, Int, Int}  # POMDP{State, Action, Observation}
    inst::Int # number of instruments (child nodes)
    bn::BayesianNetwork # Bayesian Network
    discount::Float64
end

# Custom constructor to handle dynamic initialization
function binaryLifeDetectionPOMDP(;
    inst::Int, # number of instruments (child nodes)
    bn::BayesianNetwork, # Bayesian Network
    discount::Float64 = 0.9,
)
    return binaryLifeDetectionPOMDP(
        inst,bn,discount,
        )
end

# 1 -> dead
# 2 -> alive
# 3 -> terminal state
POMDPs.states(pomdp::binaryLifeDetectionPOMDP) = [1, 2, 3] 

# run sensor (2+i), where i the ith instrument
# declare dead (1) declare alive (2)
POMDPs.actions(pomdp::binaryLifeDetectionPOMDP) = [1, 2, 3:2+pomdp.inst...]

# observe biosignature is present (1) or absent (2)
POMDPs.observations(pomdp::binaryLifeDetectionPOMDP) = [1, 2] 


POMDPs.stateindex(pomdp::binaryLifeDetectionPOMDP, s::Int) = s
POMDPs.actionindex(pomdp::binaryLifeDetectionPOMDP,a::Int) = a
POMDPs.obsindex(pomdp::binaryLifeDetectionPOMDP, s::Int)   = s

POMDPs.initialstate(pomdp::binaryLifeDetectionPOMDP) = DiscreteUniform(1, 2) # 50% chance of being alive or dead
POMDPs.isterminal(pomdp::binaryLifeDetectionPOMDP, s::Int) = (s == 3)
POMDPs.discount(pomdp::binaryLifeDetectionPOMDP) = pomdp.discount


function POMDPs.transition(pomdp::binaryLifeDetectionPOMDP, s::Int, a::Int)
    if a > 2
        return Deterministic(s)  # state (alive/dead) wont change while testing sample
    else
        return Deterministic(3) # switch to terminal state only when we declare alive/dead
    end
end

function POMDPs.observation(pomdp::binaryLifeDetectionPOMDP, a::Int,  sp::Int)

    # if we already declared alive/dead, observation doesn't matter
    if POMDPs.isterminal(pomdp, sp)
        return SparseCat([1, 2], [0.5,0.5])
    end

    # if we declare alive/dead, observation doesn't matter
    if a == 1 || a == 2
        return SparseCat([1, 2], [0.5, 0.5])
    end

    # map action index to Bayesian network variable (action 3 → A, 4 → P, 5 → C)
    instrument_index = a - 2  # convert action number to variable index in BN
    if instrument_index > length(pomdp.bn.factors)
        error("Invalid instrument action: $a")
    end
    print(a)
    # get the corresponding conditional probability P(O=2|L=s), or P(biosignature present|alive/dead)
    factor = pomdp.bn.factors[instrument_index+1]
    var_name = pomdp.bn.factors[instrument_index+1].vars[1].name
    key = Dict(var_name => 2, :l => sp)  
    print(key)

    P_yes = factor.table[key] # Probability of "yes" observation for action a (a=2 means detected)
    
    # Return a probability distribution over the two observations (1 = no, 2 = yes)
    return SparseCat([1, 2], [1 - P_yes, P_yes])
end

function POMDPs.reward(pomdp::binaryLifeDetectionPOMDP, s::Int, a::Int)
    if a == 1  # Declaring "no life"
        return s == 1 ? 1.0 : -10.0  # Reward if correct, penalty if wrong
    elseif a == 2  # Declaring "life exists"
        return s == 2 ? 1.0 : -10.0  # Reward if correct, penalty if wrong
    else  # Any other action (sensor use)
        return -1  # Small penalty to discourage excessive sensing
    end
end
