include("utils.jl")
include("bayesNet.jl")


using POMDPs: POMDP
import POMDPs
# using Distributions
using POMDPTools

struct binaryLifeDetectionPOMDP <: POMDP{Int, Int, Int}  # POMDP{State, Action, Observation}
    inst::Int # number of instruments (child nodes)
    bn::BayesianNetwork # Bayesian Network

    discount::Float64
end


# # Custom constructor to handle dynamic initialization
function binaryLifeDetectionPOMDP(;
    inst::Int, # number of instruments (child nodes)
    bn::BayesianNetwork, # Bayesian Network
    discount::Float64 = 0.9,
)
    return binaryLifeDetectionPOMDP(
        inst,bn,discount,
        )
end



# Life > 2, No life 1, 3 is end terminal state
POMDPs.states(pomdp::binaryLifeDetectionPOMDP) = [1, 2, 3] 

# can choose observations/sensors (3-3+n), n sensors
# can choose declare state not active alive (1) alive (2)
POMDPs.actions(pomdp::binaryLifeDetectionPOMDP) = [1, 2, 3:2+pomdp.inst...]

# We only observe yes or no from observation?
# TODO: Check with Mykel if this is right
POMDPs.observations(pomdp::binaryLifeDetectionPOMDP) = [1, 2] 


POMDPs.stateindex(pomdp::binaryLifeDetectionPOMDP, s::Int) = s
POMDPs.actionindex(pomdp::binaryLifeDetectionPOMDP,a::Int) = a
POMDPs.obsindex(pomdp::binaryLifeDetectionPOMDP, s::Int) = s

POMDPs.initialstate(pomdp::binaryLifeDetectionPOMDP) = DiscreteUniform(1, 2) # gives back a distribution between 1,2

POMDPs.isterminal(pomdp::binaryLifeDetectionPOMDP, s::Int) = (s == 3)
POMDPs.discount(pomdp::binaryLifeDetectionPOMDP) = pomdp.discount


function POMDPs.transition(pomdp::binaryLifeDetectionPOMDP, s::Int, a::Int)
    if a > 2
        return Deterministic(s)  # State of the sample life will never change
    else
        return Deterministic(3) # only when we take terminal action we switch to terminal action state
    end
end

function POMDPs.observation(pomdp::binaryLifeDetectionPOMDP, a::Int,  sp::Int)

    # if we declare we don't need observation anymore
    if POMDPs.isterminal(pomdp, sp)
        return SparseCat([1, 2], [0.5,0.5])
    end

    # If we declare, we don’t need observation anymore
    if a == 1 || a == 2
        return SparseCat([1, 2], [0.5, 0.5])
    end

    # Map action index to Bayesian network variable (action 3 → A, 4 → P, 5 → C)
    instrument_index = a - 2  # Convert action number to variable index in BN
    if instrument_index > length(pomdp.bn.factors)
        error("Invalid instrument action: $a")
    end
    print(a)
    # Get the corresponding conditional probability P(O=2 | L=s)
    factor = pomdp.bn.factors[instrument_index+1]  # Factor corresponding to chosen instrument
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
