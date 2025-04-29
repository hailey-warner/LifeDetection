using POMDPs
using POMDPTools

struct binaryLifeDetectionPOMDP <: POMDP{Int, Int, Int}  # POMDP{State, Action, Observation}
    inst::Int # number of instruments
    bn::BayesianNetwork # Bayesian network
    k::Vector{Float64} # cost of observations
    λ::Int  # cost of declaring alive/dead
    b::Float64 # belief state (probability of life)
    discount::Float64
end

# custom constructor to handle dynamic initialization
function binaryLifeDetectionPOMDP(;
    inst::Int, 
    bn::BayesianNetwork, 
    k::Vector{Float64} = [0.1, 0.8, 0.6, 0.2],
    λ::Int = 5,
    b::Float64 = 0.5,
    discount::Float64 = 0.9,
)
    return binaryLifeDetectionPOMDP(inst,bn,k,λ,b,discount)
end

# dead (1)
# alive (2)
# terminal state (3)
POMDPs.states(pomdp::binaryLifeDetectionPOMDP) = [1, 2, 3]

# prior (uniform)
POMDPs.initialstate(pomdp::binaryLifeDetectionPOMDP) = DiscreteUniform(1, 2)
POMDPs.isterminal(pomdp::binaryLifeDetectionPOMDP, s::Int) = (s == 3)

# run sensor i (2+i)
# declare dead (1) declare alive (2)
POMDPs.actions(pomdp::binaryLifeDetectionPOMDP) = [1, 2, 3:2+pomdp.inst...]

# observe biosignature is present (1) or absent (2)
POMDPs.observations(pomdp::binaryLifeDetectionPOMDP) = [1, 2] 

POMDPs.stateindex(pomdp::binaryLifeDetectionPOMDP, s::Int) = s
POMDPs.actionindex(pomdp::binaryLifeDetectionPOMDP,a::Int) = a
POMDPs.obsindex(pomdp::binaryLifeDetectionPOMDP, s::Int)   = s

POMDPs.discount(pomdp::binaryLifeDetectionPOMDP) = pomdp.discount

function POMDPs.transition(pomdp::binaryLifeDetectionPOMDP, s::Int, a::Int)
    if a > 2
        return Deterministic(s)  # state (alive/dead) wont change while testing sample
    else
        return Deterministic(3) # switch to terminal state once we declare alive/dead
    end
end

function POMDPs.observation(pomdp::binaryLifeDetectionPOMDP, a::Int,  sp::Int)

    # if we declare alive/dead, observation doesn't matter
    if POMDPs.isterminal(pomdp, sp) || a <= 2
        return SparseCat([1, 2], [0.5, 0.5])
    end

    # map action index to Bayesian network variable (action 3 → A, 4 → P, 5 → C, 6 → H)
    instrument_index = a - 2  # convert action number to variable index in BN
    if instrument_index > length(pomdp.bn.factors)
        error("Invalid instrument action: $a")
    end

    # get conditional probability P(child|parent)
    factor = pomdp.bn.factors[instrument_index+1]
    var_name = factor.vars[1].name
    
    # get parent nodes
    parent_vars = factor.vars[2:end]  # First var is child, rest are parents
    
    # Build key with all parent assignments
    key = Dict(var_name => 2)  # Start with child variable
    for parent_var in parent_vars
        if parent_var.name == :l
            key[parent_var.name] = sp  # Life state from POMDP state
        elseif parent_var.name == :a
            # Handle biosignature state - you'll need to track this
            key[parent_var.name] = 1  # Or however you track biosignature state
        end
        # Add more parent variable handlers as needed
    end

    P_yes = factor.table[key] # Probability of "yes" observation for action a (a=2 means detected)
    
    # Return a probability distribution over the two observations (1 = no, 2 = yes)
    return SparseCat([1, 2], [1 - P_yes, P_yes])
end

function expected_belief_change(pomdp::binaryLifeDetectionPOMDP, a::Int)
    if a <= 2  # declare alive/dead
        return 0.0
    end
    
    # map action to biosignature node in Bayesian network
    idx = a - 2
    if idx > length(pomdp.bn.factors)
        error("Invalid instrument action: $a")
    end
    
    # get probabilities from Bayesian network CPT
    factor = pomdp.bn.factors[idx]
    var_name = factor.vars[1].name
    
    # P(o|L) and P(o|!L)
    P_o_alive = factor.table[Dict(var_name => 2, :l => 2)]
    P_o_dead = factor.table[Dict(var_name => 2, :l => 1)]
    
    # compute evidence P(o) 
    P_o = P_o_alive * pomdp.b + P_o_dead * (1 - pomdp.b)
    
    # compute posterior P(L|o)
    P_L_o = (P_o_alive * pomdp.b) / P_o 

    exp_change = abs(pomdp.b - P_L_o)
    #println("exp_change: ", exp_change)
    
    return exp_change
end

function POMDPs.reward(pomdp::binaryLifeDetectionPOMDP, s::Int, a::Int)
    if a == 1  # Declaring "no life"
        return s == 1 ? 1.0 : -pomdp.λ  # Reward if correct, penalty if wrong
    elseif a == 2  # Declaring "life exists"
        return s == 2 ? 1.0 : -pomdp.λ  # Reward if correct, penalty if wrong
    else  # Sensor action
        exp_change = expected_belief_change(pomdp, a)
        return -pomdp.k[a-2] + exp_change  # Cost of using sensor plus expected information gain
    end
end


