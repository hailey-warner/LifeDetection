
struct volumeLifeDetectionPOMDP <: POMDP{Int, Int, Int}  # POMDP{State, Action, Observation}
    bn::BayesianNetwork 
    λ::Float64    
    inst::Int64 
    sample_volume::Int64
    life_states::Int64
    surface_acc_rate::Int64
    sample_use::Vector{Int64} 
    # k::Vector{Float64} # cost of observations
    discount::Float64
end

# Custom constructor to handle dynamic initialization
function volumeLifeDetectionPOMDP(;
    bn::BayesianNetwork,
    λ::Float64,    
    inst::Int64 = 7,
    sample_volume::Int64 = 500,
    life_states::Int64 = 3,
    surface_acc_rate::Int64 = 270,
    sample_use::Vector{Int64} = [1,1,1,1,1,1,1], # cost of observations
    # k::Vector{Float64} = [HRMS*10e6, SMS*10e6, μCE_LI*10e6, ESA*10e6, microscope*10e6, nanopore*10e6], # cost of observations
    discount::Float64 = 0.9,
)
    return volumeLifeDetectionPOMDP(bn,λ,inst,sample_volume,life_states,surface_acc_rate,sample_use,discount)
end

# 1 -> dead
# 2 -> alive
# 3 -> terminal state
POMDPs.states(pomdp::volumeLifeDetectionPOMDP) =  1: pomdp.sample_volume*pomdp.life_states+pomdp.life_states #(pomdp.sample_volume*((2^pomdp.life_states)))+(2^pomdp.life_states)

# run sensor (2+i), where i the ith instrument, and the last instrument is doing nothing
# declare dead (1) declare alive (2)
POMDPs.actions(pomdp::volumeLifeDetectionPOMDP) = [1:pomdp.inst..., pomdp.inst+1, pomdp.inst+2]

# observe biosignature is present (1) or absent (2)
POMDPs.observations(pomdp::volumeLifeDetectionPOMDP) = 1:pomdp.sample_volume*pomdp.life_states+pomdp.life_states + 1 # null obs. #(pomdp.sample_volume*((2^pomdp.life_states)))+(2^pomdp.life_states)

POMDPs.stateindex(pomdp::volumeLifeDetectionPOMDP, s::Int) = s
POMDPs.actionindex(pomdp::volumeLifeDetectionPOMDP,a::Int) = a
POMDPs.obsindex(pomdp::volumeLifeDetectionPOMDP, s::Int)   = s

# TODO: probably want to set P(life) << 50%
POMDPs.initialstate(pomdp::volumeLifeDetectionPOMDP) = DiscreteUniform(1, 2) # 50% chance of being alive or dead with no starting sample    
                                                                             # state_to_stateindex(0, 1) # TODO: change in future, so it starts at any state

function POMDPs.isterminal(pomdp::volumeLifeDetectionPOMDP, s::Int) 
    _, life_state = stateindex_to_state(s, pomdp.life_states)
    if life_state == 3
        return true
    else
        return false
    end
end

POMDPs.discount(pomdp::volumeLifeDetectionPOMDP) = pomdp.discount


function POMDPs.transition(pomdp::volumeLifeDetectionPOMDP, s::Int, a::Int)

    # TODO: Need things for when action is to not turn on instrument
    # TODO: When previous action was using an instrument we are in melt mode 
            # Can't accumulate any more things at that point
    
    sample_volume, life_state = stateindex_to_state(s, pomdp.life_states)

    if a < pomdp.inst # run instrument
        sample_volume -= sample_volume >= pomdp.sample_use[a] ? pomdp.sample_use[a] : 0
    elseif a == pomdp.inst # accumulate
        sample_volume += pomdp.surface_acc_rate
    else # declare dead or alive
        return Deterministic(state_to_stateindex(sample_volume, 3))
    end

    sample_volume = clamp(sample_volume, 0, pomdp.sample_volume)
    return Deterministic(state_to_stateindex(sample_volume, life_state))
end

function POMDPs.observation(pomdp::volumeLifeDetectionPOMDP, a::Int,  sp::Int)

    sample_volume, life_state = stateindex_to_state(sp, pomdp.life_states)
    ob1 = state_to_stateindex(sample_volume, 1)
    ob2 = state_to_stateindex(sample_volume, 2)

    # if we already declared alive/dead, observation doesn't matter
    if POMDPs.isterminal(pomdp, sp) || a == pomdp.inst + 1 || a == pomdp.inst + 2
        return SparseCat([ob1, ob2], [0.5,0.5])
    end
    
    # accumulating
    if a == pomdp.inst
        return SparseCat([ob1, ob2], [0.5, 0.5])
    end

    if a > length(pomdp.bn.factors) # a --> instrument index
        error("Invalid action: $a")
    end

    # get the corresponding conditional probability P(O|L), or P(biosignature present|alive/dead)
    factor = pomdp.bn.factors[a+1]
    var_name = pomdp.bn.factors[a+1].vars[1].name
    key = Dict(var_name => 2, :l => life_state)  

    P_yes = factor.table[key] # Probability of "yes" observation for action a (a=2 means detected)
    return SparseCat([ob1, ob2], [1 - P_yes, P_yes]) # Return a probability distribution over the two observations (1 = no, 2 = yes)
end

function POMDPs.reward(pomdp::volumeLifeDetectionPOMDP, s::Int, a::Int) #, b::Vector{Float64})
    if a == pomdp.inst + 1  # declaring dead
        return s == 1 ? 0 : -pomdp.λ  # No reward if correct, penalty if wrong
    elseif a == pomdp.inst + 2  # declaring alive
        return s == 2 ? 0 : -pomdp.λ  # No reward if correct, penalty if wrong
    else  # Sensor action
        #prior_life = b[2]  # Probability of life from current belief state
        #exp_change = expected_belief_change(pomdp, a, prior_life)

        sample_volume, life_state = stateindex_to_state(s, pomdp.life_states)
        if sample_volume < pomdp.sample_use[a]
            return -10000
        end
        # more you're wasting the worse it is
        return -(1 - pomdp.λ)*(sample_volume/pomdp.sample_volume)# + exp_change  # Cost of using sensor multipled by lambda factor plus expected information gain
    end
end

function state_to_stateindex(sample_volume::Int, life_states::Int)
    return (sample_volume-1)*3+life_states+3
end
function stateindex_to_state(index::Int, n_life_states::Int)
    if index%n_life_states != 0
        sample_volume = div(index,n_life_states)
        life_states = index%n_life_states
    else
        sample_volume = div(index,n_life_states)-1
        life_states = n_life_states
    end
    return sample_volume, life_states
end