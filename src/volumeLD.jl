using POMDPs # TODO: check if i can delete this
using POMDPTools

struct volumeLifeDetectionPOMDP <: POMDP{Int, Int, Int}  # POMDP{State, Action, Observation}
    bn::DiscreteBayesNet # Bayesian Network,
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
    bn::DiscreteBayesNet, # Bayesian Network,
    λ::Float64,    
    inst::Int64 = 7, # number of instruments / not using instrument
    sampleVolume::Int64 = 500,
    lifeStates::Int64 = 3,
    surfaceAccRate::Int64 = 270,
    sampleUse::Vector{Int64} = [1,1,1,1,1,1,1], # cost of observations
    # k::Vector{Float64} = [HRMS*10e6, SMS*10e6, μCE_LI*10e6, ESA*10e6, microscope*10e6, nanopore*10e6], # cost of observations
    discount::Float64 = 0.9,
)
    return volumeLifeDetectionPOMDP(bn,λ,inst,sampleVolume,lifeStates,surfaceAccRate,sampleUse,discount)
end

# 1 -> dead
# 2 -> alive
# 3 -> terminal state
POMDPs.states(pomdp::volumeLifeDetectionPOMDP) =  1: pomdp.sampleVolume*pomdp.lifeStates+pomdp.lifeStates #(pomdp.sampleVolume*((2^pomdp.lifeStates)))+(2^pomdp.lifeStates)

# run sensor (2+i), where i the ith instrument, and the last instrument is doing nothing
# declare dead (1) declare alive (2)
POMDPs.actions(pomdp::volumeLifeDetectionPOMDP) = [1:pomdp.inst..., pomdp.inst+1, pomdp.inst+2]

# Extra observation at end which will be null observation
POMDPs.observations(pomdp::volumeLifeDetectionPOMDP) = 1: pomdp.sampleVolume*pomdp.lifeStates+pomdp.lifeStates+1 #(pomdp.sampleVolume*((2^pomdp.lifeStates)))+(2^pomdp.lifeStates)

POMDPs.stateindex(pomdp::volumeLifeDetectionPOMDP, s::Int) = s
POMDPs.actionindex(pomdp::volumeLifeDetectionPOMDP,a::Int) = a
POMDPs.obsindex(pomdp::volumeLifeDetectionPOMDP, o::Int)   = o


# TODO: do we want to start with different states?
POMDPs.initialstate(pomdp::volumeLifeDetectionPOMDP) = DiscreteUniform(1, 2) # 50% chance of being alive or dead with no starting sample    
                                                                             # state_to_stateindex(0, 1) # TODO: change in future, so it starts at any state

function POMDPs.isterminal(pomdp::volumeLifeDetectionPOMDP, s::Int) 
    sampleVolume, lifeState = stateindex_to_state(s, pomdp.lifeStates)
    if lifeState == 3
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

    sampleVolume, lifeState = stateindex_to_state(s, pomdp.lifeStates)
    
    # if lifeState == 3
    #     lifeState = rand(1:2) # Set new lifeState at new iteration
    #     sampleVolume = 0#pomdp.surfaceAccRate
    # end

    if a <= pomdp.inst
        
        # # if we choose to not use an instrument, thats the only time sampleVolume goes up
        if a == pomdp.inst 
            
            # Trying to make it so that waiting to use an instrument is
            # done multiple times
            # if sampleVolume % pomdp.surfaceAccRate != 1
            #     sampleVolume = 1 #pomdp.surfaceAccRate
            # end
            # lifeState = rand(1:2)
            sampleVolume += pomdp.surfaceAccRate

            # Always make sure sample Volume can't exceed certain volume
            if sampleVolume > pomdp.sampleVolume
                sampleVolume = pomdp.sampleVolume
            end
            
            P_yes = bn.cpds[1].distributions[1].p[2]
            s1 = state_to_stateindex(sampleVolume, 1)
            s2 = state_to_stateindex(sampleVolume, 2)

            return SparseCat([s1, s2], [1 - P_yes, P_yes])

        # if we choose to use an instrument, we take away from sampleVolume, assuming we are not choosing to wait for accumulation at this step
        else
            if sampleVolume >= pomdp.sampleUse[a]
                sampleVolume = sampleVolume - pomdp.sampleUse[a]
            end

        end
        # Always make sure sample Volume can't exceed certain volume
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

    # if we delcare (or already declared) alive/dead, observation doesn't matter
    if POMDPs.isterminal(pomdp, sp) || a == pomdp.inst + 1 || a == pomdp.inst + 2 || a == pomdp.inst
        return Deterministic(pomdp.sampleVolume*pomdp.lifeStates+pomdp.lifeStates+1 ) #SparseCat([ob1, ob2], [0.5,0.5])
    end

    # Sample first, then do infer to get posterior?

    # Instrument Action to sample characteristics:
    action_to_cpds = Dict(
        1 => [:C7, :C5, :C8, :C10],   # HRMS
        2 => [:C6, :C5],              # SMS
        3 => [:C6, :C5],              # μCE_LI
        4 => [:C7, :C8],              # ESA
        5 => [:C2, :C3],              # microscope
        6 => [:C1]                    # nanopore
    )
    # Get the specific sample indicies for action selected
    cpd_indices = [bn.name_to_index[var] for var in action_to_cpds[a]]
    evidence_dict = Dict()

    # For each CPD index, sample and add to evidence
    for idx in cpd_indices
        posterior = infer(bn, bn.cpds[idx].target, evidence=Assignment(Dict(:C0 => lifeState))) # Start with life state as evidence
        sample = rand(Categorical(convert(DataFrame, posterior)[!,"potential"]))
        evidence_dict[bn.cpds[idx].target] = sample
    end

    # All this is saying is: bn.cpds[1].target = :l
    # evidence -> infer -> posterior
    p_life = infer(bn, bn.cpds[1].target, evidence=Assignment(evidence_dict))
   
    # Return a probability distribution over the new posterior of life (1 = no, 2 = yes)
    return SparseCat([ob1, ob2], [p_life[1], p_life[2]])
end

function expected_belief_change(pomdp::volumeLifeDetectionPOMDP, a::Int, prior)
    if a >= pomdp.inst  # declare alive/dead
        return 0.0
    end
    # Map action to biosignature node in Bayesian network
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
    println("exp_change: ", exp_change)

    return exp_change
end


function POMDPs.reward(pomdp::volumeLifeDetectionPOMDP, s::Int, a::Int)#, b::Vector{Float64})
    sampleVolume, lifeState = stateindex_to_state(s, pomdp.lifeStates)

    if a == pomdp.inst + 1                   # declare alive
        return s == 1 ? 0 : -pomdp.λ         # penalize if wrong
    elseif a == pomdp.inst + 2               # declare dead
        return s == 2 ? 0 : -pomdp.λ         # penalize if wrong
    elseif sampleVolume < pomdp.sampleUse[a] # instrument action
        return -10000                        # invalid
    else                                     # instrument action
        # rho-POMDP: reward expected change in belief / information gain
        #prior = b[2]
        #exp_change = expected_belief_change(pomdp, a, prior)
        
        # penalize wasting sample volume
        return -(1 - pomdp.λ)*(sampleVolume/pomdp.sampleVolume) #+ exp_change 
    end
end

function state_to_stateindex(sampleVolume::Int, lifeStates::Int)
    return (sampleVolume-1)*3+lifeStates+3
end
function stateindex_to_state(index::Int, n_lifeStates::Int)
    if index % n_lifeStates != 0
        sampleVolume = div(index,n_lifeStates)
        lifeStates = index % n_lifeStates
    else
        sampleVolume = div(index,n_lifeStates)
        if div(index,n_lifeStates) != 0
            sampleVolume -=1
        end
        lifeStates = n_lifeStates
    end
    return sampleVolume, lifeStates
end