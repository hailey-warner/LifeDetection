
function Base.length(mdp::LifeDetectionMDP)
    return 0 # of states
end

POMDPs.states(mdp::LifeDetectionMDP) = []

POMDPs.initialstate(mdp::LifeDetectionMDP) = []

# mass of sample [0, 100] ?
# pH [0, 14]
# polyelectrolite presence [boolean ]
# temperature [0, 100]
# salinity [0, 40]
# turbidity [0, 100]
# dissolved oxygen [0, 100]
# conductivity [0, 100]
