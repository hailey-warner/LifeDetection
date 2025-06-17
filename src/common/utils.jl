
function determineMaxObs(actionCpds,bn)

    maxObs = 0
    
    for action in collect(keys(actionCpds))
        for lifeState in [1,2]
            probs = Float64[]  # P(obs | l=1)
            posterior = infer(bn, actionCpds[action] ,evidence=(Assignment(:C0 => lifeState )))

            # Determine domain sizes (bins per variable)
            domain_sizes = [length(bn.cpds[bn.name_to_index[v]].distributions[1].p) for v in posterior.dimensions]
            obs_indices = collect(CartesianIndices(Tuple(domain_sizes)))

            # -- Inference Loop --
            for idx in obs_indices
                push!(probs, posterior.potential[idx])
            end
            if length(probs) > maxObs
                maxObs = length(probs)
            end
        end
    end

    return maxObs
end


function distObservations(actionCpds, lifeState, action, maxObs)

    posterior = infer(bn, actionCpds[action] ,evidence=(Assignment(:C0 => lifeState )))

    probs = Float64[]  # P(obs | l=1)

    # Determine domain sizes (bins per variable)
    domain_sizes = [length(bn.cpds[bn.name_to_index[v]].distributions[1].p) for v in posterior.dimensions]
    obs_indices = collect(CartesianIndices(Tuple(domain_sizes)))

    # -- Inference Loop --
    for idx in obs_indices
        push!(probs, posterior.potential[idx])
    end

    while length(probs) < maxObs
        push!(probs, 0)
    end

    probs ./= sum(probs)

    return probs#, Categorical(probs)
end
