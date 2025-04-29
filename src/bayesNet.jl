
include("common/utils.jl")

function bayes_net(variable_specs, dependencies, probability_tables)
    # Create variables
    vars = [Variable(name, states) for (name, states) in variable_specs]
    var_dict = Dict(var.name => var for var in vars)
    
    # Create factors
    factors = []
    for (vars_in_factor, prob_table) in probability_tables
        factor_vars = [var_dict[v] for v in vars_in_factor]
        push!(factors, Factor(factor_vars, FactorTable(prob_table...)))
    end
    
    # Create graph
    graph = SimpleDiGraph(length(vars))
    for (parent, child) in dependencies
        add_edge!(graph, findfirst(x -> x.name == parent, vars), findfirst(x -> x.name == child, vars))
    end
    
    # Return Bayesian Network
    return BayesianNetwork(vars, factors, graph)
end