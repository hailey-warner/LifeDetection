
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

function create_bayes_net(num_instruments, instrument_names, instrument_probs_alive, instrument_probs_dead)
    # Define base variables
    variable_specs = [Variable(:l, 2)] # Life variable
    dependencies = []
    probability_tables = [
        ([variable_specs[1].name], [(l=1,) => 0.5, (l=2,) => 0.5]) # Prior probability of life
    ]
    
    # Generate instrument-related variables and dependencies
    for (i, name_str) in enumerate(instrument_names)
        name = Symbol(name_str)  # Convert user input string to Symbol
        push!(variable_specs, Variable(name, 2))
        push!(dependencies, (variable_specs[1], variable_specs[i+1]))
        
        p_alive = instrument_probs_alive[i]
        p_dead = instrument_probs_dead[i]
        push!(probability_tables, (
            [variable_specs[i+1].name, variable_specs[1].name],
            [
                (name=1, l=1) => p_alive,
                (name=2, l=1) => 1 - p_alive,
                (name=1, l=2) => p_dead,
                (name=2, l=2) => 1 - p_dead
            ]
        ))
    end
    
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


# # Example Usage
# num_instruments = 3
# instrument_names = ["sensorA", "sensorB", "sensorC"]
# instrument_probs_alive = [0.9, 0.7, 0.75] # Probabilities of detection given life exists
# instrument_probs_dead = [0.1, 0.3, 0.25]  # Probabilities of false detection given no life

# bn = create_bayes_net(num_instruments, instrument_names, instrument_probs_alive, instrument_probs_dead)

# a = (:l=>2, :sensorA=>1, :sensorB=>1, :sensorC=>1)
# probability(bn, Assignment(a))

