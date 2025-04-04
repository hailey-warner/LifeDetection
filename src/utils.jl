using Graphs

# AA228 convenience functions from appendix G.5 (Textbook: Decision Making Under Uncertainty)
Base.Dict{Symbol,V}(a::NamedTuple) where V = Dict{Symbol,V}(n=>v for (n,v) in zip(keys(a), values(a)))
Base.convert(::Type{Dict{Symbol,V}}, a::NamedTuple) where V = Dict{Symbol,V}(a)
Base.isequal(a::Dict{Symbol,<:Any}, nt::NamedTuple) =
    length(a) == length(nt) && all(a[n] == v for (n,v) in zip(keys(nt), values(nt)))

struct Variable 
    name::Symbol 
    r::Int # number of possible values 
end

const Assignment = Dict{Symbol,Int}
const FactorTable = Dict{Assignment,Float64} 
struct Factor 
    vars::Vector{Variable} 
    table::FactorTable 
end

struct BayesianNetwork 
    vars::Vector{Variable} 
    factors::Vector{Factor} 
    graph::SimpleDiGraph{Int64} 
end

select(a::Assignment, varnames::Vector{Symbol}) = Assignment(n=>a[n] for n in varnames)
variablenames(φ::Factor) = [var.name for var in φ.vars]
function probability(bn::BayesianNetwork, assignment) 
    subassignment(φ) = select(assignment, variablenames(φ)) 
    probability(φ) = get(φ.table, subassignment(φ), 0.0) 
    return prod(probability(φ) for φ in bn.factors) 
end