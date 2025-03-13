using Graphs
# requires convenience functions from appendix G.5 
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
# variable name, and the number on the right represents number of states (assuming binary for now)
L = Variable(:l, 2); # probability of life (base node)
A = Variable(:a, 2); # Amino acids
P = Variable(:p, 2); # Polyelectrolytes
C = Variable(:c, 2); # Cells
vars = [L, A, P, C]


# conditional probabilities given L is True
P_a_l = 0.5 
P_p_l = 0.5
P_c_l = 0.5

factors = [

# for now, set the probability of life to be half half
Factor([L], FactorTable((l=1,) => 0.5, (l=2,) => 0.5)),

# Set factor table
Factor([A,L], FactorTable(
    (a=1,l=1) => P_a_l, 
    (a=2,l=1) => 1 - P_a_l, 
    (a=1,l=2) => P_a_l, 
    (a=2,l=2) => 1 - P_a_l, 
)),
Factor([P,L], FactorTable(
    (p=1,l=1) => P_p_l, 
    (p=2,l=1) => 1 - P_p_l, 
    (p=1,l=2) => P_p_l, 
    (p=2,l=2) => 1 - P_p_l, 
)),
Factor([C,L], FactorTable(
    (c=1,l=1) => P_c_l, 
    (c=2,l=1) => 1 - P_c_l, 
    (c=1,l=2) => P_c_l, 
    (c=2,l=2) => 1 - P_c_l, 
            
))
]
graph = SimpleDiGraph(4)
add_edge!(graph, 1, 2)  # L → A
add_edge!(graph, 1, 3)  # L → P
add_edge!(graph, 1, 4)  # L → C

bn = BayesianNetwork(vars, factors, graph)


select(a::Assignment, varnames::Vector{Symbol}) = Assignment(n=>a[n] for n in varnames)
variablenames(φ::Factor) = [var.name for var in φ.vars]
function probability(bn::BayesianNetwork, assignment) 
    subassignment(φ) = select(assignment, variablenames(φ)) 
    probability(φ) = get(φ.table, subassignment(φ), 0.0) 
    return prod(probability(φ) for φ in bn.factors) 
end

a = (l=2,a=1,p=1,c=1) 
probability(bn, Assignment(a))