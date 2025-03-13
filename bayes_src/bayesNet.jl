include("utils.jl")


# variable name, and the number on the right represents number of states (assuming binary for now)
L = Variable(:l, 2); # probability of life (base node)
A = Variable(:a, 2); # Amino acids
P = Variable(:p, 2); # Polyelectrolytes
C = Variable(:c, 2); # Cells
vars = [L, A, P, C]

# conditional probabilities of given l is false of how likely it is to be false
P_a_1 = 0.1
P_p_1 = 0.1
P_c_1 = 0.1

# conditional probabilities of given l is true of how likely it is to be false
P_a_2 = 0.5
P_p_2 = 0.5
P_c_2 = 0.5

factors = [

# for now, set the probability of life to be half half
Factor([L], FactorTable((l=1,) => 0.5, (l=2,) => 0.5)),

# Set factor table
Factor([A,L], FactorTable(
    (a=1,l=1) => P_a_1, 
    (a=2,l=1) => 1 - P_a_1, 
    (a=1,l=2) => P_a_2, 
    (a=2,l=2) => 1 - P_a_2, 
)),
Factor([P,L], FactorTable(
    (p=1,l=1) => P_p_1, 
    (p=2,l=1) => 1 - P_p_1, 
    (p=1,l=2) => P_p_2, 
    (p=2,l=2) => 1 - P_p_2, 
)),
Factor([C,L], FactorTable(
    (c=1,l=1) => P_c_1, 
    (c=2,l=1) => 1 - P_c_1, 
    (c=1,l=2) => P_c_2, 
    (c=2,l=2) => 1 - P_c_2, 
            
))
]
graph = SimpleDiGraph(4)
add_edge!(graph, 1, 2)  # L → A
add_edge!(graph, 1, 3)  # L → P
add_edge!(graph, 1, 4)  # L → C

bn = BayesianNetwork(vars, factors, graph)

a = (l=2,a=1,p=1,c=1) 
probability(bn, Assignment(a))
