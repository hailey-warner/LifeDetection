include("utils.jl")


# variable name, number of states (assuming binary for now)
L = Variable(:l, 2); # probability of life (base node)
A = Variable(:a, 2); # Amino acids
P = Variable(:p, 2); # Polyelectrolytes
C = Variable(:c, 2); # Cells
vars = [L, A, P, C]

# P(Sensor=true | L=false)
P_a_alive = 0.20
P_p_alive = 0.01
P_c_alive = 0.05

# P(Sensor=true | L=true)
P_a_dead = 0.50
P_p_dead = 0.80
P_c_dead = 0.70

# P(Sensor=true)
P_a = 0.10
P_p = 0.01
P_c = 0.02

factors = [

# Prior
# for now, set the probability of life to be half half
Factor([L], FactorTable((l=1,) => 0.5, (l=2,) => 0.5)),

# Set factor table
Factor([A,L], FactorTable(
    (a=1,l=1) => P_a_alive, 
    (a=2,l=1) => 1 - P_a_alive, 
    (a=1,l=2) => P_a_dead, 
    (a=2,l=2) => 1 - P_a_dead, 
)),
Factor([P,L], FactorTable(
    (p=1,l=1) => P_p_alive, 
    (p=2,l=1) => 1 - P_p_alive, 
    (p=1,l=2) => P_p_dead, 
    (p=2,l=2) => 1 - P_p_dead, 
)),
Factor([C,L], FactorTable(
    (c=1,l=1) => P_c_alive, 
    (c=2,l=1) => 1 - P_c_alive, 
    (c=1,l=2) => P_c_dead, 
    (c=2,l=2) => 1 - P_c_dead, 
            
))
]
graph = SimpleDiGraph(4)
add_edge!(graph, 1, 2)  # L → A
add_edge!(graph, 1, 3)  # L → P
add_edge!(graph, 1, 4)  # L → C

bn = BayesianNetwork(vars, factors, graph)

a = (l=2,a=1,p=1,c=1) 
probability(bn, Assignment(a))
