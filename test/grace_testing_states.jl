# using LifeDetection # TODO: Use this when we get to make it to a full module
include("../src/LifeDetection.jl")

using StaticArrays
using POMDPs
using SARSOP
using POMDPLinter
using DiscreteValueIteration

using .LifeDetection: LifeDetectionPOMDP, LDState, stateindex,state_from_index



num_inst = 3
# 5 is setting the number of instruments
pomdp = LifeDetectionPOMDP(num_inst)

for inst_3 in 1:5
for inst_2 in 1:5
    for inst_1 in 1:5
        for sample_certainty in 1:pomdp.SampleCertaintyMax

            # Example State
            # sample_certainty = 20

            # Create the state
            state = LDState(sample_certainty, SVector{num_inst,Int}([inst_1,inst_2,inst_3]))
            index = stateindex(pomdp,state)
            re_state = state_from_index(pomdp, index)
            print(state)
            print(index, "   ")
            println(re_state)
            end
        end
    end
end

state
re_state = state_from_index(pomdp, 1249)
re_state.inst_health[3] = re_state.inst_health[3]+5
re_state
# SARSOP Specific Solver
using Distributions
solver = SARSOPSolver(verbose = true, timeout=100)
solver = ValueIterationSolver(max_iterations=100, belres=1e-6, verbose=true) # creates the solver
@show_requirements POMDPs.solve(solver, pomdp)
solve(solver, pomdp)
using POMDPTools

mdp = UnderlyingMDP(pomdp)
solve(solver, mdp) # runs value iterationss