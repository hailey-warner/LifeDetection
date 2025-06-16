# using LifeDetection # TODO: Use this when we get to make it to a full module
include("../src/LifeDetection.jl")

using StaticArrays
using POMDPs
using SARSOP
using POMDPLinter
using DiscreteValueIteration

using .LifeDetection: LifeDetectionPOMDP, LDState, stateindex, state_from_index



num_inst = 3
# 5 is setting the number of instruments
pomdp = LifeDetectionPOMDP(num_inst)

for inst_3 = 1:5
    for inst_2 = 1:5
        for inst_1 = 1:5
            for sample_certainty = 1:pomdp.SampleCertaintyMax

                # Example State
                # sample_certainty = 20

                # Create the state
                state = LDState(
                    sample_certainty,
                    SVector{num_inst,Int}([inst_1, inst_2, inst_3]),
                )
                a = action(policy, state)
                index = stateindex_func(pomdp, state)
                re_state = state_from_index(pomdp, index)
                print(state)
                print(index, "   ")
                print(" action: ", a)
                println(" ", re_state)

            end
        end
    end
end

pomdp.SampleTrueVal
POMDPs.transition(pomdp, state_from_index(pomdp, 132), 3)

state
re_state = state_from_index(pomdp, 1249)
re_state.inst_health[3] = re_state.inst_health[3]+5
re_state
# SARSOP Specific Solver
using Distributions
solver = SARSOPSolver(verbose=true, timeout=100)
solver = ValueIterationSolver(max_iterations=100, belres=1e-6, verbose=true) # creates the solver
@show_requirements POMDPs.solve(solver, pomdp)
solve(solver, pomdp)
using POMDPTools

mdp = UnderlyingMDP(pomdp)
policy = solve(solver, mdp) # runs value iterationssl





#### TESTING

using Distributions, StatsPlots

# Parameters
μ = 10.0  # Mean
σ = 1.0  # Standard deviation
lb, ub = -1.0, 1.0  # Truncation bounds

InstSigma = [10.0^(i-1) for i = (num_inst-1):-1:0]

# Normal vs. Truncated Normal
normal_dist = Normal(μ, InstSigma[1])
truncated_dist = Truncated(normal_dist, 1, 10)
rand(truncated_dist)
# Visualization
plot(x -> pdf(normal_dist, x), -2, 12, label="Normal", lw=2)
plot!(
    x -> pdf(truncated_dist, x),
    -2,
    12,
    label="Truncated (Renormalized)",
    lw=2,
    linestyle=:dash,
)
