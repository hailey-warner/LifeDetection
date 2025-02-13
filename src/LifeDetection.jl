# module LifeDetection

import POMDPs
using POMDPs: POMDP, stateindex

using StaticArrays # for SVector
using Parameters # for @with_kw
using Distributions
using POMDPTools: Deterministic
using SparseArrays
using Random

"""
LDState{K}
Represents the state in a LifeDetectionPOMDP problem. 
`K` is an integer representing the number of instruments

# Fields
- `inst_imp::SVector{K, Bool}` the status of the rocks (false={No Event Happening}, true={Special Event, Turn on Instrument})
- `idle_time::SVector{K, Int}` idle time of each instrument in timesteps, capped at idle_time_max
"""

# terminal state >> associate with full mission success
# add an additional variable for each instrument completion?
# lean the science measurement plans that we can base power maangement on

#  - `battery::Int` integer between 1 and Battery Capacity # TODO: Implement again in the future?
mutable struct LDState
    # battery::Int  # TODO: implement again in the future?
    sample_certainty::Int
    inst_health::Vector{Int}  # N is determined dynamically at runtime
end

# # Custom Constructor for dynamic size handling
# function LDState(sample_certainty::Int, inst_health::Vector{Int})
#     N = length(inst_health)  # Determine size dynamically
#     return LDState(sample_certainty, Vector{Int}(inst_health))
# end


struct LifeDetectionPOMDP <: POMDP{LDState, Int, Int}

    # Number of Instruments
    NumInst::Int

    # Just a hardcoded thing
    InstSigma::Vector{Float64}

    # True Percentage
    SampleTrueVal::Int

    # Sample Certainty Max Percentage (typically 100)
    SampleCertaintyMax::Int

    # Health of Instruments Indicator (Maybe 1 is low, 5 is high?)
    InstHealthMax::Int

    # weights for indicies, faster index calculations
    indices::Vector{Int}

    discount_factor::Float64
end

# Custom Constructor
function LifeDetectionPOMDP(
    NumInst::Int=3;
    InstSigma::Vector{Float64}=[10.0^(i-2) for i in (NumInst-1):-1:0],
    SampleTrueVal::Int=rand(0:100),
    SampleCertaintyMax::Int=10, 
    InstHealthMax::Int=5,
    indices = cumprod([SampleCertaintyMax, fill(InstHealthMax, NumInst-1)...]),
    discount_factor::Float64 = 0.9,
    )
    return LifeDetectionPOMDP(
        NumInst,
        InstSigma,
        SampleTrueVal,
        SampleCertaintyMax, 
        InstHealthMax, 
        indices,
        discount_factor
    )
end

POMDPs.discount(pomdp::LifeDetectionPOMDP) = pomdp.discount_factor

include("states.jl")
include("actions.jl")
include("transition.jl")
include("observations.jl")
include("reward.jl")

# end # module LifeDetection


