# module power_management
import POMDPs
using POMDPs: POMDP
using POMDPTools #: Deterministic, Uniform, SparseCat
using Distributions: Normal, cdf

# using Distributions
using Statistics # std
using Random # rand, and random seed
using LinearAlgebra
using SparseArrays

Random.seed!(42) # TODO: check if this actually works?

# Define mutable struct without default values, dynamic fields
struct PowerManagementPOMDP <: POMDP{Int,Int,Int}

    # instrument specific
    num_inst::Int
    priority_arr::Vector{Int}
    inst_battery_usage::Vector{Int}

    # Battery capacity
    battery_cp::Int
    deviations::Int
    charging::Int

    # idle timesteps
    idle_time_max::Int
    idle_times::Vector{Int} # mutable? maybe include this instead in the pomdp state

    # Rewards
    exceeded_capacity::Float64
    discount_factor::Float64
    idle_time_max_penalty::Float64

end

# # Custom constructor to handle dynamic initialization
function PowerManagementPOMDP(;
    num_inst::Int,
    priority_arr::Vector{Int}, #rand(1:10, num_inst)
    inst_battery_usage::Vector{Int},#rand(1:battery_cp, num_inst)
    idle_time_max::Int=10,
    charging::Int=100,
    battery_cp::Int=100,
    exceeded_capacity::Float64=1000000.0, # penalty for reward
    discount_factor::Float64=0.9,
    idle_time_max_penalty::Float64=10.0,
    deviations::Int=5,
)
    # priority_arr = [1,10,10,1,1] #rand(1:10, num_inst)
    # inst_battery_usage = [30, 30, 30, 30, 30]#rand(1:battery_cp, num_inst)
    idle_times = zeros(Int, num_inst)
    return PowerManagementPOMDP(
        num_inst,
        priority_arr,
        inst_battery_usage,
        battery_cp,
        deviations,
        charging,
        idle_time_max,
        idle_times,
        exceeded_capacity,
        discount_factor,
        idle_time_max_penalty,
    )
end

############################### Custom Functions ######################################

# Function to get a binary representation with a fixed number of digits
function to_fixed_binary(n::Int, num_digits::Int)
    binary_digits = digits(n, base=2)  # Convert to binary
    padded_binary = vcat(binary_digits, zeros(Int, num_digits - length(binary_digits)))  # Pad with leading zeros if needed
    return padded_binary
end


function state_decompose(total_num_inst_combos::Int, index::Int)
    battery_level = div(index, total_num_inst_combos)
    prev_action = index % total_num_inst_combos
    return battery_level, prev_action
end

function state_toindex(battery_level::Int, prev_state::Int, action_space::Int)
    index = battery_level * action_space + prev_state
    return index
end

function binned_normal_distribution(pomdp, mean::Int, std::Int)
    # Define the normal distribution
    dist = Normal(mean, std)

    left = std
    right = std

    if mean - left < 0
        left = mean - 0
    end
    if mean + right > pomdp.battery_cp
        right = pomdp.battery_cp - mean
    end
    num_bins = left+right

    # Define the range and bin edges
    bin_edges = range(mean - left, mean + right, length=num_bins + 1)

    # Calculate bin probabilities
    bin_probs = [cdf(dist, bin_edges[i+1]) - cdf(dist, bin_edges[i]) for i = 1:num_bins]
    normalized_probs = bin_probs / sum(bin_probs)

    return bin_edges, normalized_probs
end

############################### Work in Progress ######################################
# Trying to make invalid actions a thing

function POMDPs.actions(m::PowerManagementPOMDP, s)
    possible_actions = Int[]
    for a = 0:((2^m.num_inst)-1)
        action_vec = reverse(to_fixed_binary(a, m.num_inst))
        if (dot(m.inst_battery_usage, action_vec) < s)
            push!(possible_actions, a)
        end
    end
    return possible_actions
end


############################### POMDP Formulation ######################################

POMDPs.states(m::PowerManagementPOMDP) =
    0:((m.battery_cp*((2^m.num_inst)))+(2^m.num_inst)-1)
POMDPs.actions(m::PowerManagementPOMDP) = 0:((2^m.num_inst)-1)
POMDPs.observations(m::PowerManagementPOMDP) =
    0:((m.battery_cp*((2^m.num_inst)))+(2^m.num_inst)-1) # same as states
POMDPs.discount(m::PowerManagementPOMDP) = m.discount_factor

POMDPs.stateindex(m::PowerManagementPOMDP, s::Int) = s+1
POMDPs.actionindex(m::PowerManagementPOMDP, a::Int) = a+1
POMDPs.obsindex(m::PowerManagementPOMDP, s::Int) = s+1

POMDPs.initialstate(m::PowerManagementPOMDP) = Uniform(0:m.battery_cp) # gives back a distribution between 0 and 100

function POMDPs.transition(m::PowerManagementPOMDP, s, a)

    action_vec = (to_fixed_binary(a, m.num_inst)) # reverse

    battery_level, prev_state = state_decompose((2^m.num_inst), s)

    battery_level = battery_level+m.charging #rand(Uniform(0: m.battery_cp))
    if battery_level > m.battery_cp
        battery_level = m.battery_cp
    end

    battery_notdead = true

    # battery usage and battery collection
    temp = battery_level
    for (a, ind) in zip(action_vec, 1:length(action_vec))
        temp -= a*m.inst_battery_usage[ind]

        if temp < 0
            battery_level = 0
            battery_notdead = false
            break
        else
            battery_level = temp
        end
    end

    if battery_notdead
        # check idles
        for (a, ind) in zip(action_vec, 1:length(action_vec))
            if a == 0
                m.idle_times[ind] += 1
            else
                m.idle_times[ind] = 0
            end
        end
    end

    prev_state = a
    sp = state_toindex(Int(battery_level), prev_state, (2^m.num_inst))

    return Deterministic(sp)
end

function POMDPs.observation(m::PowerManagementPOMDP, a, sp)

    battery_level, prev_action = state_decompose(2^m.num_inst, sp)
    bins, probs = binned_normal_distribution(m, battery_level, m.deviations)
    # print(sum(probs))
    # print(sum(sparse(probs)))
    newbins = []
    for bat_level in bins
        push!(newbins, state_toindex(Int(round(bat_level)), prev_action, 2^m.num_inst))
    end
    return SparseCat(newbins[1:(end-1)], sparse(probs))

end

function POMDPs.reward(m::PowerManagementPOMDP, s, a)

    reward = 0.0

    battery_level, prev_action = state_decompose((2^m.num_inst), s)

    if battery_level == 0
        # negative rewards for invalid actions 
        reward -= m.exceeded_capacity
    else
        # negative rewards for idle times
        action_vec = (to_fixed_binary(a, m.num_inst)) #reverse
        flipped_action = map(x -> 1 - x, action_vec)
        reward -= dot(m.priority_arr, flipped_action) #*max_idle

        # negative rewards for repeated actions
        if prev_action == a
            reward -= 1
        end
    end

    return reward
end
