
POMDPs.actions(pomdp::LifeDetectionPOMDP) = 1:(pomdp.NumInst)

POMDPs.actionindex(pomdp::LifeDetectionPOMDP, a::Int) = a



# # Not exactly sure, but I think we need this version of actions too:
# function POMDPs.actions(m::LifeDetectionPOMDP, s) 
#     possible_actions = Int[]
#     for a in 0:(2^m.num_inst)-1
#         action_vec = reverse(to_fixed_binary(a, m.num_inst))
#         if (dot(m.inst_battery_usage,action_vec) < s)
#             push!(possible_actions, a)
#         end
#     end
#     return possible_actions
# end
