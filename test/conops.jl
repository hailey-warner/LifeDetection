function conops_orbiter(pomdp, o, mode_acc, prev_action)
	"""
	
	"""
	sample_volume, _ = stateindex_to_state(o, pomdp.life_states)

	if mode_acc || prev_action >= 5 || pomdp.sample_use[prev_action+1] >= sample_volume
		return (7, true, 0)
	end

	return (prev_action + 1, false, prev_action + 1)
end
