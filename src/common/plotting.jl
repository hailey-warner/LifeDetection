using Plots
using GraphPlot
using Graphs
using Compose
using Cairo
using Fontconfig
using TikzGraphs
using TikzPictures

function plot_alpha_vectors_VLD(policy::AlphaVectorPolicy, pomdp, sample::Int)
	# get alpha vectors and action map
	alpha_vectors = policy.alphas
	action_map = policy.action_map

	num_vectors = size(alpha_vectors, 1)
	b = range(0, 1, length = 200)  # Belief in L=1 (life)
	# b = range(1, pomdp.sample, length=200)  # Belief in L=1 (life)
	# b = range(1, pomdp.sample_volume*pomdp.lifeStates+pomdp.lifeStates, length=200)  # Belief in L=1 (life)

	p = Plots.plot(title = "Alpha Vectors at Sample Volume = $sample",
		xlabel = "Belief in Life (P(L=1))", ylabel = "Value",
		legend = :topright)

	for i in 1:num_vectors
		α = alpha_vectors[i]
		# get indices for dead and life at this sample volume
		dead_idx = state_to_stateindex(sample, 1)
		life_idx = state_to_stateindex(sample, 2)

		# alpha vector values for this sample
		α_dead = α[dead_idx]
		α_life = α[life_idx]

		# compute V(b) = α_life * b + α_dead * (1 - b)
		V_b = [α_life * b_i + α_dead * (1 - b_i) for b_i in b]
		plot!(b, V_b, label = "α_$i (a=$(action_map[i]))")
	end

	display(p)
	savefig(p, "alpha_vectors_sample_$sample.png")
	return p
end


function plot_alpha_vectors(policy::AlphaVectorPolicy)
	# get alpha vectors
	alpha_vectors = policy.alphas
	num_states = size(alpha_vectors, 2)
	num_vectors = size(alpha_vectors, 1)

	b = range(0, 1, length = 100) # P(life)

	# plot each alpha vector
	p = Plots.plot(title = "Alpha Vectors",
		xlabel = "Estimated P(life)", ylabel = "Value Function", legend = false)

	for i in 1:num_vectors
		# compute the value function for belief state b
		# V(b) = α₁*b + α₂*(1-b)
		V_b = [alpha_vectors[i][1] * b_i + alpha_vectors[i][2] * (1 - b_i) for b_i in b]

		# plot each alpha vector as a line
		# note : if action i isn't present, it means running instrument i was not optimal.
		plot!(b, V_b, label = "α_$i (a=$(policy.action_map[i]))")
	end
	if !isdir("./figures")
		mkpath("./figures")
	end
	savefig(p, "./figures/alpha_vectors.png")
	return p
end

function plot_decision_tree(tree_data::Tuple{Vector{String}, Dict{Tuple{Int64, Int64}, String}, Vector{Tuple{Int64, Int64}}})
	node_labels, edge_colors, edges = tree_data

	g = DiGraph(length(node_labels))
	for (i, j) in edges
		add_edge!(g, i, j)
	end

	t = TikzGraphs.plot(g, TikzGraphs.Layouts.Layered(), node_labels, edge_styles = edge_colors,
		node_style = "draw", graph_options = "nodes={draw,circle}")
	# NOTE: delete line '\setmainfont{Latin Modern Math}' to compile
	TikzPictures.save(TikzPictures.TEX("./figures/decision_tree.tex"), t)
	return t
end

# TODO: need to fix this
function plot_bayes_net(bn::DiscreteBayesNet)
	node_labels = [string(v.name) for v in bn.vars]
	t = TikzGraphs.plot(bn.graph, TikzGraphs.Layouts.Spring(), node_labels,
		node_style = "draw", graph_options = "nodes={draw,circle}")
	# NOTE: delete line '\setmainfont{Latin Modern Math}' to compile           
	TikzPictures.save(TikzPictures.TEX("./figures/bayes_net.tex"), t)
	return t
end

function plot_pareto_frontier()
end

