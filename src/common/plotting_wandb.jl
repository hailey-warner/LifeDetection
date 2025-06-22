using Plots
using Graphs
# using GraphPlot
# using Compose
# using Cairo
# using Fontconfig
# using TikzGraphs
# using TikzPictures


# TODO: Need to fix alpha plots, and Decision tree plots
function dominating_alphas(policy::AlphaVectorPolicy)
	num_vectors = size(policy.alphas, 1)
	num_samples = 101  # or infer from alpha size
	b_vals = range(0, 1, length=100)

	dom_idx_per_sample = Vector{Int}(undef, num_samples)

	for s in 1:num_samples
		best_alpha = 0
		best_score = -Inf

		for i in 1:num_vectors
			α = policy.alphas[i]
			# get index of life=1 and life=2 at this sample
			idx1 = (s - 1) * 3 + 1  # life = 1
			idx2 = (s - 1) * 3 + 2  # life = 2

			Vb = [α[idx1]*b + α[idx2]*(1-b) for b in b_vals]
			max_V = maximum(Vb)

			if max_V > best_score
				best_score = max_V
				best_alpha = i
			end
		end

		dom_idx_per_sample[s] = best_alpha
	end

	return dom_idx_per_sample
end

function plot_alpha_dots(policy)
	dominating = dominating_alphas(policy)

	# Plots.plot(1:101, dominating,
	#     xlabel="Sample Volume",
	#     ylabel="Dominating Alpha Index",
	#     title="Dominating Alpha Vector per Sample Volume",
	#     legend=false)


	alpha_actions = [α for α in policy.action_map]  # or however actions are stored
	action_colors = Dict(a => c for (a, c) in zip(unique(alpha_actions), palette(:tab10)))

	action_per_sample = [alpha_actions[i] for i in dominating]
	sample_range = 1:length(dominating)

	p = scatter(sample_range, dominating,
		group=action_per_sample,
		color=action_per_sample .|> x -> action_colors[x],
		xlabel="Sample Volume",
		ylabel="Dominating Alpha Index",
		title="Dominating Alpha Vector per Sample Volume",
		legend=:topright,
		markersize=4)

	# yticks = unique(dominating)
	# ytick_labels = ["a=$(policy.action_map[i])" for i in yticks]

	# Plots.plot(1:101, dominating,
	#     xlabel="Sample Volume",
	#     ylabel="Dominating Alpha Index",
	#     title="Dominating Alpha Vector per Sample Volume",
	#     yticks=(yticks, ytick_labels),
	#     legend=false)

	# changes = findall(diff(dominating) .!= 0)
	# vline!(changes, line=:dash, color=:gray)

	if !isdir("./figures")
		mkpath("./figures")
	end
	savefig(p, "./figures/plot_alpha_dots.png")
	display(p)
	return p

end


function plot_decision_tree(tree_data::Tuple{Vector{String}, Dict{Tuple{Int64, Int64}, String}, Vector{Tuple{Int64, Int64}}})
	node_labels, edge_colors, edges = tree_data

	g = DiGraph(length(node_labels))
	for (i, j) in edges
		add_edge!(g, i, j)
	end

	t = TikzGraphs.plot(g, TikzGraphs.Layouts.Layered(), node_labels, edge_styles=edge_colors,
		node_style="draw", graph_options="nodes={draw,circle}")
	# NOTE: delete line '\setmainfont{Latin Modern Math}' to compile
	TikzPictures.save(TikzPictures.TEX("./figures/decision_tree.tex"), t)
	return t
end

# TODO: need to fix this
function plot_bayes_net(bn::DiscreteBayesNet)
	node_labels = [string(v.name) for v in bn.vars]
	t = TikzGraphs.plot(bn.graph, TikzGraphs.Layouts.Spring(), node_labels,
		node_style="draw", graph_options="nodes={draw,circle}")
	# NOTE: delete line '\setmainfont{Latin Modern Math}' to compile           
	TikzPictures.save(TikzPictures.TEX("./figures/bayes_net.tex"), t)
	return t
end

function plot_pareto_frontier()
end


function plot_belief_over_time(b_hist, a_hist; pomdp=nothing)
	action_labels = if pomdp === nothing
		string.(a_hist)
	else
		[a ≥ pomdp.inst+1 ? (a == pomdp.inst+1 ? "Declare Dead" : "Declare Life") :
		 (a == pomdp.inst ? "Accumulate" : "Sensor $(a)") for a in a_hist]
	end

	p = plot()
	for a_label in unique(action_labels)
		idxs = findall(==(a_label), action_labels)
		plot!(idxs, b_hist[idxs], label=a_label)
	end

	xlabel!("Step")
	ylabel!("P(life = 1)")
	title!("Belief in Life Over Time by Action")

	savefig(p, "./figures/belief_over_time.png")
	display(p)
	return p
end





function make_decision_tree(pomdp, policy; max_depth=3)
	node_labels = String[]
	edge_colors = String[]
	edges = Tuple{Int, Int}[]

	updater = DiscreteUpdater(pomdp)
	b0 = initialize_belief(updater, initialstate(pomdp))

	function traverse(b, parent_index::Union{Int, Nothing}=nothing, edge_color::Union{String, Nothing}=nothing, depth=1)
		a = action(policy, b)
		b_val = pdf(b, 2)
		action_name = a ≤ 2 ? (a == 1 ? "Dead" : "Alive") : "I$(a - 2)"
		label = "$(action_name), P(life)=$(round(b_val, digits=2))"
		push!(node_labels, label)
		current_index = length(node_labels)

		# record edge if a parent exists
		if parent_index !== nothing && edge_color !== nothing
			push!(edge_colors, edge_color)
			push!(edges, (parent_index, current_index))
		end

		# stop if we've reached max depth or if action is terminal
		if depth >= max_depth || a ≤ 2
			return
		end

		# branch for each observation
		for o in POMDPs.observations(pomdp)
			obs_name = o == 1 ? "red" : "green"
			b_new = update(updater, b, a, o)
			traverse(b_new, current_index, obs_name, depth+1)
		end
	end

	traverse(b0)
	edge_colors = Dict(zip(edges, edge_colors))
	return (node_labels, edge_colors, edges)
end
