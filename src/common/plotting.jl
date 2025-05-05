using Plots
using GraphPlot
using Graphs
using Compose
using Cairo
using Fontconfig
using TikzGraphs
using TikzPictures


function plot_alpha_vectors(policy::AlphaVectorPolicy)
    # get alpha vectors
    alpha_vectors = policy.alphas
    num_states = size(alpha_vectors, 2) 
    num_vectors = size(alpha_vectors, 1)
    
    b = range(0, 1, length=100) # P(life)

    # plot each alpha vector
    p = Plots.plot(title="Alpha Vectors",
               xlabel="Estimated P(life)", ylabel="Value Function", legend=false)

    for i in 1:num_vectors
        # compute the value function for belief state b
        # V(b) = α₁*b + α₂*(1-b)
        V_b = [alpha_vectors[i][1] * b_i + alpha_vectors[i][2] * (1 - b_i) for b_i in b]
        
        # plot each alpha vector as a line
        # note : if action i isn't present, it means running instrument i was not optimal.
        plot!(b, V_b, label="α_$i (a=$(policy.action_map[i]))")
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

    t = TikzGraphs.plot(g, TikzGraphs.Layouts.Layered(), node_labels, edge_styles=edge_colors,
                         node_style="draw", graph_options="nodes={draw,circle}")
    TikzPictures.save(TikzPictures.TEX("./figures/decision_tree.tex"), t)
    return t
end

function plot_bayes_net(bn::BayesianNetwork)
    node_labels = [string(v.name) for v in bn.vars]
    t = TikzGraphs.plot(bn.graph, TikzGraphs.Layouts.Spring(), node_labels,
                        node_style="draw", graph_options="nodes={draw,circle}")
    TikzPictures.save(TikzPictures.TEX("./figures/bayes_net.tex"), t)
    return t
end

function plot_pareto_frontier()
end

