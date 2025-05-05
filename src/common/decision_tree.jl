using Graphs
using TikzGraphs
using TikzPictures
using Compose
using Cairo
using Fontconfig
# using D3Trees

function plot_decision_tree(tree_data::Tuple{Vector{String}, Dict{Tuple{Int64, Int64}, String}, Vector{Tuple{Int64, Int64}}})
    node_labels, edge_colors, edges = tree_data

    g = DiGraph(length(node_labels))
    for (i, j) in edges
        add_edge!(g, i, j)
    end

    t = TikzGraphs.plot(g, TikzGraphs.Layouts.Layered(), node_labels, edge_styles=edge_colors,
                         node_style="draw", graph_options="nodes={draw,circle}")
    TikzPictures.save(TikzPictures.TEX("decision_tree.tex"), t)
    return t
end

function plot_bayes_net(bn::BayesianNetwork)
    node_labels = [string(v.name) for v in bn.vars]
    t = TikzGraphs.plot(bn.graph, TikzGraphs.Layouts.Spring(), node_labels,
                        node_style="draw", graph_options="nodes={draw,circle}")
    TikzPictures.save(TikzPictures.TEX("bayes_net.tex"), t)
    return t
end
