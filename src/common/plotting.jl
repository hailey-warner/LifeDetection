using Plots
using Graphs
using GraphPlot
using TikzGraphs
using TikzPictures
using Compose
using Cairo
using Fontconfig
using D3Trees

function plot_alpha_vectors(policy::AlphaVectorPolicy)
    # get alpha vectors
    alpha_vectors = policy.alphas
    num_states = size(alpha_vectors, 2) 
    num_vectors = size(alpha_vectors, 1)
    
    # x-axis represents belief L = 1, or P(life)
    b = range(0, 1, length=100)

    # plot each alpha vector
    p = Plots.plot(title="Alpha Vectors (Piecewise-Linear Value Function)",
               xlabel="Belief in L=1", ylabel="Value Function")

    for i in 1:num_vectors
        # compute the value function for belief state b
        # V(b) = α₁*b + α₂*(1-b)
        V_b = [alpha_vectors[i][1] * b_i + alpha_vectors[i][2] * (1 - b_i) for b_i in b]
        
        # plot each alpha vector as a line
        # note : if action i isn't present, it means running instrument i was not optimal.
        plot!(b, V_b, label="α_$i (a=$(policy.action_map[i]))")
    end

    display(p)
    savefig(p, "alpha_vectors.png")
    return p
end

function plot_bayes_net(bn::BayesianNetwork)
    node_labels = [string(v.name) for v in bn.vars]
    node_fill_colors = [colorant"steelblue" for _ in 1:length(bn.vars)]
    
    p = gplot(bn.graph, 
              nodelabel=node_labels,
              nodefillc=node_fill_colors,
              edgestrokec=colorant"steelblue",
              nodelabelc=colorant"white", 
              NODESIZE=0.2,
              arrowlengthfrac=0.15,
              layout=spring_layout)

    draw(PNG("bayes_net.png", 16cm, 16cm), p)
    return p
end

function plot_decision_tree(tree_data::Tuple{Vector{String}, Vector{String}, Vector{Tuple{Int64, Int64}}})
    # nodes: actions (instrument run) + belief (P(life))
    # edges: observations (biosignature present/absent)

    node_labels, edge_labels, edges = tree_data
    g = DiGraph(length(node_labels))

    for (i, j) in edges
        add_edge!(g, i, j)
    end

    edge_label_dict = Dict(zip(edges, edge_labels))
    t = TikzGraphs.plot(g, node_labels; edge_labels=edge_label_dict)
    TikzPictures.save(PDF("graph"), t)

    return g
end

