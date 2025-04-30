using Plots
using Graphs
using GraphPlot
using Compose
using Cairo
using Fontconfig
using D3Trees

function plot_alpha_vectors_VLD(policy::AlphaVectorPolicy, pomdp, sample::Int)
    # get alpha vectors and action map
    alpha_vectors = policy.alphas
    action_map = policy.action_map

    num_vectors = size(alpha_vectors, 1)
    b = range(0, 1, length=200)  # Belief in L=1 (life)
    # b = range(1, pomdp.sample, length=200)  # Belief in L=1 (life)
    # b = range(1, pomdp.sampleVolume*pomdp.lifeStates+pomdp.lifeStates, length=200)  # Belief in L=1 (life)

    p = plot(title="Alpha Vectors at Sample Volume = $sample",
             xlabel="Belief in Life (P(L=1))", ylabel="Value",
             legend=:topright)

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
        plot!(b, V_b, label="α_$i (a=$(action_map[i]))")
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
    
    # x-axis represents belief L = 1, or P(life)
    b = range(0, 1, length=100)

    # plot each alpha vector
    p = plot(title="Alpha Vectors (Piecewise-Linear Value Function)",
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
    # nodes = actions (instrument run) + belief (P(life))
    # edges = observations (biosignature present/absent)

    node_labels, edge_labels, edges = tree_data

    children = [Int[] for _ in 1:length(node_labels)]

    for (i, j) in edges
        push!(children[i], j)
    end
    println("children: ", children)

    tree = D3Tree(children, text=node_labels, init_expand=true)
    inchrome(tree)
    return tree
end

