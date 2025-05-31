using Plots
using GraphPlot
using Graphs
using Compose
using Cairo
using Fontconfig
using TikzGraphs
using TikzPictures



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

# dominating = dominating_alphas(policy)

# Plots.plot(1:101, dominating,
#     xlabel="Sample Volume",
#     ylabel="Dominating Alpha Index",
#     title="Dominating Alpha Vector per Sample Volume",
#     legend=false)


alpha_actions = [α for α in policy.action_map]  # or however actions are stored
action_colors = Dict(a => c for (a, c) in zip(unique(alpha_actions), palette(:tab10)))

action_per_sample = [alpha_actions[i] for i in dominating]
sample_range = 1:length(dominating)

scatter(sample_range, dominating,
    group=action_per_sample,
    color=action_per_sample .|> x -> action_colors[x],
    xlabel="Sample Volume",
    ylabel="Dominating Alpha Index",
    title="Dominating Alpha Vector per Sample Volume",
    legend=:topright,
    markersize=4)

yticks = unique(dominating)
ytick_labels = ["a=$(policy.action_map[i])" for i in yticks]

Plots.plot(1:101, dominating,
    xlabel="Sample Volume",
    ylabel="Dominating Alpha Index",
    title="Dominating Alpha Vector per Sample Volume",
    yticks=(yticks, ytick_labels),
    legend=false)

changes = findall(diff(dominating) .!= 0)
vline!(changes, line=:dash, color=:gray)


function plot_alpha_vector_by_state(policy::AlphaVectorPolicy,pomdp,dominating)

    # p = Plots.plot(title="α_$alpha_num (a=$(policy.action_map[alpha_num])) by Life State",
    #             xlabel="Sample Volume", ylabel="Value",
    #             legend=:topright)

    p = Plots.plot(title="alpha vectors by Life State",
            xlabel="Sample Volume", ylabel="Value",
            legend=:topright)
    for dom in dominating
        for α in [policy.alphas[dom]]
            num_samples = pomdp.sampleVolume + 1
            num_life_states = pomdp.lifeStates
            
            for life in 1:num_life_states-1
                values = [α[(sample - 1) * num_life_states + life] for sample in 1:num_samples]
                plot!(1:num_samples, values, label="Life = $life")
            end
        end
    end
    display(p)
    savefig(p, "./figures/alpha_vector_by_state.png")
    return p
end


function plot_pruned_alpha_vectors(policy::AlphaVectorPolicy)

    alpha_vectors = policy.alphas
    num_vectors = size(alpha_vectors, 1)
    b = range(1, 303, length=304)  # Belief in life (P(life))

    # Evaluate each alpha vector over belief points
    V_matrix = Matrix{Float64}(undef, num_vectors, length(b))
    for i in 1:num_vectors
        α = alpha_vectors[i]
        V_matrix[i, :] = α[1] .* b .+ α[2] .* (1 .- b)
    end

    # Prune dominated alpha vectors
    keep = trues(num_vectors)
    for i in 1:num_vectors
        for j in 1:num_vectors
            if i != j && all(V_matrix[j, :] .>= V_matrix[i, :]) && any(V_matrix[j, :] .> V_matrix[i, :])
                keep[i] = false
                break
            end
        end
    end

    pruned_indices = findall(keep)
    println("Pruned from $num_vectors to $(length(pruned_indices)) alpha vectors")

    # Plot
    p = Plots.plot(title="Pruned Alpha Vectors",
             xlabel="Estimated P(life)", ylabel="Value Function", legend=true,
             )

    for i in pruned_indices
        plot!(b, V_matrix[i, :], label="α_$i (a=$(policy.action_map[i]))")
    end

    if !isdir("./figures")
        mkpath("./figures")
    end
    savefig(p, "./figures/pruned_alpha_vectors.png")
    return p
end


function plot_alpha_vectors_VLD(policy::AlphaVectorPolicy, pomdp, sample::Int)
    # get alpha vectors and action map
    alpha_vectors = policy.alphas
    action_map = policy.action_map

    num_vectors = size(alpha_vectors, 1)
    b = range(0, 1, length=200)  # Belief in L=1 (life)
    # b = range(1, pomdp.sample, length=200)  # Belief in L=1 (life)
    # b = range(1, pomdp.sampleVolume*pomdp.lifeStates+pomdp.lifeStates, length=200)  # Belief in L=1 (life)

    p = Plots.plot(title="Alpha Vectors at Sample Volume = $sample",
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
    
    b = range(1, 303, length=100) # P(life)

    # plot each alpha vector
    p = Plots.plot(title="Alpha Vectors",
               xlabel="Estimated P(life)", ylabel="Value Function", legend=true,
               ylims=[-10,0.001])

    for i in 1:num_vectors
        # compute the value function for belief state b
        # V(b) = α₁*b + α₂*(1-b)
        V_b = [alpha_vectors[i][1] * b_i + alpha_vectors[i][2] * (1 - b_i) for b_i in b]
        
        # plot each alpha vector as a line
        # note : if action i isn't present, it means running instrument i was not optimal.
        plot!(b, V_b, label="α_$i (a=$(policy.action_map[i]))")
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

    p= plot()
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