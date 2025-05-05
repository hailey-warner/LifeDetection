using Plots
using GraphPlot
using Graphs
using Compose
using Cairo
using Fontconfig

function plot_alpha_vectors(policy::AlphaVectorPolicy)
    # get alpha vectors
    alpha_vectors = policy.alphas
    num_states = size(alpha_vectors, 2) 
    num_vectors = size(alpha_vectors, 1)
    
    # x-axis represents belief L = 1, or P(life)
    b = range(0, 1, length=100)

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

    display(p)
    savefig(p, "alpha_vectors.png")
    return p
end

